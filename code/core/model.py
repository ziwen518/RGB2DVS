from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        surrogate = (1.0 - x.abs()).clamp_min(0.0)
        return (grad * surrogate,)


def spike(x: torch.Tensor) -> torch.Tensor:
    return SpikeFn.apply(x)


class TernaryFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = float(threshold)
        positive = (x >= threshold).to(x.dtype)
        negative = (x <= -threshold).to(x.dtype)
        return positive - negative

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        threshold = float(ctx.threshold)
        if threshold <= 0.0:
            return grad, None
        normalized = (x / threshold).abs()
        surrogate = (1.0 - normalized).clamp_min(0.0)
        return (grad * surrogate, None)


def ternary(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return TernaryFn.apply(x, float(threshold))


class PLIF(nn.Module):
    def __init__(
        self, threshold: float = 1.0, init_decay: float = 0.5,
        mixed: bool = False, mixed_spread: float = 1.0,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.continuous = False
        self.mixed = bool(mixed)
        self.mixed_spread = float(mixed_spread)
        self.decay_logit = nn.Parameter(torch.logit(torch.tensor(float(init_decay))))

    def forward(self, current: torch.Tensor, membrane: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if membrane is None or membrane.shape != current.shape:
            membrane = torch.zeros_like(current)
        if self.mixed and current.ndim == 4 and current.shape[1] > 1:
            channels = current.shape[1]
            split = channels // 2
            offset = current.new_empty(channels)
            offset[:split] = -self.mixed_spread
            offset[split:] = self.mixed_spread
            decay = torch.sigmoid(self.decay_logit + offset).view(1, channels, 1, 1)
        else:
            decay = torch.sigmoid(self.decay_logit).to(dtype=current.dtype, device=current.device)
        membrane = decay * membrane + current
        out = membrane if self.continuous else spike(membrane - self.threshold)
        if not self.continuous:
            membrane = membrane - out.detach() * self.threshold
        return out, membrane


class TemporalAffine2d(nn.Module):
    """Per-time current calibration before a spiking neuron."""

    def __init__(self, channels: int, max_steps: int = 16) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(max_steps, channels))
        self.bias = nn.Parameter(torch.zeros(max_steps, channels))

    def forward(self, current: torch.Tensor, time_index: int) -> torch.Tensor:
        scale = self.scale[time_index].view(1, -1, 1, 1)
        bias = self.bias[time_index].view(1, -1, 1, 1)
        return current * scale + bias


class TokenNorm(nn.Module):
    def __init__(self, dim: int, kind: str = "batchnorm", max_steps: int = 16) -> None:
        super().__init__()
        if kind == "batchnorm":
            # Keep the historical attribute name for checkpoint compatibility.
            self.bn = nn.BatchNorm1d(dim)
        elif kind == "layernorm":
            self.norm = nn.LayerNorm(dim)
        elif kind == "bntt":
            self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(max_steps)])
        else:
            raise ValueError(f"unsupported normalization: {kind}")
        self.kind = kind

    def forward(self, x: torch.Tensor, time_index: int) -> torch.Tensor:
        if self.kind == "batchnorm":
            return self.bn(x.transpose(1, 2)).transpose(1, 2)
        if self.kind == "bntt":
            if time_index >= len(self.bns):
                raise ValueError(f"time index {time_index} exceeds BNTT capacity {len(self.bns)}")
            return self.bns[time_index](x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)


class SpikePatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        patch_size: int,
        threshold: float,
        norm: str,
        image_size: int = 48,
        local_stem: bool = False,
        use_positional_bias: bool = False,
        population_bins: int = 1,
        use_spatial_pos: bool = False,
        pyramid_stem: bool = False,
        max_steps: int = 16,
    ) -> None:
        super().__init__()
        self.local_stem = bool(local_stem)
        self.use_positional_bias = bool(use_positional_bias)
        self.population_bins = int(population_bins)
        self.use_spatial_pos = bool(use_spatial_pos)
        self.pyramid_stem = bool(pyramid_stem)
        if self.population_bins < 1:
            raise ValueError("population_bins must be positive")
        if self.population_bins > 1:
            self.register_buffer(
                "population_thresholds",
                torch.arange(1, self.population_bins + 1, dtype=torch.float32)
                / float(self.population_bins + 1),
            )
        if self.pyramid_stem:
            widths = (dim // 8, dim // 4, dim // 2, dim)
            self.pyramid_convs = nn.ModuleList()
            self.pyramid_bns = nn.ModuleList()
            self.pyramid_lifs = nn.ModuleList()
            current_channels = in_channels * self.population_bins
            for width in widths:
                self.pyramid_convs.append(nn.Conv2d(
                    current_channels, width, kernel_size=3, padding=1, bias=False
                ))
                self.pyramid_bns.append(nn.ModuleList([
                    nn.BatchNorm2d(width) for _ in range(max_steps)
                ]))
                self.pyramid_lifs.append(PLIF(threshold=threshold))
                current_channels = width
            self.pyramid_pool = nn.MaxPool2d(3, stride=2, padding=1)
            self.rpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
            self.rpe_bns = nn.ModuleList([nn.BatchNorm2d(dim) for _ in range(max_steps)])
            self.rpe_lif = PLIF(threshold=threshold)
            grid = max(1, math.ceil(int(image_size) / 4))
            self.pos_bias = nn.Parameter(torch.zeros(1, dim, grid, grid))
        elif self.local_stem:
            stem_dim = min(64, dim)
            self.local = nn.Conv2d(
                in_channels * self.population_bins, stem_dim,
                kernel_size=3, padding=1, bias=False,
            )
            if norm == "batchnorm":
                self.local_bn = nn.BatchNorm2d(stem_dim)
            elif norm == "layernorm":
                self.local_norm = nn.GroupNorm(1, stem_dim)
            elif norm == "bntt":
                self.local_bns = nn.ModuleList([nn.BatchNorm2d(stem_dim) for _ in range(max_steps)])
            else:
                raise ValueError(f"unsupported normalization: {norm}")
            self.local_lif = PLIF(threshold=threshold)
            self.proj = nn.Conv2d(stem_dim, dim, kernel_size=patch_size, stride=patch_size, bias=False)
            grid = max(1, math.ceil(int(image_size) / int(patch_size)))
            self.pos_bias = nn.Parameter(torch.zeros(1, dim, grid, grid))
        else:
            # This is the checkpoint-compatible SpikeFormer patch embedding.
            self.proj = nn.Conv2d(
                in_channels * self.population_bins, dim,
                kernel_size=patch_size, stride=patch_size, bias=False,
            )
            if self.use_positional_bias:
                grid = max(1, math.ceil(int(image_size) / int(patch_size)))
                self.pos_bias = nn.Parameter(torch.zeros(1, dim, grid, grid))
            if self.use_spatial_pos:
                grid = max(1, math.ceil(int(image_size) / int(patch_size)))
                self.spatial_pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        if norm == "batchnorm":
            # Keep the historical attribute name for checkpoint compatibility.
            self.bn = nn.BatchNorm2d(dim)
        elif norm == "layernorm":
            # GroupNorm has no running statistics and is the spatial analogue
            # of per-token LayerNorm for the convolutional stem.
            self.norm = nn.GroupNorm(1, dim)
        elif norm == "bntt":
            self.bns = nn.ModuleList([nn.BatchNorm2d(dim) for _ in range(max_steps)])
        else:
            raise ValueError(f"unsupported normalization: {norm}")
        self.norm_kind = norm
        self.lif = PLIF(threshold=threshold)

    def forward(
        self,
        x: torch.Tensor,
        membrane: torch.Tensor | None,
        local_membrane: torch.Tensor | None,
        time_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, object | None, tuple[int, int]]:
        if self.population_bins > 1:
            thresholds = self.population_thresholds.to(device=x.device, dtype=x.dtype)
            x = (x.unsqueeze(2) >= thresholds.view(1, 1, -1, 1, 1)).to(x.dtype)
            x = x.flatten(1, 2)
        if self.pyramid_stem:
            states = local_membrane if isinstance(local_membrane, dict) else {}
            hidden = x
            next_states: dict[str, torch.Tensor] = {}
            for index, (conv, bns, lif) in enumerate(zip(
                self.pyramid_convs, self.pyramid_bns, self.pyramid_lifs
            )):
                current = bns[time_index](conv(hidden))
                hidden, next_states[f"stage_{index}"] = lif(
                    current, states.get(f"stage_{index}")
                )
                if index in (0, 1):
                    hidden = self.pyramid_pool(hidden)
            rpe, next_states["rpe"] = self.rpe_lif(
                self.rpe_bns[time_index](self.rpe(hidden)), states.get("rpe")
            )
            current = hidden + rpe
            local_membrane = next_states
        elif self.local_stem:
            local = self.local(x)
            if self.norm_kind == "batchnorm":
                local = self.local_bn(local)
            elif self.norm_kind == "bntt":
                local = self.local_bns[time_index](local)
            else:
                local = self.local_norm(local)
            local, local_membrane = self.local_lif(local, local_membrane)
            current = self.proj(local)
        else:
            current = self.proj(x)
            if self.use_positional_bias:
                pos = self.pos_bias
                if pos.shape[-2:] != current.shape[-2:]:
                    pos = F.interpolate(pos, size=current.shape[-2:], mode="bilinear", align_corners=False)
                current = current + pos.to(device=current.device, dtype=current.dtype)
        if self.norm_kind == "batchnorm":
            current = self.bn(current)
        elif self.norm_kind == "bntt":
            if time_index >= len(self.bns):
                raise ValueError(f"time index {time_index} exceeds BNTT capacity {len(self.bns)}")
            current = self.bns[time_index](current)
        else:
            current = self.norm(current)
        if self.use_spatial_pos:
            spatial = self.spatial_pos
            if spatial.shape[1] != current.shape[-2] * current.shape[-1]:
                spatial = F.interpolate(
                    spatial.transpose(1, 2).reshape(1, -1, int(math.sqrt(spatial.shape[1])), int(math.sqrt(spatial.shape[1]))),
                    size=current.shape[-2:], mode="bilinear", align_corners=False,
                ).flatten(2).transpose(1, 2)
            spatial = spatial.transpose(1, 2).reshape(
                1, current.shape[1], current.shape[-2], current.shape[-1]
            )
            current = current + spatial.to(device=current.device, dtype=current.dtype)
        # Inject spatial identity after normalization so it is not erased by
        # BNTT/BN, then convert it to spikes at the patch LIF.
        if self.use_positional_bias:
            pos = self.pos_bias
            if pos.shape[-2:] != current.shape[-2:]:
                pos = F.interpolate(pos, size=current.shape[-2:], mode="bilinear", align_corners=False)
            current = current + pos.to(device=current.device, dtype=current.dtype)
        out, membrane = self.lif(current, membrane)
        h, w = out.shape[-2:]
        return out.flatten(2).transpose(1, 2), membrane, local_membrane, (h, w)


class SpikeSelfAttention(nn.Module):
    def __init__(
        self, dim: int, heads: int, threshold: float, norm: str,
        attention_mode: str = "count", compact_output: bool = False,
        hybrid_attention: bool = False, ternary_threshold: float = 0.5,
        qkv_temporal_mode: str = "standard",
        max_steps: int = 16,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = int(heads)
        self.head_dim = dim // heads
        if attention_mode not in {"count", "normalized"}:
            raise ValueError(f"unsupported spike attention mode: {attention_mode}")
        self.attention_mode = attention_mode
        temporal_modes = {
            "standard", "role_decay", "causal_kv",
            "temporal_context", "event_gate", "membrane_aware",
            "multiscale", "frequency", "delta", "motion_guided",
            "state_fusion", "mesa",
        }
        if qkv_temporal_mode not in temporal_modes:
            raise ValueError(f"unsupported QKV temporal mode: {qkv_temporal_mode}")
        if qkv_temporal_mode == "causal_kv" and attention_mode != "normalized":
            raise ValueError("causal_kv requires normalized spike attention")
        self.qkv_temporal_mode = qkv_temporal_mode
        self.compact_output = bool(compact_output)
        self.hybrid_attention = bool(hybrid_attention)
        self.ternary_threshold = float(ternary_threshold)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bn = TokenNorm(dim, norm, max_steps)
        self.k_bn = TokenNorm(dim, norm, max_steps)
        self.v_bn = TokenNorm(dim, norm, max_steps)
        # MESA starts from the standard stable membrane dynamics; its event
        # gate is learned as a residual instead of inheriting extreme role
        # decays used by the older temporal-QKV ablations.
        role_temporal = qkv_temporal_mode not in {"standard", "mesa"}
        self.q_lif = PLIF(threshold, init_decay=0.25 if role_temporal else 0.5)
        self.k_lif = PLIF(threshold, init_decay=0.8 if role_temporal else 0.5)
        self.v_lif = PLIF(threshold, init_decay=0.5)
        if qkv_temporal_mode == "causal_kv":
            initial_decay = torch.linspace(0.55, 0.9, self.heads)
            self.kv_decay_logit = nn.Parameter(torch.logit(initial_decay))
        else:
            self.register_parameter("kv_decay_logit", None)
        if qkv_temporal_mode == "temporal_context":
            self.context_logits = nn.Parameter(torch.logit(torch.tensor([0.5, 0.25])))
        else:
            self.register_parameter("context_logits", None)
        if qkv_temporal_mode == "event_gate":
            self.event_gate_ratio_logit = nn.Parameter(torch.logit(torch.tensor(0.5)))
        else:
            self.register_parameter("event_gate_ratio_logit", None)
        if qkv_temporal_mode == "membrane_aware":
            self.membrane_gain = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("membrane_gain", None)
        if qkv_temporal_mode == "multiscale":
            self.multiscale_q_logits = nn.Parameter(torch.tensor([1.5, 0.5, 0.0, -0.5]))
            self.multiscale_k_logits = nn.Parameter(torch.tensor([0.5, 1.0, 0.5, 0.0]))
        else:
            self.register_parameter("multiscale_q_logits", None)
            self.register_parameter("multiscale_k_logits", None)
        if qkv_temporal_mode == "motion_guided":
            self.motion_gain = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("motion_gain", None)
        if qkv_temporal_mode == "state_fusion":
            self.fusion_q_logits = nn.Parameter(torch.tensor([1.5, 0.0, 0.5, 0.0, 0.0]))
            self.fusion_k_logits = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0]))
        else:
            self.register_parameter("fusion_q_logits", None)
            self.register_parameter("fusion_k_logits", None)
        if qkv_temporal_mode == "mesa":
            # MESA: event evidence gates how much membrane state is trusted
            # when forming Q/K; V remains an instantaneous spike projection.
            # Zero-init the residual path so MESA starts exactly as the
            # proven standard spike attention and cannot collapse at epoch 1.
            self.mesa_q_membrane_logit = nn.Parameter(torch.tensor(0.0))
            self.mesa_k_membrane_logit = nn.Parameter(torch.tensor(0.0))
            self.mesa_q_event_logit = nn.Parameter(torch.tensor(0.0))
            self.mesa_k_event_logit = nn.Parameter(torch.tensor(0.0))
            self.mesa_event_vector = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("mesa_q_membrane_logit", None)
            self.register_parameter("mesa_k_membrane_logit", None)
            self.register_parameter("mesa_q_event_logit", None)
            self.register_parameter("mesa_k_event_logit", None)
            self.register_parameter("mesa_event_vector", None)
        self.attn_lif = PLIF(threshold=0.5)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_bn = TokenNorm(dim, norm, max_steps)
        self.out_lif = PLIF(threshold)

    def forward(self, x: torch.Tensor, state: dict[str, torch.Tensor | None], time_index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor | None], torch.Tensor]:
        b, n, c = x.shape
        q_source = k_source = v_source = x
        mode = self.qkv_temporal_mode
        previous = state.get("input")
        if mode == "temporal_context":
            previous2 = state.get("input2")
            coefficients = torch.sigmoid(self.context_logits).to(x)
            q_source = k_source = x
            if previous is not None:
                q_source = k_source = q_source + coefficients[0] * previous
            if previous2 is not None:
                q_source = k_source = q_source + coefficients[1] * previous2
            state["input2"] = previous
            state["input"] = x
        elif mode == "membrane_aware":
            membrane = state.get("input_membrane")
            if membrane is not None:
                membrane_feature = F.layer_norm(membrane.float(), (c,)).to(x)
                q_source = k_source = x + torch.tanh(self.membrane_gain).to(x) * membrane_feature
        elif mode in {"multiscale", "state_fusion"}:
            fast = x if state.get("fast") is None else 0.25 * state["fast"] + 0.75 * x
            middle = x if state.get("middle") is None else 0.6 * state["middle"] + 0.4 * x
            slow = x if state.get("slow") is None else 0.85 * state["slow"] + 0.15 * x
            state.update({"fast": fast, "middle": middle, "slow": slow})
            if mode == "multiscale":
                features = torch.stack([x, fast, middle, slow], dim=0)
                q_source = (F.softmax(self.multiscale_q_logits, dim=0).to(x).view(-1, 1, 1, 1) * features).sum(0)
                k_source = (F.softmax(self.multiscale_k_logits, dim=0).to(x).view(-1, 1, 1, 1) * features).sum(0)
            else:
                membrane = state.get("input_membrane")
                membrane_feature = (
                    torch.zeros_like(x) if membrane is None
                    else F.layer_norm(membrane.float(), (c,)).to(x)
                )
                delta = x if previous is None else x - previous
                motion = self._motion_feature(delta)
                features = torch.stack([x, membrane_feature, fast, slow, motion], dim=0)
                q_source = (F.softmax(self.fusion_q_logits, dim=0).to(x).view(-1, 1, 1, 1) * features).sum(0)
                k_source = (F.softmax(self.fusion_k_logits, dim=0).to(x).view(-1, 1, 1, 1) * features).sum(0)
                state["input"] = x
        elif mode == "mesa":
            membrane = state.get("input_membrane")
            membrane_feature = (
                torch.zeros_like(x) if membrane is None
                else F.layer_norm(membrane.float(), (c,)).to(x)
            )
            density = x.float().mean(dim=-1, keepdim=True)
            previous_density = state.get("density")
            if previous_density is None:
                novelty = density
            else:
                novelty = density - previous_density
            state["density"] = 0.8 * density.detach() + 0.2 * (
                previous_density if previous_density is not None else density.detach()
            )
            # A token with fresh event evidence can rely on its accumulated
            # membrane; stale/quiet tokens fall back toward instantaneous spikes.
            q_gate = torch.sigmoid(
                torch.tanh(self.mesa_q_event_logit).to(x) * density
            )
            k_gate = torch.sigmoid(
                torch.tanh(self.mesa_k_event_logit).to(x) * novelty.abs()
            )
            event_feature = density * self.mesa_event_vector.to(x)
            residual_scale = x.new_tensor(0.1)
            q_source = x + residual_scale * q_gate * torch.tanh(
                self.mesa_q_membrane_logit
            ).to(x) * membrane_feature + residual_scale * event_feature
            k_source = x + residual_scale * k_gate * torch.tanh(
                self.mesa_k_membrane_logit
            ).to(x) * membrane_feature + residual_scale * event_feature
        elif mode == "frequency":
            count = float(time_index + 1)
            frequency = x if previous is None else previous + (x - previous) / count
            q_source = k_source = frequency
            state["input"] = frequency
        elif mode == "delta":
            q_source = k_source = x if previous is None else x - previous
            state["input"] = x
        elif mode == "motion_guided":
            delta = x if previous is None else x - previous
            motion = self._motion_feature(delta)
            gain = torch.tanh(self.motion_gain).to(x)
            q_source = x + gain * motion
            k_source = x - gain * motion
            state["input"] = x

        q_weight, k_weight, v_weight = self.qkv.weight.split(c, dim=0)
        q = F.linear(q_source, q_weight)
        k = F.linear(k_source, k_weight)
        v = F.linear(v_source, v_weight)
        if self.hybrid_attention:
            q, state["q"] = self.q_lif(self.q_bn(q, time_index), state.get("q"))
            k = F.relu(self.k_bn(k, time_index))
            v = ternary(self.v_bn(v, time_index), self.ternary_threshold)
        else:
            q, state["q"] = self.q_lif(self.q_bn(q, time_index), state.get("q"))
            k, state["k"] = self.k_lif(self.k_bn(k, time_index), state.get("k"))
            v, state["v"] = self.v_lif(self.v_bn(v, time_index), state.get("v"))
        if mode == "event_gate":
            density = x.float().mean(dim=-1, keepdim=True)
            reference = density.mean(dim=1, keepdim=True)
            ratio = torch.sigmoid(self.event_gate_ratio_logit).to(density)
            gate = spike(density - ratio * reference).to(q)
            q, k, v = q * gate, k * gate, v * gate
        qh = q.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        kh = k.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        vh = v.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        if self.qkv_temporal_mode == "causal_kv":
            instant_context = (kh.transpose(-2, -1) @ vh) / float(max(1, n))
            decay = torch.sigmoid(self.kv_decay_logit).to(instant_context).view(1, self.heads, 1, 1)
            previous_context = state.get("kv")
            context_trace = (
                instant_context
                if previous_context is None
                else decay * previous_context + instant_context
            )
            mass = (1.0 - decay.pow(time_index + 1)) / (1.0 - decay).clamp_min(1.0e-4)
            current = (qh @ (context_trace / mass)) / math.sqrt(self.head_dim)
            state["kv"] = context_trace
        elif self.attention_mode == "count":
            current = (qh @ kh.transpose(-2, -1)) @ vh
            current = current * 0.25
        else:
            context = (kh.transpose(-2, -1) @ vh) / float(max(1, n))
            current = (qh @ context) / math.sqrt(self.head_dim)
        current = current.transpose(1, 2).reshape(b, n, c)
        if self.compact_output:
            # Projection is synaptic current; the residual PLIF emits the only
            # post-attention spike communicated to the MLP.
            out = self.proj_bn(self.proj(current), time_index)
            rate = torch.stack([q.mean(), k.mean(), v.mean()]).mean()
        else:
            current, state["attn"] = self.attn_lif(current, state.get("attn"))
            current = self.proj_bn(self.proj(current), time_index)
            out, state["out"] = self.out_lif(current, state.get("out"))
            rate = torch.stack([q.mean(), k.mean(), v.mean(), out.mean()]).mean()
        return out, state, rate

    @staticmethod
    def _motion_feature(delta: torch.Tensor) -> torch.Tensor:
        tokens = delta
        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            # Preserve a possible CLS token while applying motion to the grid.
            side = math.isqrt(tokens.shape[1] - 1)
            if side * side != tokens.shape[1] - 1:
                return torch.zeros_like(tokens)
            prefix, tokens = tokens[:, :1], tokens[:, 1:]
        else:
            prefix = None
        grid = tokens.reshape(tokens.shape[0], side, side, tokens.shape[-1])
        horizontal = torch.roll(grid, -1, dims=2) - torch.roll(grid, 1, dims=2)
        vertical = torch.roll(grid, -1, dims=1) - torch.roll(grid, 1, dims=1)
        motion = 0.5 * (horizontal + vertical).reshape_as(tokens)
        return motion if prefix is None else torch.cat([torch.zeros_like(prefix), motion], dim=1)


class SpikeMLP(nn.Module):
    def __init__(
        self, dim: int, ratio: float, threshold: float, norm: str,
        compact_output: bool = False, max_steps: int = 16,
    ) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.bn1 = TokenNorm(hidden, norm, max_steps)
        self.lif1 = PLIF(threshold)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.bn2 = TokenNorm(dim, norm, max_steps)
        self.lif2 = PLIF(threshold)
        self.compact_output = bool(compact_output)

    def forward(self, x: torch.Tensor, state: dict[str, torch.Tensor | None], time_index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor | None], torch.Tensor]:
        hidden, state["h"] = self.lif1(self.bn1(self.fc1(x), time_index), state.get("h"))
        current = self.bn2(self.fc2(hidden), time_index)
        if self.compact_output:
            # The block-final residual PLIF converts this current to spikes.
            return current, state, hidden.mean()
        out, state["o"] = self.lif2(current, state.get("o"))
        return out, state, 0.5 * (hidden.mean() + out.mean())


class EventObservableLocalMixer(nn.Module):
    """Sparse multi-scale local relay driven only by event spike evidence.

    Global spike attention is weak at preserving small boundaries. This branch
    relays local structure only for tokens whose activity or temporal novelty
    exceeds the sample mean, then re-spikes the relay before it is fused with
    attention. No image, teacher, or class signal is required at inference.
    """

    def __init__(
        self, dim: int, threshold: float, norm: str,
        dilations: tuple[int, ...] = (1, 2), max_steps: int = 16,
    ) -> None:
        super().__init__()
        if not dilations or any(int(dilation) < 1 for dilation in dilations):
            raise ValueError("local mixer dilations must be positive")
        self.dilations = tuple(int(dilation) for dilation in dilations)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                dim, dim, kernel_size=3, padding=dilation,
                dilation=dilation, groups=dim, bias=False,
            )
            for dilation in self.dilations
        ])
        self.norms = nn.ModuleList([
            TokenNorm(dim, norm, max_steps) for _ in self.dilations
        ])
        self.lifs = nn.ModuleList([
            PLIF(threshold, init_decay=0.6) for _ in self.dilations
        ])
        self.mix_logits = nn.Parameter(torch.zeros(len(self.dilations)))
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_norm = TokenNorm(dim, norm, max_steps)
        self.out_lif = PLIF(threshold)
        self.route_bias = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _spatial_tokens(x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor, int]:
        side = math.isqrt(x.shape[1])
        if side * side == x.shape[1]:
            return None, x, side
        side = math.isqrt(x.shape[1] - 1)
        if side * side != x.shape[1] - 1:
            raise ValueError("local mixer requires square patch tokens")
        return x[:, :1], x[:, 1:], side

    def forward(
        self, x: torch.Tensor, state: dict[str, object] | None, time_index: int,
    ) -> tuple[torch.Tensor, dict[str, object], torch.Tensor]:
        if state is None:
            state = {}
        prefix, tokens, side = self._spatial_tokens(x)
        previous = state.get("previous")
        novelty = (
            tokens if not torch.is_tensor(previous)
            else (tokens - previous).abs()
        )
        evidence = 0.5 * tokens.float().mean(dim=-1, keepdim=True)
        evidence = evidence + 0.5 * novelty.float().mean(dim=-1, keepdim=True)
        centered = evidence - evidence.mean(dim=1, keepdim=True)
        route = spike(centered + 0.25 * torch.tanh(self.route_bias).to(centered))

        feature_map = tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], side, side
        )
        branch_states = state.get("branches")
        if not isinstance(branch_states, list):
            branch_states = [None] * len(self.convs)
        branches = []
        next_branch_states = []
        for conv, branch_norm, branch_lif, membrane in zip(
            self.convs, self.norms, self.lifs, branch_states
        ):
            current = conv(feature_map).flatten(2).transpose(1, 2)
            branch, membrane = branch_lif(
                branch_norm(current, time_index),
                membrane if torch.is_tensor(membrane) else None,
            )
            branches.append(branch)
            next_branch_states.append(membrane)
        weights = F.softmax(self.mix_logits, dim=0).to(tokens)
        mixed = sum(weight * branch for weight, branch in zip(weights, branches))
        mixed = mixed * route
        out, out_membrane = self.out_lif(
            self.proj_norm(self.proj(mixed), time_index),
            state.get("out") if torch.is_tensor(state.get("out")) else None,
        )
        if prefix is not None:
            out = torch.cat([torch.zeros_like(prefix), out], dim=1)
        next_state: dict[str, object] = {
            "branches": next_branch_states,
            "out": out_membrane,
            "previous": tokens,
        }
        return out, next_state, 0.5 * (route.mean() + out.mean())


class PureSpikeBlock(nn.Module):
    """All inter-block activations are binary; residual sums are re-spiked."""

    def __init__(
        self, dim: int, heads: int, mlp_ratio: float, threshold: float,
        norm: str, attention_mode: str = "count",
        block_spiking: str = "full_respike",
        hybrid_attention: bool = False,
        ternary_threshold: float = 0.5,
        qkv_temporal_mode: str = "standard",
        local_structure_mixer: bool = False,
        local_mixer_dilations: tuple[int, ...] = (1, 2),
        max_steps: int = 16,
    ) -> None:
        super().__init__()
        if block_spiking not in {"full_respike", "compact_residual"}:
            raise ValueError(f"unsupported block spiking mode: {block_spiking}")
        compact_output = block_spiking == "compact_residual"
        self.attn = SpikeSelfAttention(
            dim, heads, threshold, norm, attention_mode=attention_mode,
            compact_output=compact_output, hybrid_attention=hybrid_attention,
            ternary_threshold=ternary_threshold,
            qkv_temporal_mode=qkv_temporal_mode, max_steps=max_steps,
        )
        self.attn_res_norm = TokenNorm(dim, norm, max_steps)
        self.attn_res_lif = PLIF(threshold)
        self.local_mixer = (
            EventObservableLocalMixer(
                dim, threshold, norm,
                dilations=local_mixer_dilations, max_steps=max_steps,
            )
            if local_structure_mixer else None
        )
        self.local_gain_logit = (
            nn.Parameter(torch.logit(torch.tensor(0.25)))
            if local_structure_mixer else None
        )
        self.mlp = SpikeMLP(
            dim, mlp_ratio, threshold, norm, compact_output=compact_output,
            max_steps=max_steps,
        )
        self.mlp_res_norm = TokenNorm(dim, norm, max_steps)
        self.mlp_res_lif = PLIF(threshold)

    def forward(self, x: torch.Tensor, state: dict[str, object] | None, time_index: int) -> tuple[torch.Tensor, dict[str, object], torch.Tensor]:
        if state is None:
            state = {"attn": {}, "mlp": {}, "ar": None, "mr": None}
        attn_state = state["attn"] if isinstance(state.get("attn"), dict) else {}
        mlp_state = state["mlp"] if isinstance(state.get("mlp"), dict) else {}
        input_membrane = state.get("mr")
        attn_state["input_membrane"] = input_membrane if torch.is_tensor(input_membrane) else None
        y, attn_state, r1 = self.attn(x, attn_state, time_index)  # type: ignore[arg-type]
        local_state = state.get("local") if isinstance(state.get("local"), dict) else None
        if self.local_mixer is not None:
            local, local_state, local_rate = self.local_mixer(x, local_state, time_index)
            y = y + torch.sigmoid(self.local_gain_logit).to(y) * local
            r1 = 0.5 * (r1 + local_rate)
        x, ar = self.attn_res_lif(
            self.attn_res_norm(x + y, time_index),
            state.get("ar") if torch.is_tensor(state.get("ar")) else None,
        )
        y, mlp_state, r2 = self.mlp(x, mlp_state, time_index)  # type: ignore[arg-type]
        x, mr = self.mlp_res_lif(
            self.mlp_res_norm(x + y, time_index),
            state.get("mr") if torch.is_tensor(state.get("mr")) else None,
        )
        return x, {
            "attn": attn_state, "local": local_state,
            "mlp": mlp_state, "ar": ar, "mr": mr,
        }, 0.5 * (r1 + r2)


class EventConditionedSignedPrompt(nn.Module):
    """Temporal event prompt with signed binary routing.

    The prompt is trained as a modality adapter, but the output communicated
    to the first transformer block is re-spiked. Positive prompt spikes can
    recover event-supported quiet tokens and negative prompt spikes suppress
    stale activity without introducing ANN activations between blocks.
    """

    def __init__(self, dim: int, threshold: float, norm: str,
                 max_steps: int = 16, strength: float = 0.7) -> None:
        super().__init__()
        self.strength = float(strength)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.route = nn.Linear(dim, dim, bias=False)
        self.time_bias = nn.Parameter(torch.zeros(max_steps, dim))
        self.norm = TokenNorm(dim, norm, max_steps)
        self.lif = PLIF(threshold, init_decay=0.7)
        self.threshold = float(threshold)

    def forward(
        self, x: torch.Tensor, membrane: torch.Tensor | None, time_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        density = x.float().mean(dim=-1, keepdim=True)
        centered = density - density.mean(dim=1, keepdim=True)
        route = spike(centered)
        current = self.proj(x) + self.route(x * route)
        current = current + self.time_bias[time_index].view(1, 1, -1).to(current)
        current = self.norm(current, time_index)
        prompt_state, membrane = self.lif(current, membrane)
        # Ternary current carries the sign; the state spike gates its strength.
        signed = ternary(current, threshold=max(0.25, 0.5 * self.threshold))
        signed = signed * (0.5 + 0.5 * prompt_state)
        mixed = spike(x + self.strength * signed - 0.5 * self.threshold)
        return mixed, membrane, route.mean(), prompt_state.mean()


class PureSpikeFormer(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        dim: int = 384,
        depth: int = 8,
        heads: int = 6,
        patch_size: int = 6,
        mlp_ratio: float = 4.0,
        threshold: float = 1.0,
        use_cls_token: bool = False,
        norm: str = "batchnorm",
        image_size: int = 48,
        temporal_readout: str = "uniform",
        local_stem: bool = False,
        use_positional_bias: bool = False,
        population_bins: int = 1,
        use_spatial_pos: bool = False,
        pyramid_stem: bool = False,
        attention_mode: str = "count",
        block_spiking: str = "full_respike",
        hybrid_attention: bool = False,
        hybrid_attention_suffix: int = 0,
        ternary_threshold: float = 0.5,
        qkv_temporal_mode: str = "standard",
        continuous: bool = False,
        signed_readout: bool = False,
        multidepth_readout: bool = False,
        multidepth_readout_layers: int = 3,
        membrane_readout: bool = False,
        temporal_steps: int = 16,
        event_prompt: bool = False,
        prompt_strength: float = 0.7,
        local_structure_mixer: bool = False,
        local_mixer_dilations: tuple[int, ...] = (1, 2),
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.continuous = bool(continuous)
        self.use_cls_token = bool(use_cls_token)
        if temporal_readout not in {"uniform", "learned"}:
            raise ValueError(f"unsupported temporal readout: {temporal_readout}")
        self.temporal_readout = temporal_readout
        self.signed_readout = bool(signed_readout)
        self.multidepth_readout = bool(multidepth_readout)
        self.multidepth_readout_layers = max(1, min(int(multidepth_readout_layers), int(depth)))
        self.membrane_readout = bool(membrane_readout)
        self.event_prompt = bool(event_prompt)
        self.temporal_logits = nn.Parameter(torch.zeros(temporal_steps)) if temporal_readout == "learned" else None
        self.patch = SpikePatchEmbedding(
            in_channels, dim, patch_size, threshold, norm,
            image_size=image_size, local_stem=local_stem,
            use_positional_bias=use_positional_bias,
            population_bins=population_bins,
            use_spatial_pos=use_spatial_pos,
            pyramid_stem=pyramid_stem,
            max_steps=temporal_steps,
        )
        if self.use_cls_token:
            self.cls_bias = nn.Parameter(torch.full((1, 1, dim), 0.25 * float(threshold)))
            self.cls_gain = nn.Parameter(torch.tensor(2.0))
            self.cls_lif = PLIF(threshold=threshold)
        hybrid_attention_suffix = int(hybrid_attention_suffix)
        self.blocks = nn.ModuleList([
            PureSpikeBlock(
                dim, heads, mlp_ratio, threshold, norm,
                attention_mode=attention_mode,
                block_spiking=block_spiking,
                hybrid_attention=(
                    hybrid_attention and (
                        hybrid_attention_suffix <= 0
                        or index >= depth - hybrid_attention_suffix
                    )
                ),
                ternary_threshold=ternary_threshold,
                qkv_temporal_mode=qkv_temporal_mode,
                local_structure_mixer=local_structure_mixer,
                local_mixer_dilations=local_mixer_dilations,
                max_steps=temporal_steps,
            )
            for index in range(depth)
        ])
        if self.event_prompt:
            self.event_prompt_layer = EventConditionedSignedPrompt(
                dim, threshold, norm, max_steps=temporal_steps,
                strength=prompt_strength,
            )
        else:
            self.event_prompt_layer = None
        if self.signed_readout:
            if self.multidepth_readout:
                self.semantic_projs = nn.ModuleList([
                    nn.Linear(dim, 2 * dim, bias=False)
                    for _ in range(self.multidepth_readout_layers)
                ])
                self.multidepth_logits = nn.Parameter(
                    torch.zeros(self.multidepth_readout_layers)
                )
            else:
                self.semantic_proj = nn.Linear(dim, 2 * dim, bias=False)
            self.semantic_norm = TokenNorm(2 * dim, norm, temporal_steps)
            self.semantic_lif = PLIF(threshold)
            if self.membrane_readout:
                self.membrane_readout_proj = nn.Linear(2 * dim, 2 * dim, bias=False)
                self.membrane_readout_gain = nn.Parameter(torch.tensor(0.0))
        if self.continuous:
            for module in self.modules():
                if isinstance(module, PLIF):
                    module.continuous = True

    def forward(
        self, events: torch.Tensor, return_tokens: bool = False,
        return_layers: bool = False, return_membranes: bool = False,
        return_token_layers: int = 0,
    ) -> dict[str, torch.Tensor]:
        if events.ndim != 5:
            raise ValueError(f"expected [B,T,C,H,W], got {tuple(events.shape)}")
        patch_membrane = None
        local_membrane = None
        cls_membrane = None
        states: list[dict[str, object] | None] = [None] * len(self.blocks)
        trajectory: list[torch.Tensor] = []
        core_trajectory: list[torch.Tensor] = []
        token_steps: list[torch.Tensor] = []
        token_layer_count = max(0, min(int(return_token_layers), len(self.blocks)))
        token_layer_steps: list[list[torch.Tensor]] | None = (
            [[] for _ in range(token_layer_count)] if token_layer_count else None
        )
        semantic_rate_steps: list[torch.Tensor] = []
        rates: list[torch.Tensor] = []
        semantic_membrane = None
        layer_steps: list[list[torch.Tensor]] | None = (
            [[] for _ in range(len(self.blocks) + 1)] if return_layers else None
        )
        membrane_steps: list[list[torch.Tensor]] | None = (
            [[] for _ in range(len(self.blocks))] if return_membranes else None
        )
        prompt_membrane = None
        prompt_rates: list[torch.Tensor] = []
        prompt_route_rates: list[torch.Tensor] = []
        for t in range(events.shape[1]):
            x, patch_membrane, local_membrane, _ = self.patch(
                events[:, t], patch_membrane, local_membrane, t
            )
            if self.event_prompt_layer is not None:
                x, prompt_membrane, route_rate, prompt_rate = self.event_prompt_layer(
                    x, prompt_membrane, t
                )
                prompt_route_rates.append(route_rate)
                prompt_rates.append(prompt_rate)
            if self.use_cls_token:
                cls_current = self.cls_bias + self.cls_gain.clamp(0.5, 4.0) * x.mean(dim=1, keepdim=True)
                cls, cls_membrane = self.cls_lif(cls_current, cls_membrane)
                x = torch.cat([cls, x], dim=1)
            if layer_steps is not None:
                layer_steps[0].append(
                    x[:, 0] if self.use_cls_token else x.mean(dim=1)
                )
            step_rates = [x.mean()]
            readout_sources: list[torch.Tensor] = []
            for i, block in enumerate(self.blocks):
                x, states[i], rate = block(x, states[i], t)
                step_rates.append(rate)
                if self.multidepth_readout and i >= len(self.blocks) - self.multidepth_readout_layers:
                    readout_sources.append(x)
                if layer_steps is not None:
                    layer_steps[i + 1].append(
                        x[:, 0] if self.use_cls_token else x.mean(dim=1)
                    )
                if membrane_steps is not None:
                    residual_membrane = states[i]["mr"]
                    if not torch.is_tensor(residual_membrane):
                        raise RuntimeError("missing block residual membrane")
                    membrane_steps[i].append(
                        residual_membrane[:, 0]
                        if self.use_cls_token else residual_membrane.mean(dim=1)
                    )
                if token_layer_steps is not None and i >= len(self.blocks) - token_layer_count:
                    token_layer_steps[i - (len(self.blocks) - token_layer_count)].append(
                        x[:, 1:] if self.use_cls_token else x
                    )
            core_trajectory.append(
                x[:, 0] if self.use_cls_token else x.mean(dim=1)
            )
            if self.signed_readout:
                if self.multidepth_readout:
                    weights = F.softmax(self.multidepth_logits, dim=0).to(x)
                    semantic_current = sum(
                        weight * projection(source)
                        for weight, projection, source in zip(
                            weights, self.semantic_projs, readout_sources
                        )
                    )
                else:
                    semantic_current = self.semantic_proj(x)
                if self.membrane_readout and semantic_membrane is not None:
                    membrane_context = F.layer_norm(
                        semantic_membrane.float(), (semantic_membrane.shape[-1],)
                    ).to(semantic_current)
                    membrane_modulation = torch.tanh(
                        self.membrane_readout_proj(membrane_context)
                    )
                    # Bounded multiplicative modulation keeps the membrane
                    # path SNN-native without allowing one update to erase
                    # the signed spike readout geometry.
                    semantic_current = semantic_current * (
                        1.0 + 0.05 * torch.tanh(self.membrane_readout_gain).to(
                            semantic_current
                        ) * membrane_modulation
                    )
                semantic_current = self.semantic_norm(semantic_current, t)
                semantic_spikes, semantic_membrane = self.semantic_lif(
                    semantic_current, semantic_membrane
                )
                positive, negative = semantic_spikes.chunk(2, dim=-1)
                semantic_rate_steps.append(semantic_spikes.mean(dim=1))
                semantic_step = positive - negative
                trajectory.append(
                    semantic_step[:, 0] if self.use_cls_token
                    else semantic_step.mean(dim=1)
                )
                step_rates.append(semantic_spikes.mean())
            else:
                trajectory.append(x[:, 0] if self.use_cls_token else x.mean(dim=1))
            if return_tokens:
                token_steps.append(x)
            rates.append(torch.stack(step_rates).mean())
        step_embeddings = torch.stack(trajectory, dim=1)
        core_steps = torch.stack(core_trajectory, dim=1)
        prefix = step_embeddings.cumsum(dim=1) / torch.arange(1, events.shape[1] + 1, device=events.device).view(1, -1, 1)
        core_prefix = core_steps.cumsum(dim=1) / torch.arange(1, events.shape[1] + 1, device=events.device).view(1, -1, 1)
        if self.temporal_readout == "learned":
            weights = F.softmax(self.temporal_logits[: events.shape[1]], dim=0).to(step_embeddings)
            embedding = (step_embeddings * weights.view(1, -1, 1)).sum(dim=1)
        else:
            embedding = prefix[:, -1]
        result = {
            "core_embedding": F.normalize(core_prefix[:, -1].float(), dim=-1),
            "trajectory": F.normalize(step_embeddings.float(), dim=-1),
            # Keep the signed spike-rate trajectory for rate-coded temporal
            # distillation; the normalized trajectory remains the eval API.
            "raw_trajectory": step_embeddings.float(),
            "semantic_rate_trajectory": torch.stack(semantic_rate_steps, dim=1),
            "prefix": F.normalize(prefix.float(), dim=-1),
            "embedding": F.normalize(embedding.float(), dim=-1),
            "spike_rate": torch.stack(rates).mean(),
            "prompt_rate": torch.stack(prompt_rates).mean() if prompt_rates else events.new_zeros(()),
            "route_rate": torch.stack(prompt_route_rates).mean() if prompt_route_rates else events.new_zeros(()),
        }
        if return_tokens:
            tokens = torch.stack(token_steps, dim=1)
            # Patch supervision excludes the optional CLS token.
            result["tokens"] = tokens[:, :, 1:] if self.use_cls_token else tokens
            result["token_saliency"] = tokens.float().mean(dim=-1)
        if token_layer_steps is not None:
            # [B,L,T,N,C], with shallow-to-deep ordering. Every stored value is
            # a post-residual spike tensor, so OSP adds no continuous bypass.
            result["token_pyramid"] = torch.stack([
                torch.stack(steps, dim=1) for steps in token_layer_steps
            ], dim=1)
        if layer_steps is not None:
            result["layer_embeddings"] = torch.stack([
                F.normalize(torch.stack(steps, dim=1).float().mean(dim=1), dim=-1)
                for steps in layer_steps
            ], dim=1)
        if membrane_steps is not None:
            result["membrane_embeddings"] = torch.stack([
                F.normalize(torch.stack(steps, dim=1).float().mean(dim=1), dim=-1)
                for steps in membrane_steps
            ], dim=1)
        return result


class SpikeConvBlock(nn.Module):
    """Residual convolutional block with binary inter-layer communication."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1,
        temporal_calibration: bool = False,
        mixed_lif: bool = False,
        mixed_lif_spread: float = 1.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.calibration1 = TemporalAffine2d(out_channels) if temporal_calibration else nn.Identity()
        self.lif1 = PLIF(mixed=mixed_lif, mixed_spread=mixed_lif_spread)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.calibration2 = TemporalAffine2d(out_channels) if temporal_calibration else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.out_lif = PLIF(mixed=mixed_lif, mixed_spread=mixed_lif_spread)

    def forward(
        self, x: torch.Tensor, state: dict[str, torch.Tensor | None] | None,
        time_index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        state = state or {}
        hidden_current = self.bn1(self.conv1(x))
        if isinstance(self.calibration1, TemporalAffine2d):
            hidden_current = self.calibration1(hidden_current, time_index)
        hidden, hidden_membrane = self.lif1(hidden_current, state.get("hidden"))
        residual_current = self.bn2(self.conv2(hidden))
        if isinstance(self.calibration2, TemporalAffine2d):
            residual_current = self.calibration2(residual_current, time_index)
        residual_current = residual_current + self.shortcut(x)
        out, out_membrane = self.out_lif(residual_current, state.get("out"))
        return out, {"hidden": hidden_membrane, "out": out_membrane}


class PureSpikeResNet(nn.Module):
    """Event-native pure-spike encoder with a spatial population-rate readout.

    Every tensor exchanged between trainable blocks is binary. The continuous
    embedding is formed only by counting output spikes over time and space.
    """

    def __init__(
        self,
        in_channels: int = 2,
        channels: tuple[int, ...] = (64, 128, 256, 384),
        blocks_per_stage: int = 2,
        embedding_dim: int = 384,
        binary_input: bool = False,
        readout_mode: str = "final",
        temporal_calibration: bool = False,
        cumulative_input: bool = False,
        semantic_readout: bool = False,
        signed_semantic_readout: bool = False,
        semantic_temporal: bool = False,
        temporal_steps: int = 8,
        preserve_final_resolution: bool = False,
        mixed_lif: bool = False,
        mixed_lif_spread: float = 1.0,
        dual_event_stem: bool = False,
        detail_recovery: bool = False,
        detail_hidden: int = 768,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must not be empty")
        self.embedding_dim = int(embedding_dim)
        self.binary_input = bool(binary_input)
        if readout_mode not in {"final", "multiscale", "temporal", "temporal_unit", "temporal_delta", "spatiotemporal", "spatial_temporal"}:
            raise ValueError(f"unsupported readout mode: {readout_mode}")
        self.readout_mode = readout_mode
        self.stage_channels = tuple(channels)
        self.blocks_per_stage = int(blocks_per_stage)
        self.temporal_calibration = bool(temporal_calibration)
        self.cumulative_input = bool(cumulative_input)
        self.semantic_readout_enabled = bool(semantic_readout or signed_semantic_readout)
        self.signed_semantic_readout = bool(signed_semantic_readout)
        self.semantic_temporal = bool(semantic_temporal)
        self.temporal_steps = int(temporal_steps)
        self.preserve_final_resolution = bool(preserve_final_resolution)
        self.mixed_lif = bool(mixed_lif)
        self.mixed_lif_spread = float(mixed_lif_spread)
        self.dual_event_stem = bool(dual_event_stem)
        if self.dual_event_stem and in_channels != 3:
            raise ValueError("dual event stem expects ON, OFF, and reconstruction channels")
        self.detail_recovery_enabled = bool(detail_recovery)
        if self.semantic_readout_enabled:
            semantic_channels = self.embedding_dim * (2 if self.signed_semantic_readout else 1)
            self.semantic_conv = nn.Conv2d(channels[-1], semantic_channels, 1, bias=False)
            self.semantic_bn = nn.BatchNorm2d(semantic_channels)
            self.semantic_lif = PLIF()
        self.stem_calibration = (
            TemporalAffine2d(channels[0]) if temporal_calibration else nn.Identity()
        )
        if readout_mode == "multiscale":
            self.output_dim = sum(self.stage_channels)
        elif readout_mode == "spatiotemporal":
            self.output_dim = sum(self.stage_channels) * self.temporal_steps
        elif readout_mode == "spatial_temporal":
            self.output_dim = self.embedding_dim * 4 * self.temporal_steps
        elif readout_mode in {"temporal", "temporal_unit"}:
            self.output_dim = self.embedding_dim * self.temporal_steps
        elif readout_mode == "temporal_delta":
            self.output_dim = self.embedding_dim * (2 * self.temporal_steps - 1)
        else:
            self.output_dim = self.embedding_dim
        if self.dual_event_stem:
            self.fast_stem = nn.Conv2d(2, channels[0], 3, padding=1, bias=False)
            self.fast_stem_bn = nn.BatchNorm2d(channels[0])
            self.fast_stem_lif = PLIF(init_decay=0.25)
            self.slow_stem = nn.Conv2d(1, channels[0], 5, padding=2, bias=False)
            self.slow_stem_bn = nn.BatchNorm2d(channels[0])
            self.slow_stem_lif = PLIF(init_decay=0.85)
            self.stem_fusion_lif = PLIF(
                mixed=mixed_lif, mixed_spread=mixed_lif_spread
            )
        else:
            self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False)
            self.stem_bn = nn.BatchNorm2d(channels[0])
            self.stem_lif = PLIF(mixed=mixed_lif, mixed_spread=mixed_lif_spread)
        layers = []
        current = channels[0]
        for stage, width in enumerate(channels):
            for block in range(blocks_per_stage):
                downsample = stage > 0 and block == 0
                if preserve_final_resolution and stage == len(channels) - 1:
                    downsample = False
                stride = 2 if downsample else 1
                layers.append(SpikeConvBlock(
                    current, width, stride=stride,
                    temporal_calibration=temporal_calibration,
                    mixed_lif=mixed_lif,
                    mixed_lif_spread=mixed_lif_spread,
                ))
                current = width
        self.blocks = nn.ModuleList(layers)
        if current != self.embedding_dim:
            self.readout = nn.Conv2d(current, self.embedding_dim, 1, bias=False)
            self.readout_bn = nn.BatchNorm2d(self.embedding_dim)
            self.readout_lif = PLIF(mixed=mixed_lif, mixed_spread=mixed_lif_spread)
        else:
            self.readout = None
        self.detail_recovery = (
            SpikeProjectionHead(
                dim=self.embedding_dim,
                hidden=int(detail_hidden),
                out_dim=self.embedding_dim,
            )
            if self.detail_recovery_enabled else None
        )
        if self.detail_recovery is not None:
            self.output_dim = self.embedding_dim

    def forward(self, events: torch.Tensor, return_spatial: bool = False) -> dict[str, torch.Tensor]:
        if events.ndim != 5:
            raise ValueError(f"expected [B,T,C,H,W], got {tuple(events.shape)}")
        if self.cumulative_input:
            cumulative = events.cumsum(dim=1)
            cumulative = cumulative / cumulative.amax(
                dim=(1, 3, 4), keepdim=True
            ).clamp_min(1.0e-6)
            events = torch.cat([events, cumulative], dim=2)
        stem_membrane = None
        fast_stem_membrane = None
        slow_stem_membrane = None
        block_states: list[dict[str, torch.Tensor | None] | None] = [
            None for _ in self.blocks
        ]
        readout_membrane = None
        semantic_membrane = None
        trajectory = []
        semantic_trajectory = []
        spike_rates = []
        spatial_spikes = []
        for time_index in range(events.shape[1]):
            x = events[:, time_index]
            if self.binary_input:
                x = (x > 0).to(events.dtype)
            if self.dual_event_stem:
                fast, fast_stem_membrane = self.fast_stem_lif(
                    self.fast_stem_bn(self.fast_stem(x[:, :2])),
                    fast_stem_membrane,
                )
                slow, slow_stem_membrane = self.slow_stem_lif(
                    self.slow_stem_bn(self.slow_stem(x[:, 2:3])),
                    slow_stem_membrane,
                )
                x, stem_membrane = self.stem_fusion_lif(
                    fast + slow, stem_membrane
                )
            else:
                stem_current = self.stem_bn(self.stem(x))
                if isinstance(self.stem_calibration, TemporalAffine2d):
                    stem_current = self.stem_calibration(stem_current, time_index)
                x, stem_membrane = self.stem_lif(stem_current, stem_membrane)
            rates = [x.mean()]
            stage_features = []
            for index, block in enumerate(self.blocks):
                x, block_states[index] = block(x, block_states[index], time_index)
                rates.append(x.mean())
                if (index + 1) % self.blocks_per_stage == 0:
                    stage_features.append(x.mean(dim=(-1, -2)))
            if self.readout is not None:
                x, readout_membrane = self.readout_lif(
                    self.readout_bn(self.readout(x)), readout_membrane
                )
                rates.append(x.mean())
            if self.semantic_readout_enabled:
                semantic, semantic_membrane = self.semantic_lif(
                    self.semantic_bn(self.semantic_conv(x)), semantic_membrane
                )
                semantic_rate = semantic.mean(dim=(-1, -2))
                if self.signed_semantic_readout:
                    positive, negative = semantic_rate.chunk(2, dim=-1)
                    semantic_rate = positive - negative
                semantic_trajectory.append(semantic_rate)
                rates.append(semantic.mean())
            # Spatial population coding gives a substantially finer rate code
            # than one binary CLS token per time step.
            if self.readout_mode in {"multiscale", "spatiotemporal"}:
                trajectory.append(torch.cat(stage_features, dim=-1))
            elif self.readout_mode == "spatial_temporal":
                trajectory.append(F.adaptive_avg_pool2d(x, 2).flatten(1))
            else:
                trajectory.append(x.mean(dim=(-1, -2)))
            if return_spatial:
                spatial_spikes.append(x)
            spike_rates.append(torch.stack(rates).mean())
        steps = torch.stack(trajectory, dim=1)
        semantic_steps = torch.stack(semantic_trajectory, dim=1) if semantic_trajectory else steps
        prefix = steps.cumsum(dim=1) / torch.arange(
            1, steps.shape[1] + 1, device=steps.device, dtype=steps.dtype
        ).view(1, -1, 1)
        if self.readout_mode == "temporal_delta":
            delta = steps[:, 1:] - steps[:, :-1]
            embedding = torch.cat([steps.flatten(1), delta.flatten(1)], dim=-1)
        elif self.readout_mode == "temporal_unit":
            embedding = F.normalize(steps, dim=-1).flatten(1)
        elif self.readout_mode in {"temporal", "spatiotemporal", "spatial_temporal"}:
            embedding = steps.flatten(1)
        else:
            embedding = prefix[:, -1]
        base_embedding = F.normalize(embedding, dim=-1)
        if self.detail_recovery is not None:
            embedding = self.detail_recovery(steps)
        output = {
            "raw_trajectory": steps,
            "trajectory": F.normalize(steps, dim=-1),
            "prefix": F.normalize(prefix, dim=-1),
            "base_embedding": base_embedding,
            "embedding": F.normalize(embedding, dim=-1),
            "semantic_embedding": F.normalize(
                semantic_steps.flatten(1) if self.semantic_temporal
                else semantic_steps.mean(dim=1), dim=-1
            ),
            "semantic_trajectory": F.normalize(semantic_steps, dim=-1),
            "spike_rate": torch.stack(spike_rates).mean(),
        }
        if return_spatial:
            output["spatial_spikes"] = torch.stack(spatial_spikes, dim=1)
        return output


class SpikeProjectionHead(nn.Module):
    def __init__(self, dim: int = 384, hidden: int = 1024, out_dim: int = 128, threshold: float = 1.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lif = PLIF(threshold)
        self.fc2 = nn.Linear(hidden, out_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        membrane = None
        outputs = []
        for t in range(trajectory.shape[1]):
            current = self.bn1(self.fc1(trajectory[:, t]))
            hidden, membrane = self.lif(current, membrane)
            outputs.append(self.bn2(self.fc2(hidden)))
        return F.normalize(torch.stack(outputs, dim=1).mean(dim=1), dim=-1)


class SpikeTextureCompensationHead(nn.Module):
    """Training-only decoder that reads binary patch spikes.

    The hidden state is itself spiking. Its continuous scalar output is used
    only to predict privileged Sobel texture targets during training and is
    discarded together with this head at inference time.
    """

    def __init__(self, dim: int = 384, hidden: int = 64, threshold: float = 1.0) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError("texture hidden dimension must be positive")
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.lif = PLIF(threshold)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("expected binary patch tokens as [B,T,N,C]")
        membrane = None
        predictions = []
        for t in range(tokens.shape[1]):
            hidden, membrane = self.lif(self.fc1(tokens[:, t]), membrane)
            predictions.append(torch.sigmoid(self.fc2(hidden).squeeze(-1)))
        return torch.stack(predictions, dim=1)


class SpikeTextureTrainingWrapper(nn.Module):
    """Keep texture decoding local to each data-parallel student replica."""

    def __init__(
        self,
        encoder: PureSpikeFormer,
        texture_head: SpikeTextureCompensationHead | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.texture_head = texture_head

    def forward(
        self,
        events: torch.Tensor,
        return_tokens: bool = False,
        return_texture: bool = False,
    ) -> dict[str, torch.Tensor]:
        result = self.encoder(events, return_tokens=return_tokens or return_texture)
        if return_texture:
            if self.texture_head is None:
                raise RuntimeError("texture prediction requested without a texture head")
            result["texture_prediction"] = self.texture_head(result["tokens"])
            if not return_tokens:
                del result["tokens"]
        return result


class SemanticAdapter(nn.Module):
    """Training-only low-rank bridge from spike prefixes to teacher semantics."""

    def __init__(self, dim: int = 384, bottleneck: int = 128, hidden: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck, bias=False),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim, bias=False),
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        shape = prefix.shape
        projected = self.net(prefix.reshape(-1, shape[-1])).reshape(shape)
        return F.normalize(projected, dim=-1)


class AppearanceCompletionHead(nn.Module):
    """Training-only bridge from pure-spike prefixes to RGB DINO semantics."""

    def __init__(self, dim: int = 384, hidden: int = 512, target_dim: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, target_dim, bias=False),
        )
        self.confidence = nn.Sequential(
            nn.Linear(dim, hidden // 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = prefix.shape
        flat = prefix.reshape(-1, shape[-1])
        prediction = self.net(flat).reshape(*shape[:-1], -1)
        confidence = self.confidence(flat).reshape(*shape[:-1], 1)
        return prediction, confidence
