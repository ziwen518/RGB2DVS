from __future__ import annotations

import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def read_events(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".bin":
        try:
            from tonic.datasets.utils import read_mnist_file
            dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
            return read_mnist_file(str(path), dtype=dtype)
        except ModuleNotFoundError:
            # Fallback parser for N-Caltech101's packed four-word event file.
            raw = np.fromfile(path, dtype=np.uint32)
            if raw.size % 4:
                raise ValueError(f"invalid N-Caltech101 event file: {path}")
            raw = raw.reshape(-1, 4)
            result = np.empty(len(raw), dtype=[("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.uint8)])
            result["x"], result["y"], result["t"], result["p"] = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3] & 1
            return result
    # Tonic 1.6's public DVS-128 helper still unpacks the older two-value
    # header API. Use its lower-level parser so old CIFAR10-DVS AEDAT files
    # work across Tonic releases.
    from tonic.io import get_aer_events_from_file, make_structured_array, read_aedat_header_from_file

    header = read_aedat_header_from_file(str(path))
    data_version, data_start = header[0], header[1]
    raw = get_aer_events_from_file(str(path), data_version, data_start)
    address = raw["address"]
    x = (address >> 8) & 0x007F
    y = (address >> 1) & 0x007F
    polarity = address & 0x1
    return make_structured_array(x, y, raw["timeStamp"], polarity)


def _rcls_offsets(loops: int = 6, amplitude: float = 10.0, steps_per_path: int = 5) -> np.ndarray:
    vertices = np.asarray(
        [(-amplitude, 0.0), (0.0, -amplitude), (amplitude, 0.0),
         (0.0, amplitude), (-amplitude, 0.0)],
        dtype=np.float64,
    )
    offsets = [vertices[0]]
    for _ in range(loops):
        for path_index in range(4):
            start, end = vertices[path_index : path_index + 2]
            for step_index in range(1, steps_per_path + 1):
                alpha = step_index / float(steps_per_path)
                offsets.append(start * (1.0 - alpha) + end * alpha)
    return np.asarray(offsets)


def _motion_compensated_coordinates(
    x: np.ndarray, y: np.ndarray, timestamps: np.ndarray, frame_us: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Undo the known RCLS sensor displacement without using the source RGB."""
    offsets = _rcls_offsets()
    phase = np.clip(timestamps.astype(np.float64) / frame_us, 0.0, len(offsets) - 1.0)
    lower = np.minimum(np.floor(phase).astype(np.int64), len(offsets) - 2)
    alpha = phase - lower
    displacement = offsets[lower] * (1.0 - alpha[:, None]) + offsets[lower + 1] * alpha[:, None]
    return np.rint(x - displacement[:, 0]).astype(np.int64), np.rint(y - displacement[:, 1]).astype(np.int64)


def _rcls_direction_indices(timestamps: np.ndarray, frame_us: int = 10_000) -> np.ndarray:
    offsets = _rcls_offsets()
    transition = np.clip(
        (timestamps.astype(np.int64) // frame_us), 0, len(offsets) - 2
    )
    velocity = offsets[transition + 1] - offsets[transition]
    # The four RCLS paths have unique signed x/y velocity pairs.
    direction = np.zeros(len(transition), dtype=np.int64)
    direction[(velocity[:, 0] > 0) & (velocity[:, 1] > 0)] = 0
    direction[(velocity[:, 0] > 0) & (velocity[:, 1] < 0)] = 1
    direction[(velocity[:, 0] < 0) & (velocity[:, 1] < 0)] = 2
    direction[(velocity[:, 0] < 0) & (velocity[:, 1] > 0)] = 3
    return direction


def _rcls_poisson_image(
    events: np.ndarray, sensor_h: int, sensor_w: int,
    frame_us: int = 10_000, threshold: float = 0.15,
) -> torch.Tensor:
    """Recover event-observable log intensity from the known RCLS motion."""
    offsets = _rcls_offsets()
    transitions = len(offsets) - 1
    signed = np.zeros((transitions, sensor_h, sensor_w), dtype=np.float32)
    transition = np.clip(
        events["t"].astype(np.int64) // frame_us, 0, transitions - 1
    )
    polarity = np.where(events["p"] > 0, 1.0, -1.0).astype(np.float32)
    event_x = np.clip(events["x"].astype(np.int64), 0, sensor_w - 1)
    event_y = np.clip(events["y"].astype(np.int64), 0, sensor_h - 1)
    np.add.at(signed, (transition, event_y, event_x), polarity)

    a_xx = np.zeros((sensor_h, sensor_w), dtype=np.float32)
    a_xy = np.zeros_like(a_xx)
    a_yy = np.zeros_like(a_xx)
    b_x = np.zeros_like(a_xx)
    b_y = np.zeros_like(a_xx)
    for index in range(transitions):
        active_y, active_x = np.nonzero(signed[index])
        if not len(active_x):
            continue
        midpoint = 0.5 * (offsets[index] + offsets[index + 1])
        latent_x = np.rint(active_x - midpoint[0]).astype(np.int64)
        latent_y = np.rint(active_y - midpoint[1]).astype(np.int64)
        valid = (
            (latent_x >= 0) & (latent_x < sensor_w)
            & (latent_y >= 0) & (latent_y < sensor_h)
        )
        latent_x, latent_y = latent_x[valid], latent_y[valid]
        value = signed[index, active_y[valid], active_x[valid]] * threshold
        direction = -(offsets[index + 1] - offsets[index])
        dx, dy = float(direction[0]), float(direction[1])
        np.add.at(a_xx, (latent_y, latent_x), dx * dx)
        np.add.at(a_xy, (latent_y, latent_x), dx * dy)
        np.add.at(a_yy, (latent_y, latent_x), dy * dy)
        np.add.at(b_x, (latent_y, latent_x), dx * value)
        np.add.at(b_y, (latent_y, latent_x), dy * value)

    determinant = a_xx * a_yy - a_xy * a_xy
    valid = determinant > 1.0e-6
    gx = np.zeros_like(a_xx)
    gy = np.zeros_like(a_xx)
    gx[valid] = (a_yy[valid] * b_x[valid] - a_xy[valid] * b_y[valid]) / determinant[valid]
    gy[valid] = (a_xx[valid] * b_y[valid] - a_xy[valid] * b_x[valid]) / determinant[valid]
    gx, gy = torch.from_numpy(gx), torch.from_numpy(gy)

    frequency_x = torch.fft.fftfreq(sensor_w) * (2.0 * torch.pi)
    frequency_y = torch.fft.fftfreq(sensor_h) * (2.0 * torch.pi)
    difference_x = torch.exp(1j * frequency_x) - 1.0
    difference_y = torch.exp(1j * frequency_y) - 1.0
    numerator = (
        difference_x.conj()[None, :] * torch.fft.fft2(gx)
        + difference_y.conj()[:, None] * torch.fft.fft2(gy)
    )
    denominator = (
        difference_x.abs().square()[None, :]
        + difference_y.abs().square()[:, None]
    )
    denominator[0, 0] = 1.0
    image = torch.fft.ifft2(numerator / denominator).real
    low, high = torch.quantile(image.flatten(), torch.tensor([0.01, 0.99]))
    return ((image - low) / (high - low).clamp_min(1.0e-6)).clamp(0.0, 1.0)


def events_to_frames(events: np.ndarray, steps: int = 4, size: int = 48, mode: str = "count") -> torch.Tensor:
    sensor_h = max(1, int(events["y"].max()) + 1) if len(events) else 180
    sensor_w = max(1, int(events["x"].max()) + 1) if len(events) else 240
    channels = 8 if mode == "rcls_direction" else (
        4 if mode in {
            "count_recency", "count_recency_local", "rcls_recency_fixed",
            "rcls_recency_fixed_global", "count_motion", "count_burst",
            "count_cumulative",
        } else 2
    )
    poisson_image = None
    if mode == "poisson_dynamic":
        poisson_image = _rcls_poisson_image(events, sensor_h, sensor_w)
    frames = np.zeros((steps, channels, sensor_h, sensor_w), dtype=np.float32)
    if len(events):
        if mode in {"poisson_rate", "poisson_current"}:
            image = _rcls_poisson_image(events, sensor_h, sensor_w)
            image = F.interpolate(
                image[None, None], size=(size, size), mode="bilinear", align_corners=False
            )[0, 0]
            if mode == "poisson_current":
                return image[None, None].expand(steps, 1, -1, -1).contiguous()
            levels = (torch.arange(steps, dtype=image.dtype) + 0.5) / steps
            return (image[None, None] >= levels[:, None, None, None]).to(torch.float32)
        t = events["t"].astype(np.float64)
        if mode in {"rcls_recency_fixed", "rcls_recency_fixed_global"}:
            duration_us = 1_200_000.0
            bin_width_us = duration_us / steps
            bins = np.clip((t / bin_width_us).astype(np.int64), 0, steps - 1)
        else:
            denom = max(float(t[-1] - t[0]), 1.0)
            bins = np.minimum(((t - t[0]) / denom * steps).astype(np.int64), steps - 1)
        x = np.clip(events["x"].astype(np.int64), 0, sensor_w - 1)
        y = np.clip(events["y"].astype(np.int64), 0, sensor_h - 1)
        p = np.clip(events["p"].astype(np.int64), 0, 1)
        if mode == "rcls_direction":
            direction = _rcls_direction_indices(t)
            np.add.at(frames, (bins, direction * 2 + p, y, x), 1.0)
        elif mode == "count_burst":
            counts = np.zeros((steps, 2, sensor_h, sensor_w), dtype=np.uint16)
            np.add.at(counts, (bins, p, y, x), 1)
            frames[:, :2] = (counts > 0).astype(np.float32)
            frames[:, 2:] = (counts >= 2).astype(np.float32)
        else:
            np.add.at(frames, (bins, p, y, x), 1.0)
        if mode in {
            "count_recency", "count_recency_local", "rcls_recency_fixed",
            "rcls_recency_fixed_global",
        }:
            # Recency preserves whether a pixel fired early or late inside a
            # temporal bin while retaining separate ON/OFF event counts.
            if mode in {"rcls_recency_fixed", "count_recency_local"}:
                if mode == "rcls_recency_fixed":
                    local_width = bin_width_us
                    local_start = bins * local_width
                else:
                    local_width = denom / steps
                    local_start = t.min() + bins * local_width
                age = (t - local_start) / max(float(local_width), 1.0)
            elif mode == "rcls_recency_fixed_global":
                age = t / duration_us
            else:
                age = (t - t.min()) / max(float(np.ptp(t)), 1.0)
            np.add.at(frames, (bins, p + 2, y, x), age.astype(np.float32))
        elif mode == "count_motion":
            aligned_x, aligned_y = _motion_compensated_coordinates(x, y, t)
            valid = (
                (aligned_x >= 0) & (aligned_x < sensor_w)
                & (aligned_y >= 0) & (aligned_y < sensor_h)
            )
            np.add.at(
                frames,
                (bins[valid], p[valid] + 2, aligned_y[valid], aligned_x[valid]),
                1.0,
            )
    tensor = torch.from_numpy(frames)
    if mode == "poisson_dynamic":
        raw = torch.log1p(tensor[:, :2])
        raw = raw / raw.max().clamp_min(1.0)
        raw = F.adaptive_avg_pool2d(raw, (size, size))
        static = F.interpolate(
            poisson_image[None, None], size=(size, size),
            mode="bilinear", align_corners=False,
        ).expand(steps, 1, -1, -1)
        return torch.cat([raw, static], dim=1).contiguous()
    if mode == "occupancy":
        tensor = tensor.clamp_max(1.0)
        return F.adaptive_max_pool2d(tensor, (size, size))
    if mode == "count_global":
        tensor = tensor / tensor.max().clamp_min(1.0)
        return F.adaptive_avg_pool2d(tensor, (size, size))
    if mode == "count_burst":
        return F.adaptive_max_pool2d(tensor, (size, size))
    if mode == "count_cumulative":
        raw = F.adaptive_avg_pool2d(torch.log1p(tensor[:, :2]), (size, size))
        raw = raw / raw.max().clamp_min(1.0e-6)
        # Channels are OFF, ON, cumulative-OFF, cumulative-ON.  Splitting the
        # signed integral keeps all inputs non-negative while preserving the
        # event-derived approximation of persistent contour evidence.
        signed = torch.cumsum(raw[:, 1] - raw[:, 0], dim=0)
        signed = signed / signed.abs().max().clamp_min(1.0e-6)
        cumulative_off = (-signed).clamp_min(0.0).unsqueeze(1)
        cumulative_on = signed.clamp_min(0.0).unsqueeze(1)
        return torch.cat([raw, cumulative_off, cumulative_on], dim=1)
    if mode == "log_count":
        # Log compression preserves weak texture events that otherwise vanish
        # after max normalization and uint8 cache quantization.
        tensor = torch.log1p(tensor)
        tensor = tensor / tensor.max().clamp_min(1.0)
        return F.adaptive_avg_pool2d(tensor, (size, size))
    if mode in {
        "count_recency", "count_recency_local", "rcls_recency_fixed",
        "rcls_recency_fixed_global",
    }:
        counts = F.adaptive_avg_pool2d(tensor[:, :2], (size, size))
        timestamp_sum = F.adaptive_avg_pool2d(tensor[:, 2:], (size, size))
        recency = timestamp_sum / counts.clamp_min(1.0e-6)
        counts = torch.log1p(counts)
        counts = counts / counts.max().clamp_min(1.0)
        return torch.cat([counts, recency.clamp(0.0, 1.0)], dim=1)
    if mode == "count_motion":
        # Normalize raw and motion-compensated evidence independently so the
        # repeated aligned trajectory cannot overwhelm the native event path.
        raw = tensor[:, :2]
        aligned = tensor[:, 2:]
        raw = raw / raw.max().clamp_min(1.0)
        aligned = aligned / aligned.max().clamp_min(1.0)
        return torch.cat(
            [F.adaptive_avg_pool2d(raw, (size, size)),
             F.adaptive_avg_pool2d(aligned, (size, size))],
            dim=1,
        )
    if mode == "rcls_direction":
        tensor = tensor / tensor.max().clamp_min(1.0)
        return F.adaptive_avg_pool2d(tensor, (size, size))
    if mode != "count":
        raise ValueError(f"unsupported event frame mode: {mode}")
    tensor = tensor / tensor.amax(dim=(-1, -2), keepdim=True).clamp_min(1.0)
    return F.adaptive_avg_pool2d(tensor, (size, size))


def augment_events(x: torch.Tensor, strength: str = "weak") -> torch.Tensor:
    if strength == "none":
        return x.clone()
    if strength == "phase":
        # RCLS is a closed trajectory, so changing its starting phase preserves
        # the object while discouraging phase-specific temporal shortcuts.
        shift = random.randrange(x.shape[0])
        y = x.roll(shift, dims=0).clone()
        if y.shape[1] >= 4 and shift:
            active = y[:, :2] > 0
            recency = y[:, 2:4]
            recency[active] = torch.remainder(
                recency[active] + shift / float(x.shape[0]), 1.0
            )
        if random.random() < 0.5:
            y[:, :2] = y[:, :2] * random.uniform(0.9, 1.1)
        if random.random() < 0.5:
            keep = torch.rand_like(y[:, :2]) > random.uniform(0.0, 0.025)
            y[:, :2] = y[:, :2] * keep
            if y.shape[1] >= 4:
                y[:, 2:4] = y[:, 2:4] * keep
        return y.clamp(0.0, 1.0).contiguous()
    if strength in {"sensor", "sensor_calibrated", "dogs_sparse_calibrated", "log_dual_threshold"}:
        # Preserve the paired scene while varying event-camera response and
        # temporal sampling statistics seen by the synthetic Student.
        y = x.clone()
        if strength in {"sensor", "dogs_sparse_calibrated"} and y.shape[0] > 1:
            y = y.roll(random.randrange(y.shape[0]), dims=0)
        if strength == "dogs_sparse_calibrated":
            # Dogs synthetic streams are much denser than CIFAR. Calibrate a
            # sparse sensor response without changing the paired target.
            y[:, :2] = torch.log1p(6.0 * y[:, :2]) / math.log(7.0)
            keep = torch.rand_like(y[:, :2]) < random.uniform(0.45, 0.70)
            y[:, :2] = y[:, :2] * keep
            y[:, :2] = (y[:, :2] > random.uniform(0.08, 0.16)).to(y.dtype) * y[:, :2]
            if y.shape[0] > 2:
                y[1:] = y[1:] * (torch.rand(y.shape[0] - 1, 1, 1, 1) > 0.08).to(y.device)
            return y.clamp(0.0, 1.0).contiguous()
        if strength == "log_dual_threshold":
            # ESIM/DVS-Voltmeter-style proxy: log intensity response, separate
            # polarity thresholds, refractory thinning, and sensor noise.
            y[:, :2] = torch.log1p(10.0 * y[:, :2]) / math.log(11.0)
            thresholds = torch.tensor([0.12, 0.16], device=y.device).view(1, 2, 1, 1)
            y[:, :2] = (y[:, :2] >= thresholds).to(y.dtype) * y[:, :2]
            refractory = torch.ones(y.shape[0], 1, 1, 1, device=y.device)
            if y.shape[0] > 1:
                refractory[1:] = (torch.rand(y.shape[0] - 1, 1, 1, 1, device=y.device) > 0.12).to(y.dtype)
            y[:, :2] = y[:, :2] * refractory
            noise = (torch.rand_like(y[:, :2]) < 0.01).to(y.dtype)
            y[:, :2] = torch.maximum(y[:, :2], noise * 0.12)
            return y.clamp(0.0, 1.0).contiguous()
        if y.shape[0] > 1 and random.random() < (0.7 if strength == "sensor" else 0.9):
            # Real DVS streams often contain an initial burst rather than a
            # uniform temporal mass distribution.
            first_gain = random.uniform(1.05, 1.30) if strength == "sensor_calibrated" else 1.0
            y[0] = y[0] * first_gain
            keep_steps = torch.rand(y.shape[0], 1, 1, 1) > random.uniform(
                0.0, 0.2 if strength == "sensor" else 0.1,
            )
            y = y * keep_steps.to(y.device)
        event_channels = min(2, y.shape[1])
        if event_channels:
            if strength == "sensor_calibrated" and event_channels >= 2:
                polarity_gain = torch.tensor([0.88, 1.18]).view(1, 2, 1, 1)
            else:
                polarity_gain = torch.empty(1, event_channels, 1, 1).uniform_(0.7, 1.3)
            y[:, :event_channels] = y[:, :event_channels] * polarity_gain.to(y.device)
            if random.random() < 0.8:
                keep_probability = random.uniform(
                    0.85, 1.0 if strength == "sensor_calibrated" else 0.95,
                )
                keep = torch.rand_like(y[:, :event_channels]) < keep_probability
                y[:, :event_channels] = y[:, :event_channels] * keep
            if strength == "sensor" and random.random() < 0.5:
                gamma = random.uniform(0.7, 1.5)
                y[:, :event_channels] = y[:, :event_channels].clamp_min(0).pow(gamma)
        if y.shape[1] > event_channels:
            active = y[:, :event_channels].sum(dim=1, keepdim=True) > 0
            y[:, event_channels:] = y[:, event_channels:] * active
        if random.random() < 0.5:
            shift = max(1, y.shape[-1] // 24)
            y = y.roll(random.randint(-shift, shift), -1).roll(
                random.randint(-shift, shift), -2,
            )
        return y.clamp(0.0, 1.0).contiguous()
    if strength == "hflip":
        return x.flip(-1).contiguous()
    y = x.clone()
    spatial_size = y.shape[-1]
    crop_scale = random.uniform(0.75, 1.0) if strength == "weak" else random.uniform(0.1, 1.0) ** 0.5
    crop_size = max(1, min(spatial_size, round(spatial_size * crop_scale)))
    top = random.randrange(0, spatial_size - crop_size + 1)
    left = random.randrange(0, spatial_size - crop_size + 1)
    y = F.interpolate(y[..., top : top + crop_size, left : left + crop_size], size=(spatial_size, spatial_size), mode="nearest")
    if random.random() < 0.5:
        y = y.flip(-1)
    if random.random() < 0.5:
        max_shift = max(1, spatial_size // 12) if strength == "weak" else max(1, spatial_size // 4)
        y = y.roll(random.randint(-max_shift, max_shift), dims=-1).roll(random.randint(-max_shift, max_shift), dims=-2)
    if random.random() < (0.35 if strength == "weak" else 0.8):
        channel_mean = y.mean(dim=1, keepdim=True)
        contrast = random.uniform(0.85, 1.15) if strength == "weak" else random.uniform(0.5, 1.5)
        brightness = random.uniform(0.9, 1.1) if strength == "weak" else random.uniform(0.5, 1.5)
        y = ((y - channel_mean) * contrast + channel_mean) * brightness
    if strength != "weak" and random.random() < 0.2:
        y = y.mean(dim=1, keepdim=True).expand_as(y)
    if strength != "weak" and random.random() < 0.35:
        keep = torch.rand_like(y) > random.uniform(0.02, 0.15)
        y = y * keep
    if strength != "weak" and y.shape[0] > 2 and random.random() < 0.5:
        length = random.randrange(1, y.shape[0] + 1)
        start = random.randrange(0, y.shape[0] - length + 1)
        crop = y[start : start + length]
        idx = torch.linspace(0, crop.shape[0] - 1, y.shape[0]).round().long()
        y = crop[idx]
    return y.clamp(0.0, 1.0).contiguous()


class CIFAR10DVSEvents(Dataset):
    def __init__(
        self,
        root: str | Path,
        train: bool,
        steps: int = 4,
        size: int = 48,
        two_views: bool = False,
        seed: int = 42,
        max_samples: int | None = None,
        use_all: bool = False,
        frame_mode: str = "count",
    ) -> None:
        self.root = Path(root)
        self.steps = int(steps)
        self.size = int(size)
        self.two_views = bool(two_views)
        self.frame_mode = str(frame_mode)
        cache_name = (
            f"frames_v2_t{self.steps}_s{self.size}"
            if self.frame_mode == "count"
            else f"frames_v2_{self.frame_mode}_t{self.steps}_s{self.size}"
        )
        self.cache_root = self.root.parent / cache_name
        all_samples: list[tuple[Path, int, int]] = []
        pattern = re.compile(r"_(\d+)\.aedat$")
        for label, name in enumerate(CLASSES):
            files = sorted((self.root / name).glob("*.aedat"))
            for path in files:
                match = pattern.search(path.name)
                ordinal = int(match.group(1)) if match else len(all_samples)
                all_samples.append((path, label, ordinal))
        if not all_samples:
            raise FileNotFoundError(f"no .aedat files under {self.root}")
        rng = np.random.default_rng(seed)
        selected: list[tuple[Path, int, int]] = []
        for label in range(len(CLASSES)):
            group = [sample for sample in all_samples if sample[1] == label]
            order = rng.permutation(len(group))
            cut = int(0.9 * len(group))
            ids = order[:cut] if train else order[cut:]
            selected.extend(group[int(i)] for i in ids)
        if max_samples is not None and max_samples < len(selected):
            if max_samples < len(CLASSES):
                raise ValueError(f"max_samples must be at least {len(CLASSES)} for a stratified subset")
            per_class, remainder = divmod(int(max_samples), len(CLASSES))
            limited: list[tuple[Path, int, int]] = []
            for label in range(len(CLASSES)):
                quota = per_class + int(label < remainder)
                group = [sample for sample in selected if sample[1] == label]
                limited.extend(group[:quota])
            selected = limited
        self.samples = sorted(selected, key=lambda item: str(item[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, ordinal = self.samples[index]
        cache_path = self.cache_root / path.parent.name / f"{path.stem}.npy"
        if cache_path.exists():
            frames = torch.from_numpy(np.load(cache_path, allow_pickle=False)).float().div_(255.0)
        else:
            frames = events_to_frames(
                read_events(path), self.steps, self.size, self.frame_mode,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            quantized = frames.mul(255.0).round().to(torch.uint8).numpy()
            np.save(cache_path, quantized, allow_pickle=False)
        if self.two_views:
            sample_id = f"{path.parent.name}/{path.name}"
            return (augment_events(frames), augment_events(frames)), label, ordinal, sample_id
        sample_id = f"{path.parent.name}/{path.name}"
        return frames, label, ordinal, sample_id


class NCALTECH101Events(Dataset):
    """N-Caltech101 event streams with a deterministic stratified split.

    Tonic stores the official files as ``<class>/*.bin``. The loader also
    accepts AEDAT files so converted copies can use the same training code.
    """

    def __init__(
        self,
        root: str | Path,
        train: bool,
        steps: int = 4,
        size: int = 48,
        two_views: bool = False,
        frame_mode: str = "count",
        seed: int = 42,
        max_samples: int | None = None,
        use_all: bool = False,
    ) -> None:
        self.root = Path(root)
        data_root = self.root / "Caltech101" if (self.root / "Caltech101").is_dir() else self.root
        self.steps = int(steps)
        self.size = int(size)
        self.two_views = bool(two_views)
        self.frame_mode = str(frame_mode)
        cache_name = (
            f"frames_ncaltech101_t{self.steps}_s{self.size}"
            if self.frame_mode == "count" else
            f"frames_ncaltech101_{self.frame_mode}_t{self.steps}_s{self.size}"
        )
        self.cache_root = self.root / cache_name
        paths = sorted([*data_root.rglob("*.bin"), *data_root.rglob("*.aedat")])
        if not paths:
            raise FileNotFoundError(
                f"no N-Caltech101 event files under {data_root}; "
                "download with tonic.datasets.NCALTECH101 or provide an extracted root"
            )
        class_names = sorted({path.parent.name for path in paths})
        class_to_id = {name: index for index, name in enumerate(class_names)}
        all_samples = [(path, class_to_id[path.parent.name], index) for index, path in enumerate(paths)]
        rng = np.random.default_rng(seed)
        if use_all:
            selected = list(all_samples)
        else:
            selected = []
            for label in range(len(class_names)):
                group = [sample for sample in all_samples if sample[1] == label]
                order = rng.permutation(len(group))
                cut = max(1, int(round(0.9 * len(group))))
                ids = order[:cut] if train else order[cut:]
                selected.extend(group[int(i)] for i in ids)
        if max_samples is not None and max_samples < len(selected):
            if max_samples < len(class_names):
                raise ValueError(f"max_samples must be at least {len(class_names)} for a stratified subset")
            per_class, remainder = divmod(int(max_samples), len(class_names))
            limited: list[tuple[Path, int, int]] = []
            for label in range(len(class_names)):
                quota = per_class + int(label < remainder)
                limited.extend([sample for sample in selected if sample[1] == label][:quota])
            selected = limited
        self.class_names = tuple(class_names)
        self.samples = sorted(selected, key=lambda item: str(item[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, ordinal = self.samples[index]
        relative = path.relative_to(self.root).with_suffix("")
        if relative.parts and relative.parts[0].lower() == "events":
            relative = Path(*relative.parts[1:])
        cache_path = self.cache_root / relative.parent / f"{relative.name}.npy"
        if cache_path.exists():
            frames = torch.from_numpy(np.load(cache_path, allow_pickle=False)).float().div_(255.0)
        else:
            frames = events_to_frames(
                read_events(path), self.steps, self.size, self.frame_mode,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, frames.mul(255.0).round().to(torch.uint8).numpy(), allow_pickle=False)
        sample_id = str(path.relative_to(self.root)).replace("\\", "/")
        if self.two_views:
            return (augment_events(frames), augment_events(frames)), label, ordinal, sample_id
        return frames, label, ordinal, sample_id


class MotionEvents(Dataset):
    """Paired event files generated from an image-folder dataset.

    Each ``.npz`` stores x/y/t/p and the original RGB relative path. The
    split is represented by the directory layout, so no guessed pairing is
    involved.
    """

    def __init__(
        self,
        root: str | Path,
        train: bool,
        steps: int = 4,
        size: int = 48,
        two_views: bool = False,
        max_samples: int | None = None,
        augment_strength: str = "weak",
        frame_mode: str = "count",
        paired_first: bool = False,
        preload_frames: bool = False,
    ) -> None:
        self.root = Path(root)
        split = "train" if train else "test"
        self.split = split
        data_root = self.root / split
        self.steps = int(steps)
        self.size = int(size)
        self.two_views = bool(two_views)
        self.augment_strength = augment_strength
        self.frame_mode = frame_mode
        self.paired_first = bool(paired_first)
        self.cache_root = self.root / f"frames_motion_{self.frame_mode}_t{self.steps}_s{self.size}" / split
        self.fusion_cache_roots = None
        if self.frame_mode in {
            "poisson_dynamic_cached", "poisson_residual_cached",
            "poisson_sparse05_cached", "poisson_sparse15_cached",
        }:
            poisson_root = (
                self.root / f"frames_motion_poisson_current_t{self.steps}_s{self.size}" / split
            )
            if not poisson_root.exists() or not any(poisson_root.rglob("*.npy")):
                candidates = sorted(
                    self.root.glob(f"frames_motion_poisson_current_t*_s{self.size}/{split}")
                )
                candidates = [
                    candidate for candidate in candidates
                    if any(candidate.rglob("*.npy"))
                ]
                if candidates:
                    poisson_root = max(
                        candidates,
                        key=lambda candidate: sum(1 for _ in candidate.rglob("*.npy")),
                    )
            self.fusion_cache_roots = (
                self.root / f"frames_motion_count_global_t{self.steps}_s{self.size}" / split,
                poisson_root,
            )
        paths = sorted(data_root.rglob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no generated event .npz files under {data_root}")
        class_names = sorted({path.parent.name for path in paths})
        class_to_id = {name: index for index, name in enumerate(class_names)}
        samples = [(path, class_to_id[path.parent.name], index) for index, path in enumerate(paths)]
        if max_samples is not None and max_samples < len(samples):
            per_class, remainder = divmod(int(max_samples), len(class_names))
            limited = []
            for label in range(len(class_names)):
                quota = per_class + int(label < remainder)
                limited.extend([sample for sample in samples if sample[1] == label][:quota])
            samples = limited
        self.class_names = tuple(class_names)
        self.samples = samples
        self.preloaded_frames = None
        if preload_frames:
            if self.fusion_cache_roots is not None:
                pairs = [tuple(
                    cache_root / path.relative_to(self.root / self.split).with_suffix(".npy")
                    for cache_root in self.fusion_cache_roots
                ) for path, _, _ in self.samples]
                if all(all(path.exists() for path in pair) for pair in pairs):
                    fused = []
                    for pair in pairs:
                        count = np.load(pair[0], allow_pickle=False)
                        poisson = np.load(pair[1], allow_pickle=False)
                        if poisson.shape[0] != count.shape[0]:
                            poisson = np.repeat(poisson[:1], count.shape[0], axis=0)
                        if self.frame_mode.startswith("poisson_sparse"):
                            alpha = 0.05 if "sparse05" in self.frame_mode else 0.15
                            base = poisson.astype(np.float32) / 255.0
                            on = count[:, 0:1].astype(np.float32) / 255.0
                            off = count[:, 1:2].astype(np.float32) / 255.0
                            residual = np.concatenate([on, off, on - off], axis=1)
                            value = np.clip(
                                np.repeat(base, 3, axis=1) + alpha * residual, 0.0, 1.0
                            )
                        elif self.frame_mode == "poisson_residual_cached":
                            base = poisson.astype(np.float32)
                            on = count[:, 0:1].astype(np.float32) - 127.5
                            off = count[:, 1:2].astype(np.float32) - 127.5
                            residual = np.concatenate([on, off, on - off], axis=1)
                            value = np.clip(
                                np.repeat(base, 3, axis=1) + 0.10 * residual,
                                0.0, 255.0,
                            ).round().astype(np.uint8)
                        else:
                            value = np.concatenate([count, poisson], axis=1)
                        fused.append(value)
                    self.preloaded_frames = torch.from_numpy(np.stack(fused))
                    if self.frame_mode.startswith("poisson_sparse"):
                        self.preloaded_frames = self.preloaded_frames.mul(255.0)
            else:
                cache_paths = [
                    self.cache_root / path.relative_to(self.root / self.split).with_suffix(".npy")
                    for path, _, _ in self.samples
                ]
                if all(path.exists() for path in cache_paths):
                    self.preloaded_frames = torch.from_numpy(
                        np.stack([np.load(path, allow_pickle=False) for path in cache_paths])
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, ordinal = self.samples[index]
        cache_path = self.cache_root / path.relative_to(self.root / self.split).with_suffix(".npy")
        if self.preloaded_frames is not None:
            frames = self.preloaded_frames[index].float().div_(255.0)
        elif self.fusion_cache_roots is not None:
            relative = path.relative_to(self.root / self.split).with_suffix(".npy")
            cached = [cache_root / relative for cache_root in self.fusion_cache_roots]
            if not all(item.exists() for item in cached):
                raise FileNotFoundError(f"missing cached fusion input for {relative}")
            count = np.load(cached[0], allow_pickle=False)
            poisson = np.load(cached[1], allow_pickle=False)
            if poisson.shape[0] != count.shape[0]:
                poisson = np.repeat(poisson[:1], count.shape[0], axis=0)
            if self.frame_mode.startswith("poisson_sparse"):
                alpha = 0.05 if "sparse05" in self.frame_mode else 0.15
                base = poisson.astype(np.float32) / 255.0
                on = count[:, 0:1].astype(np.float32) / 255.0
                off = count[:, 1:2].astype(np.float32) / 255.0
                value = np.clip(
                    np.repeat(base, 3, axis=1)
                    + alpha * np.concatenate([on, off, on - off], axis=1),
                    0.0, 1.0,
                ) * 255.0
            elif self.frame_mode == "poisson_residual_cached":
                base = poisson.astype(np.float32)
                on = count[:, 0:1].astype(np.float32) - 127.5
                off = count[:, 1:2].astype(np.float32) - 127.5
                value = np.clip(
                    np.repeat(base, 3, axis=1)
                    + 0.10 * np.concatenate([on, off, on - off], axis=1),
                    0.0, 255.0,
                ).round().astype(np.uint8)
            else:
                value = np.concatenate([count, poisson], axis=1)
            frames = torch.from_numpy(value).float().div_(255.0)
        elif cache_path.exists():
            frames = torch.from_numpy(np.load(cache_path, allow_pickle=False)).float().div_(255.0)
        else:
            with np.load(path, allow_pickle=False) as payload:
                event = np.zeros(len(payload["x"]), dtype=[("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.uint8)])
                for field in event.dtype.names:
                    event[field] = payload[field]
            frames = events_to_frames(event, self.steps, self.size, self.frame_mode)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, frames.mul(255.0).round().to(torch.uint8).numpy(), allow_pickle=False)
        sample_id = str(path.relative_to(self.root)).replace("\\", "/")
        if self.two_views:
            first = frames if self.paired_first else augment_events(frames, self.augment_strength)
            return (first, augment_events(frames, self.augment_strength)), label, ordinal, sample_id
        return frames, label, ordinal, sample_id


class DSECEvents(Dataset):
    """Read DSEC ``events.h5`` archives and timestamp-aligned RGB frames.

    Each sample is one RGB frame; its event window ends at that frame timestamp
    and starts at the preceding frame timestamp (with ``steps`` temporal bins).
    The archive remains zipped on disk and is opened lazily per sequence.
    """
    def __init__(self, root: str | Path, train: bool, steps: int = 16,
                 size: int = 48, two_views: bool = False,
                 frame_mode: str = "count_global", max_samples: int | None = None,
                 **kwargs) -> None:
        self.root, self.split = Path(root), ("train" if train else "test")
        self.steps, self.size = int(steps), int(size)
        self.two_views, self.frame_mode = bool(two_views), str(frame_mode)
        self.cache_root = self.root / f"frames_dsec_{self.frame_mode}_t{self.steps}_s{self.size}" / self.split
        self.samples = []
        import zipfile
        split_root = self.root / self.split
        for seq_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            seq = seq_dir.name
            event_zip = seq_dir / f"{seq}_events_left.zip"
            image_zip = seq_dir / f"{seq}_images_rectified_left.zip"
            ts_path = seq_dir / f"{seq}_image_timestamps.txt"
            complete = all((seq_dir / f"{seq}_{suffix}.ok").exists() for suffix in (
                "events_left.zip", "images_rectified_left.zip", "image_timestamps.txt"))
            if not complete or not event_zip.exists() or not image_zip.exists() or not ts_path.exists():
                continue
            try:
                with zipfile.ZipFile(event_zip) as z:
                    if "events.h5" not in z.namelist():
                        continue
                with zipfile.ZipFile(image_zip) as z:
                    image_names = {p for p in z.namelist() if p.lower().endswith(".png")}
                timestamps = [int(line.strip()) for line in ts_path.read_text().splitlines() if line.strip()]
            except Exception:
                continue
            for index, timestamp in enumerate(timestamps):
                name = f"{index:06d}.png"
                if name in image_names:
                    self.samples.append((seq, index, timestamp, event_zip, image_zip))
        if max_samples is not None:
            self.samples = self.samples[:int(max_samples)]
        if not self.samples:
            raise FileNotFoundError(f"no complete DSEC samples under {split_root}")
        self.class_names = ("dsec",)
        self._handles = {}

    def __len__(self):
        return len(self.samples)

    def _events(self, event_zip, start_us, end_us):
        import io, zipfile, h5py
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass
        key = str(event_zip)
        if key not in self._handles:
            with zipfile.ZipFile(event_zip) as z:
                payload = z.read("events.h5")
            cache_path = event_zip.with_suffix(".events.h5")
            if not cache_path.exists() or cache_path.stat().st_size != len(payload):
                cache_path.write_bytes(payload)
            self._handles[key] = h5py.File(str(cache_path), "r")
        f = self._handles[key]
        offset = int(f["t_offset"][()])
        t = f["events/t"]
        # DSEC provides a millisecond index; refine only the small candidate
        # slices instead of materializing hundreds of millions of timestamps.
        index = np.asarray(f["ms_to_idx"][:], dtype=np.int64)
        rel_start, rel_end = max(0, start_us - offset), max(0, end_us - offset)
        lo_ms, hi_ms = rel_start // 1000, rel_end // 1000
        lo0 = int(index[min(lo_ms, len(index) - 1)])
        hi0 = int(index[min(hi_ms + 1, len(index) - 1)])
        cand_t = np.asarray(t[lo0:hi0], dtype=np.int64)
        lo = lo0 + int(np.searchsorted(cand_t, rel_start, side="left"))
        hi = lo0 + int(np.searchsorted(cand_t, rel_end, side="right"))
        return np.rec.fromarrays([
            np.asarray(f["events/x"][lo:hi]), np.asarray(f["events/y"][lo:hi]),
            np.asarray(t[lo:hi], dtype=np.int64) + offset,
            np.asarray(f["events/p"][lo:hi]),
        ], names="x,y,t,p")

    def __getitem__(self, index):
        import io, zipfile
        seq, frame_index, timestamp, event_zip, image_zip = self.samples[index]
        cache_path = self.cache_root / seq / f"{frame_index:06d}.npy"
        if cache_path.exists():
            frames = torch.from_numpy(np.load(cache_path, allow_pickle=False)).float().div_(255.0)
            return frames, 0, frame_index, f"{self.split}/{seq}/{frame_index:06d}"
        previous = self.samples[index - 1][2] if index and self.samples[index - 1][0] == seq else timestamp - 100000
        events = self._events(event_zip, previous, timestamp)
        frames = events_to_frames(events, self.steps, self.size, self.frame_mode)
        sample_id = f"{self.split}/{seq}/{frame_index:06d}"
        if self.two_views:
            return (frames, augment_events(frames, "weak")), 0, frame_index, sample_id
        return frames, 0, frame_index, sample_id


class NImageNetEvents(Dataset):
    """N-ImageNet-mini event files with an external manifest split."""
    def __init__(self, root: str | Path, train: bool, steps: int = 16,
                 size: int = 48, two_views: bool = False, seed: int = 42,
                 max_samples: int | None = None, use_all: bool = True,
                 frame_mode: str = "count_global", preload_frames: bool = False) -> None:
        self.root = Path(root)
        self.steps, self.size = int(steps), int(size)
        self.two_views, self.frame_mode = bool(two_views), str(frame_mode)
        self.cache_root = self.root / (
            f"frames_nimagenet_t{self.steps}_s{self.size}"
            if self.frame_mode == "count_global" else
            f"frames_nimagenet_{self.frame_mode}_t{self.steps}_s{self.size}"
        )
        paths = sorted((self.root / "events").rglob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no N-ImageNet event files under {self.root / 'events'}")
        class_names = sorted({p.parent.name for p in paths})
        class_to_id = {name: i for i, name in enumerate(class_names)}
        samples = [(p, class_to_id[p.parent.name], i) for i, p in enumerate(paths)]
        if max_samples is not None and max_samples < len(samples):
            per_class, remainder = divmod(int(max_samples), len(class_names))
            samples = sum(([s for s in samples if s[1] == label][:per_class + int(label < remainder)]
                           for label in range(len(class_names))), [])
        self.class_names = tuple(class_names)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, ordinal = self.samples[index]
        relative = path.relative_to(self.root).with_suffix(".npy")
        cache_path = self.cache_root / relative.relative_to("events")
        if cache_path.exists():
            frames = torch.from_numpy(np.load(cache_path, allow_pickle=False)).float().div_(255.0)
        else:
            with np.load(path, allow_pickle=False) as payload:
                event = np.zeros(len(payload["x"]), dtype=[("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.uint8)])
                for field in event.dtype.names:
                    event[field] = payload[field]
            frames = events_to_frames(event, self.steps, self.size, self.frame_mode)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, frames.mul(255).round().to(torch.uint8).numpy(), allow_pickle=False)
        sample_id = str(path.relative_to(self.root)).replace("\\", "/")
        return frames, label, ordinal, sample_id


def make_event_dataset(name: str, *args, **kwargs) -> Dataset:
    name = name.lower().replace("-", "").replace("_", "")
    if name in {"cifar10dvs", "cifar10"}:
        kwargs.pop("paired_first", None)
        kwargs.pop("preload_frames", None)
        return CIFAR10DVSEvents(*args, **kwargs)
    if name in {"ncaltech101", "caltech101", "ncaltech"}:
        kwargs.pop("paired_first", None)
        kwargs.pop("preload_frames", None)
        return NCALTECH101Events(*args, **kwargs)
    if name in {"motion", "simulated", "imagenetdogsmotion"}:
        return MotionEvents(*args, **kwargs)
    if name in {"nimagenet", "nimagenetmini", "nimagenet101"}:
        return NImageNetEvents(*args, **kwargs)
    if name in {"dsec", "dsecsubset"}:
        return DSECEvents(*args, **kwargs)
    raise ValueError(f"unsupported event dataset: {name}")


def spatial_band_stop(x: torch.Tensor, band: int, bands: int = 4) -> torch.Tensor:
    """Remove one radial spatial-frequency band; supports [B,C,H,W] and [B,T,C,H,W]."""
    original = x.shape
    flat = x.reshape(-1, *x.shape[-3:]).float()
    h, w = flat.shape[-2:]
    fy = torch.fft.fftfreq(h, device=x.device).view(h, 1)
    fx = torch.fft.fftfreq(w, device=x.device).view(1, w)
    radius = torch.sqrt(fy.square() + fx.square())
    radius = radius / radius.max().clamp_min(1.0e-8)
    low = float(band) / float(bands)
    high = float(band + 1) / float(bands)
    mask = ~((radius >= low) & (radius < high))
    spectrum = torch.fft.fft2(flat, dim=(-2, -1))
    filtered = torch.fft.ifft2(spectrum * mask, dim=(-2, -1)).real
    return filtered.reshape(original).to(dtype=x.dtype)
