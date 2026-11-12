from pathlib import Path
import sys

import torch

CORE = Path(__file__).resolve().parents[1] / "code" / "core"
EVAL = Path(__file__).resolve().parents[1] / "code" / "eval"
TRAIN = Path(__file__).resolve().parents[1] / "code" / "train"
for _p in (CORE, EVAL, TRAIN):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model import PureSpikeFormer  # noqa: E402
from multidepth_spectral_readout import (  # noqa: E402
    DetachedMultiDepthHead,
    NextGenMultiDepthSpikingStudent,
    block_whiten_pca384,
    cosine_linear_probe,
    evaluate_all_modes,
    whiten_block,
)
from train_nextgen_dense_spikeformer import (  # noqa: E402
    centered_saliency_patch_affinity,
    forward_pure_rich,
    gpu_vp_shift_and_energy_mask,
    recalibrate_bn_on_train,
)
from train_warmstart_vp_multidepth import gpu_vp_augment  # noqa: E402


def test_small_student_emits_binary_intermediate_tokens_and_finite_embedding():
    model = PureSpikeFormer(
        in_channels=2,
        dim=96,
        depth=1,
        heads=3,
        patch_size=4,
        threshold=0.5,
        norm="bntt",
        image_size=32,
        temporal_readout="learned",
        use_positional_bias=True,
        attention_mode="normalized",
        signed_readout=True,
        multidepth_readout=False,
        temporal_steps=2,
    ).eval()
    events = torch.rand(2, 2, 2, 32, 32)

    with torch.inference_mode():
        output = model(events, return_tokens=True)

    tokens = output["tokens"]
    assert tokens.shape == (2, 2, 64, 96)
    assert torch.isfinite(output["embedding"]).all()
    assert torch.all((tokens == 0) | (tokens == 1))

    nextgen = NextGenMultiDepthSpikingStudent(model, sal_blend=0.35, sal_floor=0.10).eval()
    with torch.inference_mode():
        ng_out = nextgen(events)
    assert ng_out["embedding"].shape == (2, 96)
    assert torch.allclose(ng_out["embedding"].norm(dim=-1), torch.ones(2), atol=1e-5)


def test_vp_augment_preserves_mean_and_variance():
    torch.manual_seed(42)
    x = torch.randn(4, 8, 2, 16, 16)
    x_aug = gpu_vp_augment(x, max_shift=3, p_flip=0.5, p_shift=1.0)
    assert x_aug.shape == x.shape
    assert torch.allclose(x.mean(dim=(-2, -1)), x_aug.mean(dim=(-2, -1)), atol=1e-6)
    assert torch.allclose(x.var(dim=(-2, -1)), x_aug.var(dim=(-2, -1)), atol=1e-6)


def test_multidepth_spectral_readout_and_detached_head():
    torch.manual_seed(42)
    tr_blocks = [torch.rand(64, 32) for _ in range(5)]
    va_blocks = [torch.rand(16, 32) for _ in range(5)]

    w_tr, w_va = whiten_block(tr_blocks[0], va_blocks[0], p=0.30)
    assert w_tr.shape == (64, 32)
    assert w_va.shape == (16, 32)
    assert torch.isfinite(w_tr).all() and torch.isfinite(w_va).all()

    weighted_blocks = [(1.0, tr_blocks[i], va_blocks[i]) for i in range(5)]
    z_tr, z_va = block_whiten_pca384(weighted_blocks, pb=0.30, pg=0.35, out_dim=48)
    assert z_tr.shape == (64, 48)
    assert z_va.shape == (16, 48)
    assert torch.allclose(z_tr.norm(dim=-1), torch.ones(64), atol=1e-5)
    assert torch.allclose(z_va.norm(dim=-1), torch.ones(16), atol=1e-5)

    head = DetachedMultiDepthHead(dim=32, n_sources=7)
    out = {
        "embedding": torch.randn(8, 32),
        "core_embedding": torch.rand(8, 32),
        "layer_embeddings": torch.rand(8, 4, 32),
        "membrane_embeddings": torch.randn(8, 4, 32),
    }
    bridged, cat_feat = head(out)
    assert bridged.shape == (8, 32)
    assert cat_feat.shape == (8, 32 * 7)
    assert torch.allclose(bridged.norm(dim=-1), torch.ones(8), atol=1e-5)

    y_tr = torch.randint(0, 4, (64,))
    y_va = torch.randint(0, 4, (16,))
    clp_acc, clp_meta = cosine_linear_probe(z_tr, y_tr, z_va, y_va, epochs=5, device="cpu")
    assert 0.0 <= clp_acc <= 1.0
    assert clp_meta["epochs"] == 5


def test_nextgen_hires256_masked_distillation_and_dynamic_readout():
    torch.manual_seed(7)
    model = PureSpikeFormer(
        in_channels=2,
        dim=48,
        depth=4,
        heads=3,
        patch_size=3,
        threshold=0.5,
        norm="bntt",
        image_size=24,  # 24 / 3 = 8x8 = 64 tokens
        temporal_readout="learned",
        local_stem=True,
        use_positional_bias=True,
        attention_mode="normalized",
        signed_readout=True,
        multidepth_readout=True,
        multidepth_readout_layers=3,
        local_structure_mixer=True,
        local_mixer_dilations=(1, 2),
        temporal_steps=2,
    )
    events = torch.rand(3, 2, 2, 24, 24)
    rich = forward_pure_rich(model, events, need_membranes=True)
    assert rich["sem_patches"].shape == (3, 64, 48)
    assert rich["tp_mean"].shape == (3, 4, 64, 48)
    assert torch.allclose(rich["emb_blend"].norm(dim=-1), torch.ones(3), atol=1e-5)

    # Coordinate-free patch affinity must be invariant to orthogonal rotations of teacher basis
    q_orth, _ = torch.linalg.qr(torch.randn(48, 48))
    t_patches = torch.randn(3, 64, 48)
    loss_a = centered_saliency_patch_affinity(rich["sem_patches"], t_patches, rich["sal_w"])
    loss_b = centered_saliency_patch_affinity(rich["sem_patches"], t_patches @ q_orth, rich["sal_w"])
    assert torch.allclose(loss_a, loss_b, atol=1e-5)

    # Energy-compensated patch masking keeps L1 event energy bounded in [1.0x, 1.35x] of masked sum
    masked = gpu_vp_shift_and_energy_mask(
        events, max_shift=2, p_shift=1.0, p_mask=1.0, drop_ratio=0.18, patch_size=3
    )
    assert masked.shape == events.shape
    orig_sum = events.sum(dim=(1, 2, 3, 4))
    masked_sum = masked.sum(dim=(1, 2, 3, 4))
    assert torch.all(masked_sum <= orig_sum * 1.001)
    assert torch.all(masked_sum >= orig_sum * 0.70)

    # BNTT population calibration and dynamic `patch_size=3` U1 readout modes
    x_tr_u8 = torch.randint(0, 10, (12, 2, 2, 24, 24), dtype=torch.uint8)
    x_va_u8 = torch.randint(0, 10, (6, 2, 2, 24, 24), dtype=torch.uint8)
    recalibrate_bn_on_train(model, x_tr_u8, torch.device("cpu"), bs=6)
    ev = evaluate_all_modes(
        model=model,
        head=None,
        x_tr_u8=x_tr_u8,
        x_va_u8=x_va_u8,
        y_tr=torch.randint(0, 3, (12,)),
        y_va=torch.randint(0, 3, (6,)),
        device=torch.device("cpu"),
        run_full_probes=False,
    )
    assert "purespike_quad_dualsal_pca384" in ev
    assert 0.0 <= float(ev["purespike_quad_dualsal_pca384"]["knn10_val"]) <= 1.0
