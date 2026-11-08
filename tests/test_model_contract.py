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
    whiten_block,
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
