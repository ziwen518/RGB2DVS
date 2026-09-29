from pathlib import Path
import sys

import torch

CORE = Path(__file__).resolve().parents[1] / "code" / "core"
sys.path.insert(0, str(CORE))

from model import PureSpikeFormer  # noqa: E402


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
