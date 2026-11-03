from __future__ import annotations

import hashlib
import json


CONTRACT = {
    "method": "Observability-Routed Global-Local Spike Relay",
    "rgb_teacher": {
        "type": "DINOv2-ViT-S/14",
        "frozen": True,
        "embedding_dim": 384,
    },
    "event_teacher": {
        "type": "RCLS-Poisson-DINO semantic relay",
        "input": "dvs_only",
        "inverse": "directional_gradient_least_squares_fft_poisson",
        "encoder": "DINOv2-ViT-S/14",
        "embedding_dim": 384,
    },
    "semantic_transport": {
        "type": "residual MLP",
        "dimensions": [384, 1024, 1024, 384],
        "training_only": True,
        "losses": [
            "paired cosine", "RGB prototype", "relation KL",
            "cross-modal contrast", "covariance alignment",
        ],
    },
    "student": {
        "type": "LegacyMaxFormer-10-384",
        "time_steps": 16,
        "stages": [
            {"channels": 96, "blocks": 1, "type": "spike depthwise"},
            {"channels": 192, "blocks": 2, "type": "spike depthwise"},
            {"channels": 384, "blocks": 7, "type": "SSA + spike MLP"},
        ],
        "global_readout": {
            "type": "positive-negative LIF population",
            "embedding_dim": 384,
        },
        "local_readout": {
            "grid": [3, 3],
            "type": "spatial positive-negative LIF population",
            "embedding_dim": 384,
        },
        "binary_intermediate_communication": True,
        "continuous_state": ["membrane potential", "synaptic current"],
        "final_representation": "normalized signed spike-rate embedding",
    },
    "protocol": {
        "selection": "validation only",
        "test_used_for_model_selection": False,
    },
}


def architecture_fingerprint() -> str:
    content = json.dumps(
        CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()
