# Reproducibility notes

## Environment and entry points

Use Python 3.10 or newer and install `requirements-dev.txt` for the CPU model
check. Dataset tools and plotting utilities may additionally require SciPy,
scikit-learn, matplotlib, OpenCV, requests, or v2e, depending on the selected
script. Consult that script's imports and `--help` before running it.

The recommended training entry point is
`code/train/train_clean_observable_spikeformer_eval_safe.py`. For post-convergence
warm-start refinement with locked BNTT running statistics, variance-preserving
spatial translation/flip invariance, full-bank neighborhood KL distillation, and
multi-depth spectral readout, use
`code/train/train_warmstart_vp_multidepth.py` together with the standalone
evaluator `code/eval/multidepth_spectral_readout.py` (see
`configs/ncaltech101_warmstart_vp_multidepth.json`). The original
trainer is retained so that historical checkpoints can be attributed to the
source that actually produced them. Core model and trainer implementations are
unchanged by repository cleanup.

DINOv2 initialization and teacher-cache tools load `dinov2_vits14` from the
local PyTorch Hub directory `facebookresearch_dinov2_main`. Prepare the upstream
DINOv2 source and weights in that cache before enabling `--dino-init` or building
teacher targets. Record the upstream revision and weight hash for each run.
Dependencies and weights are not bundled in this repository.

## Reference configurations

The JSON files in `configs/` record experiment settings; the trainers accept
command-line arguments rather than a `--config` option. Supply the appropriate
dataset, paths, and corresponding flags explicitly, as shown in the root README.

The `standard_purespikeformer_*_medium.json` references describe a four-block,
384-dimensional, six-head PureSpikeFormer with patch size 4, 16 event frames at
48 by 48 pixels, full-respike blocks, standard instantaneous Q/K/V, and signed
readout over the last three depths. Their objective weights are observable
alignment 10 and geometry 1, with label losses disabled. Temporal targets,
membrane readout, and augmentation are disabled in these reference definitions.

The earlier `cifar10_observable.json` and `dogs_observable.json` files describe
different settings. Do not silently substitute one configuration for another
when reproducing a reported checkpoint. Dataset roots, manifests, and target
caches are relative examples and must match the prepared assets.

## Data preparation and locked splits

Datasets, RGB samples, raw event streams, frame tensors, teacher caches, and
checkpoints are supplied separately under their respective distribution terms.
Use the dataset/cache preparation tools in `code/tools/`; no machine-specific
launcher is required.

`configs/splits/real_cifar10dvs_split_v1.json` preserves the existing real
CIFAR10-DVS sample assignments: 8,000 training, 1,000 validation, and 1,000 test
recordings, with seed 20260908. Only its event/frame root paths were changed to
repository-relative examples during cleanup; sample IDs and protocol metadata
were retained. Its preprocessing is `frame_mode=count`, T=16, size=48, which must
not be silently replaced by a synthetic-data configuration's `count_global`.
Pass the manifest explicitly to the relevant evaluation entry point.

The source-image overlap between synthetic CIFAR training data and real
CIFAR10-DVS recordings remains unresolved in this manifest. Do not claim
instance-disjoint synthetic-to-real transfer without a separate overlap audit.
This manifest is a protocol asset, not evidence that a test evaluation has run.

## Training and evaluation boundaries

The student representation objective is label-free. Report separately any
supervision used to construct offline teacher targets, select a checkpoint,
fit a linear head, or fine-tune downstream. Keep fit, validation, and test IDs
disjoint, and use train-only data to fit caches and learned target transforms.

Use validation to select checkpoints and hyperparameters. Evaluate the locked
test split after selection, and record test access. Distinguish signed/core kNN,
fixed-ridge diagnostics, trained linear probes, and end-to-end fine-tuning.
Few-shot label subsets do not count as independent pretraining seeds.

Record code revision, trainer, configuration, checkpoint and target-cache hashes,
split manifest, seed, label budget, readout, frame preprocessing, and metric with
each result. Include spike rate, representation rank/spread, latency, and memory
when those measurements support a claim. SynOps and technology-based energy
proxies are estimates rather than measured device energy.

DSEC utilities and other diagnostic scripts are included as tools. Their presence
does not establish a completed official benchmark. Identify reduced-subset,
smoke, synthetic, oracle, or unmatched-protocol results explicitly.

## Source and figure provenance

`source_manifest.json` records hashes for the original 2026-09-29 source snapshot.
It is a historical manifest, not a complete inventory of the current repository.
Preserve the trainer/source revision associated with each historical result.

`representation_pca_provenance.json` records the qualitative figure's sample IDs,
training-only PCA fitting rule, extraction method, and comparison limitations.
PCA colors are fitted separately per model. The addressed-consistency checkpoint
is a four-epoch pilot, while the historical students used 40 epochs; this panel
does not establish a compute-matched quantitative advantage.
