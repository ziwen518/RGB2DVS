# RGB-to-DVS Pure-Spike Representation Distillation

Research code for transferring RGB-DINO semantic information into an event-only spiking student. The student consumes DVS frames at inference; teacher caches and datasets are supplied separately and are not part of this repository.

The current baseline uses a four-block, 384-dimensional PureSpikeFormer with BNTT, PLIF neurons, signed spike-rate readout, and DINOv2 ViT-S/14 initialization. Intermediate student communication is binary. The final normalized spike-rate embedding is continuous. Training is label-free for the student objective; validation labels may be used for model selection and representation reporting, and must not be mixed into the training target cache.

This is a research prototype. It does not claim ICLR/CVPR acceptance, SOTA performance, measured energy savings, or that the latest experimental architecture has passed its gates. Historical results are tied to their recorded code, split, and evaluation protocol.

## Repository contents

- `code/core/`: student architecture, event dataset readers, and transforms.
- `code/train/`: the historical standard trainer and a follow-up evaluation-safe trainer.
- `code/tools/`: event/RGB teacher-cache preparation, event generation, and representation/downstream evaluation utilities.
- `tests/`: a synthetic CPU contract smoke test; it downloads no data and updates no weights.

The `_eval_safe.py` trainer avoids indexing a train-only teacher cache for validation examples. It is a later source revision and should be treated as a separate code version when comparing with results produced by the original standard trainer.

## Environment

Python 3.10 or newer is recommended. Install the packages in `requirements.txt`; install the PyTorch build appropriate for your hardware from the official PyTorch instructions. DINOv2 weights are loaded through the local PyTorch Hub cache when requested and are not included here.

Run the synthetic architecture check with:

```bash
python -m pytest -q
```

Inspect the training options with:

```bash
python code/train/train_clean_observable_spikeformer_eval_safe.py --help
```

## Training

Prepare your own event frames, locked train/validation split manifest, and train-only observable RGB target cache. These files are intentionally excluded. A typical N-Caltech invocation is:

```bash
python code/train/train_clean_observable_spikeformer_eval_safe.py \
  --project-root . \
  --dataset-name ncaltech101 \
  --data-root /path/to/ncaltech101_event_data \
  --split-manifest /path/to/student_split.json \
  --target-cache /path/to/observable_train_targets.pt \
  --output-dir runs/ncaltech101/example \
  --dino-init --signed-readout --multidepth-readout \
  --gpu 0
```

The paths above are placeholders. Do not put datasets, caches, or checkpoints in Git. Each experiment should record the exact code revision, source hashes, split manifest, target-cache provenance, seed, and evaluation protocol.

Cache-construction utilities are in `code/tools/`. Check each tool's `--help` and its dataset license before use. The code does not download or bundle any dataset automatically.

## Evaluation

`code/tools/evaluate_event_classification.py`, `evaluate_fewshot_lp_ft_cifar10dvs.py`, and `evaluate_downstream_protocols.py` provide frozen-feature or downstream evaluation entry points. Follow the split and label-budget rules for the selected dataset. Do not compare scores across different splits, teachers, or label budgets as if they were the same benchmark.

## Data and artifacts

This repository excludes datasets, event streams, RGB images, frame caches, teacher targets, DINO weights, checkpoints, result directories, and logs. The `.gitignore` also blocks common dataset and model-artifact extensions as a safety net.
