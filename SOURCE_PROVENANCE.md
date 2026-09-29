# Source provenance

This release candidate was assembled on 2026-09-29 from the RGB2DVS project root on the user's server. The workspace did not contain a Git repository or remote, so the listed server files were copied by path and recorded with SHA-256 hashes in `source_manifest.json`.

The candidate intentionally uses the server's project `code/` tree rather than the unrelated local `src/semantic_snn`, MedMNIST/SDA prototype, older `event_snn_isgd` archive, third-party repositories, or experiment outputs.

The standard training entry point and its later `_eval_safe` revision are both retained because prior metrics must not silently be attributed to code that was not used to produce them. No datasets, weights, caches, results, user credentials, server paths, or launch scripts tied to one machine are included.
