# assets.py

Validates local benchmark declarations and builds model/evaluator-separated task records.

* Confines declared files to the ignored `data/benchmark/` root.
* Verifies per-file provenance, hashes, inventories, and task IDs.
* Keeps public question input separate from gold evaluator records.
* Parses the local Base-Lite public task file without requiring gold assets.
* Verifies ignored, symlink-free report destinations; static preflight contacts no services.
