# Chapter 5 — DVC / Dataset Versioning

Machine learning is not just code — it is code plus data plus models plus the exact configuration that produced a result. Traditional software engineering solved the "what changed?" problem for source code decades ago with Git. But the moment you try to store a 40 GB training set or a 2 GB model checkpoint in a Git repository, the whole system buckles. Cloning becomes glacial, history bloats forever, and collaboration collapses. This is the gap **Data Version Control (DVC)** was built to fill: it brings Git-style versioning, branching, and reproducibility to large data and model artifacts without stuffing those bytes into Git itself.

In this chapter we build up from first principles. We start with why Git alone fails, then set up DVC and learn how its pointer files work. We wire DVC to cloud remote storage, then graduate to reproducible pipelines and experiments with `dvc.yaml`. We cover dataset lineage, CI/CD integration, and finish with a hard-nosed comparison of DVC against LakeFS and Delta Lake so you can pick the right tool for the job. Throughout, expect production-grade examples and interview questions with model answers — this chapter doubles as interview prep for any MLOps role.

---

## 1. Git Basics Recap — Why Git Alone Fails for Large Data and Model Files

### Plain-language explanation

Git is a **content-addressable, distributed version control system**. Every commit snapshots your files, computes SHA hashes, and stores compressed *deltas* (differences) in a hidden `.git` directory. This is brilliant for text: a one-line change to a Python file stores only that tiny diff. The design assumption is that files are small, text-based, and change incrementally.

Data and model files violate every one of those assumptions. A trained model or a Parquet dataset is a large binary blob. Binaries do not diff meaningfully — change one row and Git often stores an entirely new copy. Because Git keeps *all history forever* and every clone pulls *the entire history*, a repo that has cycled through ten versions of a 1 GB dataset can balloon to 10+ GB even though the working copy is only 1 GB.

### Why it matters in production

- **Clone and fetch times explode.** New engineers wait 30 minutes to clone. CI runners pay that cost on every fresh checkout.
- **History is permanent bloat.** Even if you `git rm` a large file, it lingers in history. Removing it requires rewriting history (`git filter-repo`), which breaks every existing clone.
- **GitHub/GitLab hard limits.** GitHub blocks pushes of files over 100 MB and warns above 50 MB. Repos over a few GB get throttled.
- **No reproducibility guarantee.** Git tracks code, but the data that produced a model lives "somewhere else" (a shared drive, an S3 bucket). Nothing ties commit `a1b2c3` of the code to the exact dataset snapshot used.

Git LFS (Large File Storage) partially helps by replacing large files with pointers, but it is tightly coupled to the Git host, has its own storage quotas and bandwidth billing, and lacks pipeline and experiment features. DVC decouples storage entirely and adds a reproducibility engine on top.

### Real-world example / use case

A computer-vision team commits a 3 GB image dataset directly into Git "just to keep it together with the code." Six months later the repo is 45 GB after multiple dataset refreshes, cloning takes 25 minutes, and CI storage costs have tripled. They cannot even delete the old data without rewriting history and invalidating every developer's local clone. Migrating to DVC lets them shrink the Git repo back to a few megabytes of code plus tiny pointer files, while the actual image bytes live in S3.

### Code or config example

```bash
# The anti-pattern: committing large binaries directly
git add data/images.tar.gz   # 3 GB
git commit -m "Add training images"
git push
# remote: error: File data/images.tar.gz is 3000.00 MB; this exceeds
# remote: GitHub's file size limit of 100.00 MB
# ! [remote rejected] main -> main (pre-receive hook declined)

# Inspect how much history bloat large files cause
git count-objects -vH
# size-pack: 8.42 GiB   <- history you can never cheaply shed
```

### Interview questions

**Q: Why can't you just use Git for large datasets and models?**
A: Git stores full history in every clone and does not delta-compress binaries well, so large files bloat the `.git` directory permanently and slow down clone/fetch. Hosts like GitHub also enforce per-file size limits (~100 MB). Critically, Git alone gives no link between a code commit and the exact data snapshot used, so reproducibility is not guaranteed.

**Q: How is Git LFS different from DVC, and when would you still choose DVC?**
A: Git LFS replaces large files with pointers stored on the Git host's LFS backend — it is storage-only and coupled to the host's quotas and bandwidth pricing. DVC is storage-agnostic (any S3/GCS/Azure/SSH backend), and adds pipelines (`dvc.yaml`), experiment tracking (`dvc exp`), metrics, and lineage. Choose DVC when you need reproducible ML pipelines and flexible storage, not just large-file offloading.

### Pitfalls & best practices

- **Never commit large binaries to Git**, even once — history bloat is permanent.
- If a large file already leaked into history, use `git filter-repo` to purge it, then adopt DVC or LFS going forward.
- Keep `.gitignore` disciplined so data directories are never staged accidentally.

---

## 2. DVC Setup — `dvc init`, `dvc add`, and How `.dvc` Pointer Files Work

### Plain-language explanation

DVC sits *alongside* Git in the same repository. You run `dvc init` once to create a `.dvc/` directory (config and cache metadata). When you `dvc add` a large file or folder, DVC does three things:

1. Moves the actual bytes into a local content-addressable **cache** (`.dvc/cache`), named by an MD5 hash of the content.
2. Creates a tiny text **pointer file** with a `.dvc` extension (e.g., `data/images.tar.gz.dvc`) that records the hash, size, and path.
3. Adds the real file to `.gitignore` so Git ignores the bytes but tracks the pointer.

You then commit the small `.dvc` pointer to Git. Git now versions *the reference*; DVC manages *the content*. To save space, DVC links files from the cache into your workspace (reflink/hardlink/symlink when the filesystem supports it), so the data is not duplicated.

### Why it matters in production

The pointer file is the linchpin of reproducibility. Because the hash lives in Git, checking out an old Git commit and running `dvc checkout` restores the *exact* data that existed at that point in history. Your Git repo stays tiny and fast, while terabytes of data are addressed by content hash and fetched on demand.

### Real-world example / use case

A fraud-detection team versions a `transactions.parquet` dataset (8 GB). Data scientists work on feature branches; each branch's `.dvc` pointer references a different dataset snapshot. Switching branches and running `dvc checkout` swaps the local data to match — no manual copying, no "which version of the CSV did you use?" confusion.

### Code or config example

```bash
# One-time setup in an existing Git repo
git init                      # if not already a repo
dvc init                      # creates .dvc/ and .dvcignore
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# Track a large dataset
dvc add data/transactions.parquet
# Creates data/transactions.parquet.dvc and updates data/.gitignore

git add data/transactions.parquet.dvc data/.gitignore
git commit -m "Track transactions dataset with DVC"
```

An example `.dvc` pointer file (`data/transactions.parquet.dvc`):

```yaml
outs:
  - md5: 3b1a9f2c4d5e6f708192a3b4c5d6e7f8
    size: 8589934592
    hash: md5
    path: transactions.parquet
```

Restoring data after checking out an old commit:

```bash
git checkout v1.2.0            # code + .dvc pointers move to old version
dvc checkout                   # data files restored to match the pointers
```

### Interview questions

**Q: What exactly is stored in a `.dvc` file, and why commit it to Git?**
A: A `.dvc` file is a small YAML pointer containing the content hash (MD5 by default), file size, and relative path of the tracked artifact. Committing it to Git means Git versions the reference while DVC's cache/remote holds the bytes. Checking out any commit plus `dvc checkout` deterministically restores the matching data.

**Q: How does DVC avoid duplicating large files between the cache and your workspace?**
A: DVC links files from `.dvc/cache` into the workspace using reflinks (copy-on-write) where supported, falling back to hardlinks or symlinks, and only copying as a last resort. This means the same content is not stored twice on disk. You can configure the link strategy via `dvc config cache.type`.

### Pitfalls & best practices

- **Add the cache to `.gitignore`** — `.dvc/cache` should never be committed (DVC handles this by default).
- Prefer `dvc add` on **directories** rather than thousands of individual files to keep the number of `.dvc` files manageable.
- On Windows/NTFS, reflinks are unavailable; set `dvc config cache.type hardlink,symlink` and enable `cache.protected` to avoid accidental edits corrupting the cache.
- Commit the `.dvc` pointer and the `.gitignore` change together so collaborators get a consistent state.

---

## 3. Remote Storage — S3/GCS/Azure Remotes, `dvc push` / `dvc pull`

### Plain-language explanation

The local cache lives only on your machine. A **DVC remote** is shared storage — an S3 bucket, GCS bucket, Azure Blob container, SSH host, or even a mounted NAS — where the content-addressed cache is mirrored so teammates and CI can retrieve it. The mental model mirrors Git: `dvc push` uploads cached data to the remote (like `git push`), and `dvc pull` downloads it (like `git pull`). The difference is that Git moves code/pointers to the Git host, while DVC moves data blobs to object storage.

### Why it matters in production

Remotes are what make DVC a *team* tool rather than a local convenience. They separate the "control plane" (Git: code + pointers) from the "data plane" (object storage: bytes). This lets you use cheap, scalable, durable cloud storage for data, apply lifecycle/retention policies there, and keep the Git host lean. CI/CD pipelines pull exactly the data a commit references, guaranteeing reproducible builds.

### Real-world example / use case

A team stores its DVC remote in an S3 bucket with versioning and a lifecycle rule that transitions objects older than 90 days to Glacier. Developers `dvc push` after adding data; the CI pipeline runs `dvc pull` to fetch the precise dataset for the commit under test. Because the cache is content-addressed, identical data across branches is stored once, saving storage cost.

### Code or config example

```bash
# Configure an S3 remote as the default
dvc remote add -d storage s3://my-ml-bucket/dvcstore
dvc remote modify storage region us-east-1
# For credentials, prefer environment/instance roles over hardcoding:
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or an IAM instance profile

git add .dvc/config
git commit -m "Add S3 DVC remote"

# Push cached data to the remote
dvc push

# On another machine / in CI
git clone https://github.com/org/repo.git
cd repo
dvc pull            # fetches only the data referenced by current commit
```

Other backends:

```bash
# Google Cloud Storage
dvc remote add -d gcsremote gs://my-ml-bucket/dvcstore

# Azure Blob Storage
dvc remote add -d azremote azure://mycontainer/dvcstore
dvc remote modify --local azremote connection_string "<CONN_STRING>"

# S3-compatible (MinIO)
dvc remote add -d minio s3://bucket/path
dvc remote modify minio endpointurl http://minio.internal:9000
```

`.dvc/config` after adding the S3 remote:

```ini
[core]
    remote = storage
['remote "storage"']
    url = s3://my-ml-bucket/dvcstore
    region = us-east-1
```

### Interview questions

**Q: What is a DVC remote and how does `dvc push`/`pull` relate to `git push`/`pull`?**
A: A DVC remote is external shared storage (S3/GCS/Azure/SSH/etc.) holding the content-addressed data cache. `dvc push` uploads local cache blobs to it; `dvc pull` downloads the blobs referenced by the current commit's `.dvc`/`dvc.lock` files. Conceptually Git handles code and pointers, DVC handles the large data — you typically `git push` and `dvc push` together, and `git pull` then `dvc pull` on the other side.

**Q: How should credentials for a cloud remote be managed securely?**
A: Never hardcode keys in `.dvc/config` (it is committed to Git). Use IAM instance/role-based credentials, environment variables, or `dvc remote modify --local` which writes to `.dvc/config.local` (gitignored). In CI, inject secrets via the platform's secret store (e.g., GitHub Actions secrets, OIDC federation).

**Q: If two branches reference the same dataset, is it stored twice on the remote?**
A: No. Because the cache is content-addressed by hash, identical content is deduplicated — it is stored once and referenced by both branches' pointers.

### Pitfalls & best practices

- Use `--local` config or env vars for secrets; the main `.dvc/config` is committed to Git.
- Enable **bucket versioning** and lifecycle policies for durability and cost control.
- Set a `-d` (default) remote so `dvc push`/`pull` need no explicit target.
- Run `dvc gc` (garbage collect) carefully with `--cloud` to prune unused cache blobs — always scope it (e.g., `--all-branches --all-tags`) to avoid deleting data still referenced elsewhere.

---

## 4. Pipeline Reproducibility with `dvc.yaml` — Stages, Deps, Outs, Params, `dvc repro`, `dvc exp`

### Plain-language explanation

`dvc add` versions static files. But ML is a *pipeline*: raw data → cleaned data → features → trained model → metrics. `dvc.yaml` defines that pipeline as a **DAG of stages**. Each stage declares its command (`cmd`), inputs (`deps`), parameters (`params`), and outputs (`outs`). DVC computes hashes of every dependency and, when you run `dvc repro`, re-executes *only the stages whose inputs changed* — skipping unchanged stages (like a build system for ML). The resulting `dvc.lock` file records exact hashes, making the whole pipeline reproducible.

`dvc exp` (experiments) builds on this: you can tweak params, run `dvc exp run`, and DVC captures each experiment as a lightweight, comparable snapshot without polluting your Git branches.

### Why it matters in production

Reproducibility is the core promise of MLOps. With `dvc.yaml` + `dvc.lock` committed to Git, anyone can check out a commit, run `dvc repro`, and get the *same* model from the *same* data and code. Caching saves hours of recomputation. Experiment tracking lets teams sweep hyperparameters and compare metrics systematically instead of via ad-hoc scripts and spreadsheets.

### Real-world example / use case

A recommendation team runs a four-stage pipeline: prepare → featurize → train → evaluate. When a data scientist changes only the `train` learning rate in `params.yaml`, `dvc repro` skips prepare and featurize (unchanged hashes) and reruns only train and evaluate — cutting iteration time from 40 minutes to 8. They then run `dvc exp run --set-param train.lr=0.01,0.05,0.1` to sweep three learning rates and compare metrics with `dvc exp show`.

### Code or config example

A complete `dvc.yaml` with multiple stages:

```yaml
stages:
  prepare:
    cmd: python src/prepare.py data/raw data/prepared
    deps:
      - src/prepare.py
      - data/raw
    outs:
      - data/prepared

  featurize:
    cmd: python src/featurize.py data/prepared data/features
    deps:
      - src/featurize.py
      - data/prepared
    params:
      - featurize.max_features
      - featurize.ngram_range
    outs:
      - data/features

  train:
    cmd: python src/train.py data/features models/model.pkl
    deps:
      - src/train.py
      - data/features
    params:
      - train.n_estimators
      - train.max_depth
      - train.lr
    outs:
      - models/model.pkl

  evaluate:
    cmd: python src/evaluate.py models/model.pkl data/features
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/features
    metrics:
      - metrics.json:
          cache: false
    plots:
      - plots/roc.csv:
          cache: false
```

Corresponding `params.yaml`:

```yaml
featurize:
  max_features: 5000
  ngram_range: 2
train:
  n_estimators: 200
  max_depth: 12
  lr: 0.05
```

Running and experimenting:

```bash
# Run the full pipeline (only stale stages rerun)
dvc repro

# Commit the reproducible state
git add dvc.yaml dvc.lock params.yaml metrics.json
git commit -m "Train v2 pipeline"

# Run experiments without cluttering Git
dvc exp run --set-param train.lr=0.01
dvc exp run --set-param train.n_estimators=500

# Sweep multiple values
dvc exp run --set-param 'train.lr=0.01,0.05,0.1'

# Compare experiments
dvc exp show --only-changed
dvc metrics diff
dvc params diff

# Promote the best experiment to a branch/commit
dvc exp apply exp-a1b2c
```

### Interview questions

**Q: How does DVC decide which pipeline stages to re-run on `dvc repro`?**
A: DVC hashes each stage's declared `deps` (code, data), `params`, and outputs and compares them against `dvc.lock`. A stage is rerun only if any of its dependencies or params changed, or its outputs are missing. Downstream stages rerun if their upstream outputs changed. Unchanged stages are skipped and their cached outputs reused, giving fast, incremental, reproducible runs.

**Q: What is the difference between `dvc repro` and `dvc exp run`?**
A: `dvc repro` executes the pipeline and updates `dvc.lock` in your workspace, meant for the canonical committed state. `dvc exp run` runs the pipeline as a tracked *experiment* — a lightweight, hidden commit — letting you sweep params (`--set-param`), compare with `dvc exp show`, and later apply or drop experiments without creating messy Git branches.

**Q: What is `dvc.lock` and why must it be committed?**
A: `dvc.lock` is an auto-generated file recording the exact resolved hashes of every stage's deps, params, and outs from the last successful run. Committing it pins the pipeline to specific content so `dvc repro`/`dvc checkout` reproduce the identical result; without it, reproducibility is lost.

### Pitfalls & best practices

- **Declare all real dependencies.** Missing a dep means DVC won't rerun a stage when it should — silent staleness.
- Keep **metrics and plots with `cache: false`** so they are small and tracked in Git for easy diffing.
- Don't put outputs of one stage as untracked inputs of another — wire them explicitly via `deps`/`outs`.
- Commit `dvc.yaml`, `dvc.lock`, and `params.yaml` together; treat `dvc.lock` like a lockfile (never hand-edit).
- Use `dvc dag` to visualize the pipeline and catch missing/incorrect edges.

---

## 5. Dataset Lineage

### Plain-language explanation

**Lineage** answers "where did this artifact come from, and what produced it?" Because DVC records, in Git, the hashes of every input and output for every stage, you get an end-to-end provenance graph: this model came from *these* features, which came from *this* prepared data, which came from *this* raw dataset — all pinned to exact content hashes and a specific code commit. DVC can also import data from other DVC/Git repos (`dvc import`), tracking the *source* version so you know precisely which external dataset version you depend on.

### Why it matters in production

Lineage is essential for **debugging, auditing, and compliance**. When a model regresses in production, lineage lets you trace back to the exact data and code that produced it. In regulated industries (finance, healthcare), you must prove which data trained a deployed model — GDPR "right to be forgotten" and model audits demand it. Lineage also enables impact analysis: if a raw dataset is found to be corrupted, you can identify every downstream model built from it.

### Real-world example / use case

A bank's model risk team is audited. They must demonstrate that the credit-scoring model in production was trained on approved, bias-checked data. Using `dvc.lock` and the Git commit tied to the deployment, they reconstruct the full lineage: raw data hash → feature version → training code commit → model hash. They reproduce the model bit-for-bit with `dvc checkout && dvc repro`, satisfying the auditor.

### Code or config example

```bash
# Import a dataset from another DVC repo, pinning its version
dvc import https://github.com/org/datasets data/raw/customers.csv \
    --rev v2.3.0 -o data/raw/customers.csv
# Creates a .dvc file recording the source repo, path, and revision

# Update to the latest upstream version when ready
dvc update data/raw/customers.csv.dvc

# Inspect lineage of the whole pipeline
dvc dag                    # ASCII DAG of stages
dvc dag --outs             # show data artifacts in the graph

# See what produced a given output
dvc dag models/model.pkl
```

An imported `.dvc` pointer capturing source lineage:

```yaml
deps:
  - path: data/raw/customers.csv
    repo:
      url: https://github.com/org/datasets
      rev: v2.3.0
      rev_lock: 7f3c9a1b8e2d4f6a0c5b3d9e1a2f4c6b
outs:
  - md5: a3f5c9d1e2b4f6a8c0d2e4f6a8b0c2d4
    size: 104857600
    path: customers.csv
```

### Interview questions

**Q: How does DVC provide dataset and model lineage?**
A: Every stage in `dvc.yaml`/`dvc.lock` records the content hashes of its inputs and outputs, all committed to Git alongside the code. This creates a hash-pinned provenance graph tying raw data → intermediate artifacts → model → metrics to a specific code commit. `dvc dag` visualizes it, and checking out a commit reproduces the exact artifacts.

**Q: What does `dvc import` add over `dvc get` or copying a file?**
A: `dvc import` records the *source* repo URL, path, and revision in a `.dvc` file, so the dependency on an external dataset version is tracked and updatable via `dvc update`. `dvc get` (or a plain copy) just downloads bytes with no lineage link back to the source, so you lose provenance.

### Pitfalls & best practices

- Pin imports to a specific `--rev` (tag/commit) for reproducibility, not a moving branch like `main`.
- Tag meaningful dataset/model versions in Git (`git tag data-v1.0`) so lineage points to human-readable milestones.
- Combine DVC lineage with an experiment tracker (MLflow, W&B) for richer metrics/model-registry metadata — DVC gives content lineage, the tracker gives run metadata.

---

## 6. Integrating DVC with CI/CD

### Plain-language explanation

CI/CD for ML must handle data, not just code. Integrating DVC means your pipeline runner (GitHub Actions, GitLab CI, etc.) installs DVC, authenticates to the remote, runs `dvc pull` to fetch the exact data for the commit, executes `dvc repro` (or `dvc exp run`), and reports metrics — optionally pushing new artifacts back with `dvc push`. Tools like **CML (Continuous Machine Learning)** post metric reports and plots as comments on pull requests, turning model changes into reviewable events.

### Why it matters in production

This closes the reproducibility loop in automation. Every PR can automatically retrain or evaluate on the correct data and surface metric deltas, so reviewers see whether a change improves or regresses the model *before* merge. It also standardizes environment and data access, eliminating "works on my machine" for ML. Caching the DVC cache in CI speeds up repeated runs.

### Real-world example / use case

A team gates merges on model quality. On every PR, GitHub Actions pulls the dataset via DVC, reruns the pipeline, and CML posts a comment: "AUC 0.912 (+0.004 vs main), confusion matrix attached." If AUC drops below a threshold, the check fails and the PR is blocked. Approved changes trigger a `dvc push` of the new model to the remote for downstream deployment.

### Code or config example

GitHub Actions workflow (`.github/workflows/ml-pipeline.yml`):

```yaml
name: ML Pipeline
on:
  pull_request:
    branches: [main]

jobs:
  train-and-report:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write        # for CML comments
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install dvc[s3] dvclive

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/dvc-ci
          aws-region: us-east-1

      - name: Pull data with DVC
        run: dvc pull

      - name: Reproduce pipeline
        run: dvc repro

      - name: Report metrics with CML
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          echo "## Model Report" > report.md
          dvc metrics diff main --md >> report.md
          dvc plots diff main --show-vega-html > /dev/null || true
          npx github:iterative/cml comment create report.md

      - name: Push new artifacts
        if: github.ref == 'refs/heads/main'
        run: dvc push
```

GitLab CI equivalent (`.gitlab-ci.yml` snippet):

```yaml
train:
  image: iterativeai/cml:0-dvc3-base1
  variables:
    AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY
  script:
    - pip install -r requirements.txt
    - dvc pull
    - dvc repro
    - dvc metrics diff main --md >> report.md
    - cml comment create report.md
```

### Interview questions

**Q: What are the key steps to integrate DVC into a CI pipeline?**
A: Install DVC and the remote's extras (`dvc[s3]`), inject remote credentials via CI secrets/OIDC, run `dvc pull` to fetch the commit's data, run `dvc repro`/`dvc exp run` to reproduce the pipeline, report metrics (often via CML PR comments), and optionally `dvc push` new artifacts. Caching the DVC cache directory speeds up subsequent runs.

**Q: How do you handle cloud credentials for DVC in CI securely?**
A: Use the CI platform's secret store or, preferably, OIDC federation (e.g., GitHub Actions assuming an AWS IAM role) so no long-lived keys are stored. Never commit credentials to `.dvc/config`; use env vars or `--local` config that the runner injects at runtime.

**Q: What is CML and why pair it with DVC?**
A: CML (Continuous Machine Learning) is an open-source tool from the DVC team that turns CI runs into ML reports — posting metrics, tables, and plots as PR/MR comments and even provisioning cloud runners. Paired with DVC, it makes model changes reviewable: reviewers see metric deltas and visualizations inline before merging.

### Pitfalls & best practices

- **Cache the DVC cache** in CI (`actions/cache` keyed on `dvc.lock`) to avoid re-downloading unchanged data.
- Use **read-only** credentials for PR checks; grant push only on protected branches.
- Fail the build on metric regression thresholds to gate merges automatically.
- Pin DVC and tool versions in CI for deterministic runs.

---

## 7. Comparing DVC vs LakeFS vs Delta Lake

### Plain-language explanation

These three tools all touch "data versioning" but operate at different layers:

- **DVC** — Git-centric versioning for files/folders and ML pipelines. Best for versioning datasets *and models* alongside code, with reproducible pipelines and experiments. Storage-agnostic, file-oriented.
- **LakeFS** — Git-*like* version control for an entire **object-storage data lake**. It sits in front of S3/GCS/Azure and gives you branches, commits, and merges over petabyte-scale buckets *without copying data* (zero-copy, metadata-only branching). Format-agnostic (any files in the lake).
- **Delta Lake** — A **storage format / table layer** (built on Parquet + a transaction log) that adds ACID transactions, time travel, schema enforcement, and upserts to data lakes, primarily within the Spark/Databricks ecosystem. It versions *tables*, not arbitrary files or pipelines.

### Why it matters in production

Picking the wrong layer causes friction. DVC shines for ML experiment reproducibility on moderate data sizes tied to code. LakeFS shines when many teams share a huge lake and need isolated branches, atomic cross-collection commits, and rollbacks at lake scale. Delta Lake shines for analytical/streaming table workloads needing ACID guarantees and time travel inside Spark. They are often *complementary*: a team might store curated Delta tables in a lake managed by LakeFS branches, while DVC versions the ML training pipeline and models.

### Real-world example / use case

- A 6-person ML team versioning a 20 GB dataset and model checkpoints tied to their Git repo → **DVC**.
- A data platform team giving 50 engineers isolated, revertible branches over a 5 PB S3 data lake, with atomic "promote to production" merges → **LakeFS**.
- A data-engineering team running Spark streaming upserts into fact tables that must be ACID-consistent and queryable "as of yesterday" → **Delta Lake**.

### Comparison table

| Dimension | DVC | LakeFS | Delta Lake |
|---|---|---|---|
| **Primary abstraction** | Files/folders + ML pipelines | Branches/commits over an object-store data lake | ACID table format (Parquet + transaction log) |
| **Versioning granularity** | File/dir (content-hash pointers in Git) | Whole-lake, metadata-only branches (zero-copy) | Table-level (versions/snapshots) |
| **Coupled to Git?** | Yes — versions live in Git commits | No — Git-*like* semantics, own metadata store | No — versioning via transaction log |
| **ML pipelines / experiments** | First-class (`dvc.yaml`, `dvc exp`) | No (data-lake focused) | No (table/analytics focused) |
| **ACID transactions** | No | Atomic commits/merges over the lake | Yes (per-table ACID, upserts/deletes) |
| **Time travel / rollback** | Via Git checkout + `dvc checkout` | Branch/commit checkout & revert | `VERSION AS OF` / `TIMESTAMP AS OF` |
| **Scale sweet spot** | GBs–low TBs tied to code | Petabyte-scale shared lakes | Large analytical/streaming tables |
| **Ecosystem** | Git, any object store, Python/CLI | S3/GCS/Azure, any engine via S3 API | Spark/Databricks (also Trino, Flink) |
| **Best for** | Reproducible ML: data + model + pipeline | Data-lake-wide branching, isolation, rollback | ACID analytics, upserts, schema evolution |
| **Not ideal for** | Petabyte multi-team lakes; ACID tables | Model/pipeline/experiment tracking | Arbitrary file/model versioning; pipelines |

### Interview questions

**Q: When would you choose DVC over LakeFS or Delta Lake?**
A: Choose DVC when the goal is reproducible ML tied to code — versioning datasets *and* model artifacts, defining pipelines, and running experiments — at moderate (GB–low-TB) scale with Git as the source of truth. LakeFS is for Git-like branching over a huge shared data lake; Delta Lake is for ACID table storage in Spark. If your problem is "reproduce this model from this data and code," DVC is the natural fit.

**Q: How does LakeFS avoid copying data when you create a branch?**
A: LakeFS versions *metadata*, not the objects themselves. Creating a branch is a metadata operation that points to the same underlying objects (copy-on-write). New writes create new objects; unchanged data is shared. This makes branching a petabyte-scale lake instant and cheap, unlike physically copying data.

**Q: What core capability does Delta Lake add that plain Parquet on S3 lacks?**
A: A transaction log (the `_delta_log`) providing ACID transactions, so concurrent reads/writes are consistent, plus upserts/deletes (MERGE), schema enforcement/evolution, and time travel (`VERSION AS OF`). Plain Parquet has none of these — partial writes and concurrent updates can corrupt or produce inconsistent reads.

**Q: Can these tools be used together?**
A: Yes. They live at different layers, so a stack might store curated Delta tables inside a lake governed by LakeFS branches for isolation/rollback, while DVC versions the ML pipeline, params, and model artifacts derived from that data. Each solves a distinct concern: table ACID, lake branching, and ML reproducibility respectively.

### Pitfalls & best practices

- Don't force DVC to manage a multi-team petabyte lake — that's LakeFS territory.
- Don't expect Delta Lake to version your code, models, or pipelines — it versions tables.
- Match the tool to the *layer* of the problem; combine tools rather than overloading one.

---

## Summary

Git revolutionized code collaboration but was never designed for the large binary datasets and models at the heart of machine learning — committing them causes permanent history bloat, slow clones, and host size-limit rejections. **DVC** closes this gap by keeping tiny content-hash pointer files in Git while the actual bytes live in a content-addressed cache and shared **remote** (S3/GCS/Azure). `dvc add` tracks artifacts, `dvc push`/`pull` synchronize them across the team and CI, and checking out any commit plus `dvc checkout` deterministically restores the matching data.

Beyond storage, DVC is a **reproducibility engine**. `dvc.yaml` defines an ML pipeline as a DAG of stages with explicit deps, params, and outs; `dvc repro` reruns only what changed; `dvc.lock` pins exact hashes; and `dvc exp` enables systematic experiment tracking. Because every input and output hash is committed to Git, DVC delivers end-to-end **lineage** for debugging, auditing, and compliance. Integrating DVC into **CI/CD** — pulling data, reproducing pipelines, and reporting metrics via CML — makes model changes reviewable and reproducible in automation.

Finally, DVC is one tool in a layered ecosystem. **LakeFS** brings Git-like, zero-copy branching to petabyte data lakes, and **Delta Lake** adds ACID transactions and time travel to lake tables. Choose DVC for reproducible ML tied to code, LakeFS for lake-wide branching and isolation, and Delta Lake for ACID analytics — and combine them when the problem spans multiple layers.

## Chapter Q&A

**Q1: Why is committing large data or model files directly to Git a bad idea?**
A: Git keeps full history in every clone and delta-compresses binaries poorly, so large files bloat `.git` permanently, slowing clones and fetches. Hosts enforce per-file limits (~100 MB on GitHub). Removing a large file later requires history rewrites that break all clones. And Git alone doesn't tie a code commit to the exact data used, so reproducibility is not guaranteed.

**Q2: Walk me through what happens when you run `dvc add data/train.csv`.**
A: DVC moves the file's bytes into the content-addressed cache (`.dvc/cache`, keyed by MD5), creates a small `data/train.csv.dvc` pointer file recording the hash/size/path, adds the real file to `.gitignore`, and links the cached content back into the workspace (reflink/hardlink/symlink). You then commit the `.dvc` pointer and `.gitignore` to Git.

**Q3: How do you restore the exact dataset that matched an old code version?**
A: Check out the Git commit or tag (`git checkout v1.0`) — this restores the code and the `.dvc`/`dvc.lock` pointers — then run `dvc checkout` (and `dvc pull` if the data isn't cached locally) to restore the data files to the hashes those pointers reference.

**Q4: What is a DVC remote and how is it different from the local cache?**
A: The local cache (`.dvc/cache`) holds data on your machine only. A remote is shared external storage (S3/GCS/Azure/SSH) that mirrors the cache so teammates and CI can retrieve data. `dvc push` uploads cache blobs to the remote; `dvc pull` downloads the blobs referenced by the current commit.

**Q5: Explain how DVC decides which stages to re-run during `dvc repro`.**
A: DVC hashes each stage's deps, params, and outs and compares them to `dvc.lock`. Only stages with changed inputs/params, missing outputs, or changed upstream outputs are re-executed; unchanged stages reuse cached outputs. This gives fast, incremental, reproducible pipeline runs.

**Q6: What roles do `dvc.yaml`, `dvc.lock`, and `params.yaml` play?**
A: `dvc.yaml` declares the pipeline stages (cmd/deps/params/outs/metrics). `params.yaml` holds tunable hyperparameters referenced by stages. `dvc.lock` is auto-generated and records the exact resolved hashes from the last run, pinning the pipeline for reproducibility. All three are committed to Git; `dvc.lock` is never hand-edited.

**Q7: How does DVC support experiment tracking, and how does it differ from committing each run?**
A: `dvc exp run` executes the pipeline as a lightweight, hidden experiment commit, letting you sweep params (`--set-param 'train.lr=0.01,0.05,0.1'`), compare with `dvc exp show`/`dvc metrics diff`, and apply or drop experiments without creating messy Git branches. It avoids polluting history with dozens of trial commits while keeping results reproducible.

**Q8: How does DVC provide data and model lineage, and why does it matter?**
A: Every stage's input and output hashes are recorded in `dvc.lock` and committed to Git, forming a hash-pinned provenance graph from raw data through features to model and metrics, tied to a code commit. `dvc dag` visualizes it. This enables debugging regressions, impact analysis on corrupted data, and audit/compliance proof of exactly what data and code produced a deployed model.

**Q9: Describe a secure, efficient DVC-in-CI setup.**
A: Install `dvc[s3]`, authenticate to the remote via OIDC/role assumption or CI secrets (never hardcoded), `dvc pull` the commit's data, `dvc repro` the pipeline, report metrics with CML PR comments, and `dvc push` new artifacts only on protected branches. Cache the DVC cache keyed on `dvc.lock` for speed, use read-only creds for PR checks, and gate merges on metric thresholds.

**Q10: When would you reach for LakeFS or Delta Lake instead of, or alongside, DVC?**
A: Use LakeFS when many teams need Git-like, zero-copy branches, isolation, and atomic rollbacks over a petabyte-scale object-store data lake. Use Delta Lake when you need ACID transactions, upserts/deletes, schema evolution, and time travel on analytical/streaming tables in the Spark ecosystem. Use DVC for reproducible ML tying data, models, pipelines, and experiments to code. They operate at different layers and are often combined.
