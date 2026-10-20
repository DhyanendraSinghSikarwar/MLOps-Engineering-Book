# Chapter 8 — Terraform / Infrastructure as Code

Machine learning systems are only as reliable as the infrastructure beneath them. A model that trains perfectly on your laptop is worthless if you cannot reproducibly stand up the GPU cluster, object storage, networking, and IAM policies that let it serve traffic at scale. Infrastructure as Code (IaC) turns that infrastructure into version-controlled, reviewable, repeatable artifacts — and Terraform has become the de facto standard for declaring cloud resources across AWS, GCP, Azure, and beyond.

This chapter builds from HCL fundamentals up to provisioning GPU node pools and designing reproducible, multi-cloud ML platforms. The goal is not only to teach you the syntax, but to give you the production judgment — and interview vocabulary — that separates someone who *ran* `terraform apply` once from an engineer who *owns* a platform. Throughout, we anchor examples in real MLOps concerns: reproducibility, cost control of expensive accelerators, state safety, and blast-radius management.

---

## 1. HCL Basics: Resources, Variables, Outputs, Data Sources, and the Plan/Apply Lifecycle

### Plain-language explanation

Terraform is a declarative tool: instead of writing a script that says "create this, then that," you *describe the desired end state* of your infrastructure in HashiCorp Configuration Language (HCL), and Terraform figures out the actions needed to reach it. The core building blocks are:

- **Resources** — the actual infrastructure objects (an S3 bucket, an EKS cluster, a VPC).
- **Variables** — typed inputs that parameterize your configuration.
- **Outputs** — values exported from your configuration for humans or other modules to consume.
- **Data sources** — read-only queries that fetch information about existing infrastructure (e.g., "the latest EKS-optimized GPU AMI").
- **The plan/apply lifecycle** — `terraform plan` computes the diff between desired and current state; `terraform apply` executes it.

Terraform maintains a **state file** that maps your HCL declarations to real-world resource IDs. This mapping is what lets Terraform know whether a resource already exists, needs updating, or must be created.

### Why it matters in production

In an ML platform you might manage hundreds of resources across dev, staging, and prod. Declarative IaC gives you a *reviewable diff* before every change — critical when a typo could tear down a GPU cluster costing thousands of dollars a day. The plan/apply separation means changes are auditable in pull requests and gated by CI, giving you the same code-review discipline for infrastructure that you already have for application code.

### Real-world example / use case

Your ML team needs an S3 bucket for training datasets and model artifacts, with versioning enabled so you never lose a model checkpoint and can roll back to any prior artifact. You want the bucket name to differ by environment, and you want to output the bucket ARN so a downstream IAM module can grant your training jobs access.

### Code or config example

```hcl
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# --- Variables (typed inputs) ---
variable "region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type        = string
  description = "Deployment environment (dev/staging/prod)"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of dev, staging, prod."
  }
}

# --- Data source: read-only lookup of caller account ---
data "aws_caller_identity" "current" {}

# --- Resource: versioned artifact bucket ---
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "mlops-artifacts-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Environment = var.environment
    ManagedBy   = "terraform"
    Team        = "mlops"
  }
}

resource "aws_s3_bucket_versioning" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# --- Output: expose the ARN for downstream consumers ---
output "artifacts_bucket_arn" {
  value       = aws_s3_bucket.ml_artifacts.arn
  description = "ARN of the ML artifacts bucket"
}
```

```bash
terraform init      # download providers, initialize backend
terraform fmt       # canonical formatting
terraform validate  # static config check
terraform plan      # show the diff (no changes made)
terraform apply     # execute after review
```

### Interview questions

**Q: What is the difference between `terraform plan` and `terraform apply`, and why does the separation matter?**
A: `plan` computes an execution plan — the set of create/update/delete actions needed to reconcile desired state (your HCL) with current state (the state file plus real-world refresh) — without changing anything. `apply` executes that plan. The separation matters because it enables review and safety: you can inspect the diff in CI or a PR, catch destructive changes (like a resource replacement that would delete a database), and require approval before mutating production. In practice you often run `terraform plan -out=plan.tfplan` and then `apply plan.tfplan` to guarantee that what was reviewed is exactly what gets executed.

**Q: What is the difference between a resource and a data source?**
A: A *resource* is something Terraform manages — it creates, updates, and destroys it, and tracks it in state. A *data source* is a read-only query for information about infrastructure that already exists (whether created by Terraform or not), such as looking up an AMI ID or an existing VPC. Data sources let you reference external facts without owning their lifecycle.

**Q: How does Terraform know whether to create, update, or destroy a resource?**
A: It compares three things: the desired configuration in your HCL, the last-known state in the state file, and the actual current state fetched by refreshing from the provider API. The delta drives the plan. If a resource is in state but absent from config, it plans a destroy; if in config but not state, a create; if attributes differ, an in-place update or a replacement depending on whether the changed attribute forces recreation.

### Pitfalls & best practices

- **Never hand-edit the state file.** Use `terraform state` subcommands instead.
- **Pin provider versions** with `~>` to avoid surprise breaking changes on `init`.
- **Use variable `validation` blocks** to fail fast on bad inputs rather than mid-apply.
- **Run `terraform fmt` and `validate` in CI** so style and syntax errors never reach review.
- **Be alert to "forces replacement"** in plan output — for stateful resources this can mean data loss.

---

## 2. Providers and Modules: Writing and Consuming Reusable Modules

### Plain-language explanation

A **provider** is a plugin that teaches Terraform how to talk to a specific API — AWS, Google Cloud, Kubernetes, Helm, Datadog, and thousands more. You configure providers with credentials and region, and they expose the resources and data sources you use.

A **module** is a reusable, parameterized package of Terraform configuration — essentially a function for infrastructure. It has inputs (variables), a body (resources), and outputs. The top-level directory you run `terraform apply` in is the *root module*; any module it calls is a *child module*. Modules can come from the local filesystem, a Git repository, or the Terraform Registry.

### Why it matters in production

Without modules, every environment copy-pastes the same 400 lines of VPC and cluster config, and they drift apart over time. Modules give you a single, versioned source of truth for "how we build an ML training cluster," so dev, staging, and prod stay consistent. They encode organizational standards — mandatory tags, encryption, logging — so individual teams cannot accidentally opt out of security requirements.

### Real-world example / use case

Your platform team publishes a `gpu-training-cluster` module. Data science teams consume it with three lines, passing their team name and desired GPU count. The module internally wires up the node group, IAM roles, autoscaling, and taints — none of which the consuming team needs to understand.

### Code or config example

A minimal reusable module (`modules/ml-bucket/main.tf`):

```hcl
variable "name_prefix" { type = string }
variable "environment" { type = string }
variable "force_destroy" {
  type    = bool
  default = false
}

resource "aws_s3_bucket" "this" {
  bucket        = "${var.name_prefix}-${var.environment}"
  force_destroy = var.force_destroy
  tags          = { Environment = var.environment, ManagedBy = "terraform" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  bucket = aws_s3_bucket.this.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

output "bucket_id"  { value = aws_s3_bucket.this.id }
output "bucket_arn" { value = aws_s3_bucket.this.arn }
```

Consuming the module from a root config:

```hcl
module "datasets" {
  source      = "./modules/ml-bucket"
  name_prefix = "acme-ml-datasets"
  environment = var.environment
}

module "checkpoints" {
  source      = "git::https://github.com/acme/tf-modules.git//ml-bucket?ref=v1.4.0"
  name_prefix = "acme-ml-checkpoints"
  environment = var.environment
}

output "dataset_bucket" {
  value = module.datasets.bucket_arn
}
```

### Interview questions

**Q: What makes a good, reusable Terraform module?**
A: Clear, typed inputs with sensible defaults and descriptions; well-defined outputs so consumers don't reach into internals; a single responsibility (a VPC module shouldn't also manage IAM users); no hardcoded environment-specific values; semantic version tags when sourced from Git; and documentation of required providers. Crucially, a good module does *not* configure providers itself — provider config is the root module's responsibility so consumers control credentials and region.

**Q: How do you version and pin a module sourced from Git?**
A: Use the `git::` source with a `?ref=` pointing at a tag or commit SHA, e.g. `git::https://github.com/acme/tf-modules.git//path?ref=v1.4.0`. Pinning to a tag or SHA guarantees reproducible builds; pinning to a branch is dangerous because the branch can move under you between applies.

**Q: What is the difference between the root module and a child module regarding provider configuration?**
A: Providers should be configured in the root module and passed implicitly (or explicitly via the `providers` argument for aliased providers). Child modules should declare `required_providers` in `terraform` blocks but not configure credentials. This keeps modules portable and lets the root control which account/region resources land in.

### Pitfalls & best practices

- **Don't configure providers inside shared modules** — it breaks reuse and multi-region use.
- **Version-pin module sources**; a floating `main` branch destroys reproducibility.
- **Keep modules small and composable** rather than one god-module with 50 toggles.
- **Expose outputs deliberately** — they are your module's public API and hard to remove later.
- **Avoid over-abstraction early**; extract a module only after you've copy-pasted the pattern two or three times.

---

## 3. State Management: Remote State, S3 + DynamoDB Locking, Workspaces, Drift, and Import

### Plain-language explanation

Terraform's **state** is the JSON file that records which real resources correspond to your configuration. By default it lives locally as `terraform.tfstate` — fine for a solo experiment, disastrous for a team, because two people applying simultaneously will corrupt it, and the file often contains secrets.

**Remote state** stores this file in a shared, durable backend (S3, GCS, Terraform Cloud). **State locking** prevents concurrent writes: with the S3 backend, a DynamoDB table (or, in newer setups, S3-native locking) holds a lock so only one `apply` runs at a time. **Workspaces** let one configuration manage multiple independent state files (e.g., per-environment). **Drift** is when reality diverges from state — someone clicked in the console and changed a resource Terraform thinks it owns. **Import** brings an existing, unmanaged resource under Terraform control.

### Why it matters in production

State is the single most operationally sensitive part of Terraform. A corrupted or lost state file can mean Terraform tries to recreate resources that already exist, or worse, loses track of them entirely. Remote state with locking is non-negotiable for any team. Drift detection catches out-of-band changes before they cause a confusing apply. Import lets you adopt legacy infrastructure without a risky teardown.

### Real-world example / use case

Your MLOps team of eight engineers shares one AWS account. You put state in an encrypted S3 bucket with DynamoDB locking so no two `apply` runs collide. Someone manually resizes a GPU node group in the console during an incident; your nightly `terraform plan` in CI flags the drift, and you reconcile it back into code.

### Code or config example

Remote backend with S3 and DynamoDB locking:

```hcl
terraform {
  backend "s3" {
    bucket         = "acme-terraform-state"
    key            = "mlops/gpu-platform/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
```

Bootstrapping the backend resources (run once with local state, then migrate):

```hcl
resource "aws_s3_bucket" "tf_state" {
  bucket = "acme-terraform-state"
}

resource "aws_s3_bucket_versioning" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_dynamodb_table" "tf_locks" {
  name         = "terraform-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"
  attribute {
    name = "LockID"
    type = "S"
  }
}
```

Workspaces and import in practice:

```bash
# Workspaces: isolate state per environment
terraform workspace new staging
terraform workspace select staging
terraform workspace list

# Detect drift
terraform plan -refresh-only

# Import an existing resource into state (Terraform 1.5+ config-driven import)
```

```hcl
import {
  to = aws_s3_bucket.legacy_datasets
  id = "acme-legacy-datasets-bucket"
}

resource "aws_s3_bucket" "legacy_datasets" {
  bucket = "acme-legacy-datasets-bucket"
}
```

```bash
terraform plan   # generates the import plan
terraform apply  # brings the bucket under management
```

### Interview questions

**Q: Why do you need a remote backend with locking, and how does S3 + DynamoDB provide it?**
A: A remote backend gives you a shared, durable, versioned store so a whole team works from one state, and it keeps secrets out of local disks and Git. Locking prevents two concurrent `apply` operations from racing and corrupting state. With the S3 backend, S3 stores the state (with versioning for recovery and encryption at rest) while a DynamoDB table holds a lock item keyed by `LockID`; Terraform acquires the lock before writing and releases it after, so a second apply blocks or fails fast rather than clobbering the first. (Recent Terraform versions also support S3-native conditional-write locking, reducing the need for DynamoDB.)

**Q: What is state drift and how do you detect and remediate it?**
A: Drift is divergence between the state file and real infrastructure, usually from manual console changes or another tool. Detect it with `terraform plan` (which refreshes) or `terraform plan -refresh-only`, ideally on a schedule in CI. Remediate by either updating your HCL to match the intended reality and applying, or, if the manual change was undesired, applying to revert it. The long-term fix is preventing out-of-band changes via IAM guardrails and read-only console access in prod.

**Q: When would you use workspaces versus separate state files / directories per environment?**
A: Workspaces are lightweight — one configuration, multiple state files — good for small variations or ephemeral environments. But they hide the environment in a CLI selection, which is easy to get wrong (applying prod thinking you're in dev). For strongly isolated environments with different variables, backends, or blast radii, many teams prefer separate directories/state keys (or separate backends/accounts) so environment separation is explicit and hard to fumble.

### Pitfalls & best practices

- **Always enable S3 versioning and encryption** on the state bucket for recovery and secret protection.
- **Bootstrap the backend carefully** — the state bucket/table can't live in the state they hold on first creation; create them, then migrate.
- **Treat state as secret** — it can contain passwords and keys in plaintext.
- **Prefer `terraform state mv` / `import`** over editing state by hand.
- **Beware workspace footguns** — always confirm `terraform workspace show` before applying to prod.
- **Run drift detection on a schedule** so surprises surface early, not during an emergency change.

---

## 4. Provisioning ML Infrastructure: GPU Node Pools, Object Storage, and Networking

### Plain-language explanation

ML workloads need three things above ordinary web infrastructure: **accelerated compute** (GPU nodes), **large durable storage** (object stores for datasets and checkpoints), and **networking** (a VPC with subnets, so training and serving pods can reach data and each other securely). On Kubernetes-based platforms you provision *GPU node pools* (GKE) or *GPU node groups* (EKS) — pools of GPU-equipped instances, usually with autoscaling and taints so only GPU workloads schedule onto expensive hardware.

### Why it matters in production

GPUs are the single largest cost in most ML platforms. IaC lets you codify autoscaling from zero, spot/preemptible usage, taints and labels, and the exact instance types — so you don't leave a $30/hour node idle overnight. Codifying networking ensures reproducible, secure isolation, and codifying storage guarantees encryption and lifecycle policies are never forgotten.

### Real-world example / use case

You run distributed training on 8×A100 nodes that must scale to zero when idle. You provision an EKS cluster with a GPU-tainted managed node group (min 0, max 4), plus a GKE example for a second region. Datasets live in an encrypted object store; the VPC has private subnets so nodes have no public IPs.

### Code or config example

EKS GPU managed node group with autoscaling and a GPU taint:

```hcl
data "aws_ssm_parameter" "eks_gpu_ami" {
  name = "/aws/service/eks/optimized-ami/1.29/amazon-linux-2-gpu/recommended/image_id"
}

resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "gpu-a100"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.private[*].id

  # p4d.24xlarge = 8x A100. Scale to zero when idle to control cost.
  instance_types = ["p4d.24xlarge"]
  capacity_type  = "ON_DEMAND"
  ami_type       = "AL2_x86_64_GPU"

  scaling_config {
    desired_size = 0
    min_size     = 0
    max_size     = 4
  }

  # Taint so only GPU-tolerant pods schedule here
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  labels = {
    "workload-type"     = "gpu-training"
    "nvidia.com/gpu"    = "present"
  }

  update_config { max_unavailable = 1 }

  tags = { Environment = var.environment, ManagedBy = "terraform" }
}
```

GKE GPU node pool equivalent:

```hcl
resource "google_container_node_pool" "gpu" {
  name     = "gpu-a100-pool"
  cluster  = google_container_cluster.this.id
  location = var.region

  autoscaling {
    min_node_count = 0
    max_node_count = 4
  }

  node_config {
    machine_type = "a2-highgpu-1g"
    guest_accelerator {
      type  = "nvidia-tesla-a100"
      count = 1
    }
    # Preemptible/spot for cheaper batch training
    spot = true

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
    labels = { workload-type = "gpu-training" }

    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}
```

Networking (private subnets) and object storage tie it together:

```hcl
resource "aws_vpc" "ml" {
  cidr_block           = "10.20.0.0/16"
  enable_dns_hostnames = true
  tags                 = { Name = "ml-platform-vpc" }
}

resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.ml.id
  cidr_block        = cidrsubnet(aws_vpc.ml.cidr_block, 4, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = {
    Name                              = "ml-private-${count.index}"
    "kubernetes.io/role/internal-elb" = "1"
  }
}
```

### Interview questions

**Q: Why do you taint GPU node groups, and how do workloads land on them?**
A: GPU instances are expensive, so you taint them (e.g., `nvidia.com/gpu=present:NoSchedule`) to repel ordinary pods; only pods that declare a matching *toleration* — and typically a `nvidia.com/gpu` resource request — will schedule there. Combined with node labels and `nodeSelector`/affinity, this guarantees the pricey hardware is reserved for GPU workloads and not wasted on stateless web pods. You'd also install the NVIDIA device plugin so Kubernetes can advertise and allocate GPUs.

**Q: How do you make a GPU node group cost-efficient in Terraform?**
A: Set `min_size = 0` so the cluster autoscaler can scale the pool to zero when there are no pending GPU pods; use spot/preemptible capacity for interruptible batch training; right-size instance types to the model (don't put a single-GPU job on an 8-GPU node); and use separate pools per accelerator type so autoscaling is precise. Tagging enables cost allocation reporting per team.

**Q: Why provision the VPC and networking in the same IaC as the cluster?**
A: The cluster, node groups, load balancers, and storage endpoints all depend on subnet IDs, routing, and security groups. Managing them together (or via well-defined module outputs) keeps dependencies explicit, lets Terraform order creation correctly, and ensures reproducible, isolated environments — private subnets with no public IPs for nodes, plus NAT or VPC endpoints for controlled egress.

### Pitfalls & best practices

- **Match the GPU AMI/driver to the cluster version** — a mismatched EKS GPU AMI means CUDA won't initialize.
- **Set `min_size = 0` plus cluster autoscaler** to avoid idle GPU spend.
- **Separate node groups per accelerator type** for clean autoscaling and scheduling.
- **Use private subnets and VPC endpoints** so nodes reach S3/ECR without public exposure.
- **Remember spot interruptions** — checkpoint training frequently so preemption doesn't lose hours of work.
- **Install the NVIDIA device plugin** (often via a Helm provider resource) or GPUs won't be schedulable.

---

## 5. IaC for Reproducible ML Environments

### Plain-language explanation

Reproducibility means anyone can recreate an identical environment — same cluster version, same instance types, same drivers, same storage layout — from code alone, in any region or account. In ML this extends to pinning not just infrastructure versions but the whole platform: Kubernetes version, Helm chart versions for tooling (Kubeflow, MLflow, KServe), and the container images the platform relies on.

### Why it matters in production

ML results must be reproducible for debugging, compliance, and trust. "It worked in staging" is meaningless if staging and prod were built by hand and differ subtly. IaC with pinned versions and parameterized modules lets you spin up a byte-for-byte staging clone of prod, reproduce a training issue, and tear it down — and it lets you rebuild the platform from scratch after a regional outage.

### Real-world example / use case

A regulator asks you to reproduce the exact environment a model was trained in six months ago. Because your infrastructure, Kubernetes version, and Helm chart versions were all pinned in Git tags, you check out the tag and `terraform apply` into a fresh account, recreating the environment deterministically.

### Code or config example

Pinning everything and parameterizing per environment:

```hcl
terraform {
  required_version = "= 1.7.5"
  required_providers {
    aws  = { source = "hashicorp/aws",  version = "= 5.40.0" }
    helm = { source = "hashicorp/helm", version = "= 2.12.1" }
  }
}

locals {
  env_config = {
    staging = { k8s_version = "1.29", gpu_max = 2 }
    prod    = { k8s_version = "1.29", gpu_max = 8 }
  }
  cfg = local.env_config[var.environment]
}

module "ml_platform" {
  source          = "git::https://github.com/acme/tf-modules.git//ml-platform?ref=v2.3.0"
  environment     = var.environment
  cluster_version = local.cfg.k8s_version
  gpu_max_nodes   = local.cfg.gpu_max
}

# Pin the exact chart version of the training platform
resource "helm_release" "mlflow" {
  name       = "mlflow"
  repository = "https://community-charts.github.io/helm-charts"
  chart      = "mlflow"
  version    = "0.7.19"   # pinned, not "latest"
  namespace  = "mlflow"
}
```

### Interview questions

**Q: What does "reproducible infrastructure" require beyond just committing Terraform to Git?**
A: You must pin *everything* deterministic: `required_version` for Terraform itself, exact provider versions, module `?ref=` tags, Kubernetes version, Helm chart versions, and container image digests. You also need the state and secrets management to be reconstructable, and environment differences expressed as parameters rather than divergent code. Only then does checking out a Git tag and applying give you an identical environment.

**Q: How do you keep staging a faithful mirror of production?**
A: Use the same modules for both, driven by a small per-environment config map (locals or tfvars) that varies only scale and non-functional parameters — never the resource definitions themselves. That way the *shape* of the infrastructure is identical and only capacity differs, so bugs surface in staging before prod.

### Pitfalls & best practices

- **Never use `latest`** for AMIs, chart versions, or images in reproducible environments — pin digests/versions.
- **Express environment differences as parameters**, not forked configs.
- **Tag your infra releases** in Git so you can reproduce any historical environment.
- **Reproduce, don't snowflake** — forbid manual changes; if it's not in code it doesn't exist.
- **Store tfvars per environment** and keep secrets out of them (use a secrets manager).

---

## 6. Multi-Cloud Considerations

### Plain-language explanation

Multi-cloud means running infrastructure across more than one provider (AWS + GCP, say) — for resilience, to use best-of-breed services (TPUs on GCP, specific instance availability on AWS), for cost arbitrage, or to satisfy customer/regulatory requirements. Terraform supports this because it has providers for all major clouds, but Terraform does *not* magically abstract the clouds away: an S3 bucket and a GCS bucket are still different resources with different arguments.

### Why it matters in production

GPU capacity shortages are real — a team may need to burst training to whichever cloud has A100/H100 availability. Multi-cloud also avoids lock-in and can be a compliance requirement. But it multiplies complexity: two IAM models, two networking stacks, two sets of quotas. IaC is what makes multi-cloud tractable at all, by giving you one workflow and one review process across providers.

### Real-world example / use case

Your inference serving runs on AWS EKS for latency to your users, but you burst large training jobs to GKE to access TPU and cheaper preemptible A100 capacity. A single Terraform configuration uses both the `aws` and `google` providers, with a shared object-storage module interface implemented per cloud.

### Code or config example

Multiple providers with aliases, and a thin abstraction over storage:

```hcl
provider "aws" {
  region = "us-east-1"
}

provider "google" {
  project = var.gcp_project
  region  = "us-central1"
}

# Inference cluster on AWS
module "inference" {
  source      = "./modules/eks-gpu-cluster"
  environment = var.environment
  # implicitly uses the aws provider
}

# Burst training on GCP
module "training_burst" {
  source      = "./modules/gke-gpu-cluster"
  environment = var.environment
  providers = {
    google = google
  }
}

# Same logical bucket concept, different implementations
output "endpoints" {
  value = {
    inference_cluster = module.inference.cluster_endpoint
    training_cluster  = module.training_burst.cluster_endpoint
  }
}
```

### Interview questions

**Q: Does Terraform abstract away differences between clouds?**
A: No. Terraform provides a *consistent workflow and language* across providers, but each provider exposes cloud-specific resources. You still write `aws_s3_bucket` vs `google_storage_bucket` with different schemas. True abstraction requires you to build it yourself — e.g., a module interface with per-cloud implementations behind a common set of inputs/outputs. Terraform unifies *how* you provision, not *what* you provision.

**Q: What are the main challenges of managing multi-cloud infrastructure with Terraform?**
A: Distinct IAM/auth models and credential management per provider; separate networking, quotas, and regional availability; different resource schemas that prevent code reuse; more complex state (often one backend but many providers); and increased operational surface (monitoring, cost, drift across clouds). You also need provider aliasing when using the same provider in multiple regions/accounts. The payoff — resilience, capacity access, and reduced lock-in — must justify this overhead.

**Q: How do provider aliases help in multi-cloud or multi-region setups?**
A: Aliases let you configure the same provider multiple times (e.g., `aws` in us-east-1 and an aliased `aws.eu` in eu-west-1) and pass the right one to each module or resource via the `providers` argument. This is essential for cross-region replication or deploying identical modules to different accounts/regions from one configuration.

### Pitfalls & best practices

- **Don't adopt multi-cloud for its own sake** — the operational cost is high; do it for a concrete driver (resilience, capacity, compliance).
- **Build per-cloud modules with a common interface** rather than pretending resources are interchangeable.
- **Manage credentials per provider carefully** — use OIDC/workload identity, not long-lived keys.
- **Watch quotas and regional GPU availability** — they differ sharply across clouds.
- **Use provider aliases** for multi-region/multi-account fan-out.
- **Centralize state and CI** so one workflow governs all clouds.

---

## Summary

Terraform turns cloud infrastructure into reviewable, versioned, reproducible code — a foundational capability for any serious ML platform. We began with HCL fundamentals: resources, typed variables, outputs, data sources, and the plan/apply lifecycle that makes changes safe and auditable. We saw how providers connect Terraform to cloud APIs and how modules package infrastructure into reusable, versioned building blocks with clean input/output contracts.

State management is where Terraform is most operationally sensitive: remote backends (S3 with DynamoDB or S3-native locking), encryption and versioning, workspaces, drift detection, and config-driven import are the tools that keep a team's shared reality consistent. We then applied all of this to ML-specific infrastructure — GPU node groups on EKS and node pools on GKE with taints, autoscaling to zero, and spot capacity for cost control, alongside private-subnet networking and encrypted object storage. Reproducibility ties the chapter together: by pinning Terraform, provider, module, Kubernetes, and Helm versions, you can rebuild an identical environment from a Git tag alone. Finally, multi-cloud extends the same disciplined workflow across providers — powerful for resilience and capacity, but a deliberate trade-off in complexity, not a free abstraction.

The through-line: treat infrastructure with the same rigor as application code — version control, review, pinning, and automation — and your GPUs, data, and models become reproducible, cost-controlled, and trustworthy.

## Chapter Q&A

**Q1: Walk through the Terraform core workflow from empty directory to running infrastructure.**
A: `terraform init` downloads providers and configures the backend; `terraform fmt` and `terraform validate` check style and syntax; `terraform plan -out=plan.tfplan` computes and saves the diff for review; `terraform apply plan.tfplan` executes exactly that reviewed plan; and `terraform destroy` tears it down. In a team, plan runs in CI on PRs and apply runs post-merge with locking on remote state.

**Q2: Why should the state file never be committed to Git or edited by hand?**
A: State often contains secrets in plaintext (passwords, keys), so Git would leak them and lack locking to prevent concurrent corruption. Hand-editing risks desynchronizing state from reality, causing Terraform to duplicate or orphan resources. Use a remote backend for storage/locking and `terraform state`/`import` subcommands for surgical changes.

**Q3: A colleague changed a security group in the AWS console. What happens on your next apply and how do you handle it?**
A: `terraform plan` refreshes state, detects drift, and shows a diff to revert the manual change back to what your code declares. If the manual change was correct, you update your HCL to match and apply (no-op); if it was undesired, you apply to restore the declared state. Long term, restrict console write access in prod to prevent drift.

**Q4: How do you provision a cost-efficient GPU cluster for bursty training?**
A: Define a dedicated GPU node group/pool tainted for GPU workloads, with `min_size = 0` and a cluster autoscaler so it scales to zero when idle; use spot/preemptible capacity for interruptible jobs; separate pools per accelerator type; right-size instances; and checkpoint frequently to survive spot interruptions. Codify it all in a reusable module so every team gets the same guardrails.

**Q5: Explain modules, and what belongs in a module versus the root.**
A: A module is a reusable, parameterized package of resources with inputs and outputs. Child modules contain resource logic and declare required providers but do *not* configure provider credentials; the root module configures providers, supplies variable values, wires modules together, and holds the backend config. This keeps modules portable across accounts and regions.

**Q6: What is the purpose of `data` sources and give an ML example.**
A: Data sources read existing information without managing it. An ML example is `data "aws_ssm_parameter"` to fetch the recommended EKS GPU-optimized AMI ID for your Kubernetes version, ensuring node groups always launch with a driver-matched image rather than a hardcoded, drifting AMI ID.

**Q7: Compare workspaces with directory-per-environment for environment isolation.**
A: Workspaces reuse one configuration with separate state files — lightweight but easy to misfire since the active workspace is a hidden CLI setting (risk of applying prod thinking you're in dev). Directory/state-key-per-environment (optionally per account) makes separation explicit and supports differing backends and blast radii, which most teams prefer for prod-grade isolation.

**Q8: How do you achieve reproducible ML environments with Terraform?**
A: Pin `required_version`, exact provider versions, module Git `?ref=` tags, Kubernetes version, Helm chart versions, and image digests; express environment differences only as parameters, not forked code; tag infra releases in Git; forbid manual changes; and keep secrets in a secrets manager. Then checking out a tag and applying rebuilds an identical environment anywhere.

**Q9: What does Terraform give you, and not give you, in a multi-cloud setup?**
A: It gives one language, one workflow, and one review/CI process across every cloud via native providers, plus provider aliasing for multi-region/account fan-out. It does not abstract resource differences — `aws_s3_bucket` and `google_storage_bucket` have different schemas. Any cross-cloud abstraction (e.g., a common storage module interface) is something you design yourself.

**Q10: How does S3 + DynamoDB provide safe collaborative state, and what is the newer alternative?**
A: S3 stores versioned, encrypted state; DynamoDB holds a lock item keyed by `LockID` that Terraform acquires before writing and releases after, blocking concurrent applies. Newer Terraform versions support S3-native locking via conditional writes (`use_lockfile`), which can remove the DynamoDB dependency while providing the same mutual exclusion.
