# Chapter 10 — GPU / CUDA Environments

Modern machine learning is, in practice, an exercise in feeding matrices to accelerators as fast as they can consume them. The GPU is the workhorse of that process, and CUDA is the software contract that lets your Python code reach the silicon. For an MLOps engineer, GPUs are rarely something you write CUDA kernels for by hand — instead, your job is to provision, containerize, schedule, and debug GPU environments so that data scientists' models train reliably and serve at low latency. That means understanding just enough of the hardware to reason about memory and throughput, and a great deal about the notoriously brittle software stack that sits on top of it.

This chapter walks the full path from silicon to cluster. We start with GPU architecture fundamentals, move through installing and verifying the CUDA stack, then containerize GPU workloads, scale to multi-GPU and multi-node training, schedule GPUs on Kubernetes, and finally untangle the driver/toolkit/framework compatibility matrix that causes more production incidents than any other single topic in this space. Throughout, the emphasis is on what actually breaks in production and how to prevent it.

---

## 1. GPU Architecture Basics

### Plain-language explanation

A CPU has a handful of powerful cores optimized for latency — doing one complex thing quickly. A GPU has thousands of simpler cores optimized for throughput — doing the same simple thing to enormous amounts of data at once. Physically, a GPU is organized into **Streaming Multiprocessors (SMs)**. Each SM contains many **CUDA cores** (which handle general floating-point and integer math) and, on modern NVIDIA data-center and consumer cards, **Tensor Cores** (specialized units that perform fused matrix-multiply-and-accumulate operations on small tiles in a single step). Tensor Cores are what make mixed-precision deep learning fast.

The GPU has its own dedicated memory, **VRAM** (e.g., HBM3 on an H100, GDDR6 on consumer cards), which is much faster than system RAM but physically separate from it. Because of that separation, data must be copied across the PCIe bus (or NVLink) from **host** (CPU/system RAM) to **device** (GPU/VRAM) before the GPU can work on it, and results copied back. This **host-device transfer** is often the hidden bottleneck.

GPUs suit ML because training is dominated by dense linear algebra — matrix multiplications and convolutions — which is embarrassingly parallel. Thousands of cores multiplying rows and columns simultaneously is exactly the workload GPUs were built for.

### Why it matters in production

Almost every GPU cost and performance decision traces back to two numbers: how much VRAM you have and how well you keep the SMs busy. VRAM caps your model size and batch size; run out and you get the infamous `CUDA out of memory` error. Meanwhile, if your data pipeline can't feed the GPU fast enough, expensive Tensor Cores sit idle while the card waits on host-device transfers — you pay for an H100 and get A100-level utilization. Understanding the memory hierarchy lets you right-size instances, choose batch sizes, and diagnose whether you're compute-bound, memory-bound, or I/O-bound.

### Real-world example / use case

A team fine-tuning a 7B-parameter LLM finds training is slow and GPU utilization (from `nvidia-smi`) hovers at 30%. The model fits in VRAM, so it isn't a memory problem. Profiling reveals the CPU data loader is decoding and tokenizing on the fly, unable to keep the queue full. Increasing `num_workers` in the DataLoader, pinning memory (`pin_memory=True`), and pre-tokenizing the dataset pushes utilization to 90%+ and cuts wall-clock training time by more than half — without changing a single GPU.

### Code or config example

```bash
# Inspect the GPU topology, memory, and utilization
nvidia-smi

# Sample output (abridged):
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.4     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |   0  NVIDIA H100  On          | 00000000:17:00.0 Off |                    0 |
# | N/A  42C   P0    120W / 700W  |  38912MiB / 81559MiB |     91%   Default    |
# +-------------------------------+----------------------+----------------------+

# Show NVLink / PCIe topology between multiple GPUs (matters for transfer speed)
nvidia-smi topo -m
```

```python
# Demonstrate host-device transfer and the memory hierarchy in PyTorch
import torch

x_cpu = torch.randn(8192, 8192)          # lives in host RAM
x_gpu = x_cpu.to("cuda", non_blocking=True)  # copied across PCIe to VRAM
y = x_gpu @ x_gpu                         # matmul runs on SMs / Tensor Cores
result = y.cpu()                          # copied back host-side

print(torch.cuda.memory_allocated() / 1e9, "GB allocated on device")
```

### Interview questions

**Q: Why are GPUs better suited to deep learning than CPUs?**
A: Deep learning training is dominated by dense matrix operations that are highly parallelizable. A GPU has thousands of cores (plus Tensor Cores for matrix-multiply-accumulate) and very high memory bandwidth, so it can execute the same operation across huge tensors simultaneously — throughput-oriented. CPUs have few, latency-optimized cores and are far better at branchy, sequential control flow but far worse at the SIMD-style bulk math that dominates training.

**Q: What are Tensor Cores and why do they matter?**
A: Tensor Cores are specialized hardware units that perform a small matrix multiply-and-accumulate (e.g., 4x4 or larger tiles) in a single operation, typically in reduced precision like FP16, BF16, or FP8 with FP32 accumulation. They dramatically accelerate the GEMM and convolution operations at the heart of neural nets, which is why mixed-precision training (`torch.cuda.amp`) can be several times faster than pure FP32.

**Q: What is host-device transfer and why can it become a bottleneck?**
A: GPU VRAM is separate from system RAM, so data must be copied over PCIe (or NVLink) between them. If the data pipeline copies too often, transfers small tensors, or blocks the training loop waiting for data, the GPU starves. Mitigations include pinned memory, `non_blocking=True` async copies, larger batches, prefetching, and keeping intermediate tensors on-device.

### Pitfalls & best practices

- **Pitfall:** Assuming a bigger/newer GPU automatically means faster training. If you're I/O- or CPU-bound, the GPU idles regardless.
- **Best practice:** Always measure GPU utilization and memory before optimizing. Use `nvidia-smi dmon` or `nvidia-smi -l 1` for live sampling.
- **Best practice:** Prefer BF16/FP16 mixed precision on Tensor-Core GPUs; it saves VRAM and boosts throughput with minimal accuracy loss.
- **Best practice:** Minimize `.cpu()`/`.item()` calls inside training loops — each forces a synchronization and a host-device transfer.

---

## 2. CUDA / cuDNN Setup

### Plain-language explanation

The GPU software stack has three layers you must keep straight:

1. **NVIDIA driver** — the kernel-level software that lets the OS talk to the GPU. It reports a "CUDA Version" in `nvidia-smi`, which is the *maximum* CUDA runtime the driver can support.
2. **CUDA Toolkit** — the runtime libraries and compiler (`nvcc`) that applications link against. This is the CUDA version your framework was built with.
3. **cuDNN** — a library of highly optimized primitives for deep learning (convolutions, pooling, RNN/attention kernels) that sits on top of CUDA. Frameworks call into cuDNN for performance-critical layers.

The crucial simplification for most MLOps work: when you install PyTorch or TensorFlow via pip/conda, the **CUDA runtime and cuDNN come bundled inside the wheel**. You do *not* need a system-wide CUDA Toolkit install to run them — you only need a compatible **driver**. You need a full toolkit only when compiling custom CUDA extensions.

### Why it matters in production

The single most common GPU support ticket is "PyTorch can't see the GPU" or "CUDA error: no kernel image is available." Almost always the root cause is a mismatch in this stack — the driver is too old for the CUDA runtime the framework needs, or someone installed a CPU-only wheel, or a system CUDA path shadows the bundled one. Knowing exactly which layer owns which responsibility lets you diagnose in minutes instead of hours.

### Real-world example / use case

A CI pipeline that trains nightly suddenly fails after a base-image bump with `CUDA driver version is insufficient for CUDA runtime version`. The image maintainer upgraded PyTorch to a build compiled against CUDA 12.4, but the GPU nodes still run driver 525 (which supports CUDA up to 12.0). The fix is either to pin PyTorch to a CUDA 12.0-compatible build or to upgrade the node drivers. Understanding the "driver caps the runtime" rule points straight to the answer.

### Code or config example

```bash
# 1. Confirm the driver is installed and see the max CUDA it supports
nvidia-smi          # "CUDA Version: 12.4" = driver's ceiling, NOT the toolkit installed

# 2. (Only if you need to compile CUDA code) check the installed toolkit
nvcc --version      # shows the CUDA Toolkit compiler version

# 3. Install a PyTorch build for a specific CUDA runtime (bundled, no system toolkit needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```python
# 4. Verify PyTorch actually sees and can use the GPU
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime built with:", torch.version.cuda)   # e.g. 12.1
print("cuDNN version:", torch.backends.cudnn.version())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    # Smoke test: force a real kernel launch
    x = torch.rand(1000, 1000, device="cuda")
    print("Matmul OK:", (x @ x).sum().item() > 0)
```

### Interview questions

**Q: What is the difference between the CUDA version shown by `nvidia-smi` and the one shown by `nvcc --version`?**
A: `nvidia-smi` reports the maximum CUDA runtime version the installed **driver** can support — it's a ceiling, not necessarily what's installed. `nvcc --version` reports the version of the CUDA **Toolkit** (compiler) actually installed on the system. They can legitimately differ; for example, a driver supporting CUDA 12.4 can run applications built against toolkit 11.8.

**Q: Do you need to install the CUDA Toolkit to run PyTorch on a GPU?**
A: No. PyTorch pip/conda wheels bundle their own CUDA runtime and cuDNN. You only need a compatible NVIDIA driver on the host. A full system CUDA Toolkit is required only when you compile custom CUDA extensions (e.g., custom ops, some versions of flash-attention, or `nvcc`-based builds).

**Q: `torch.cuda.is_available()` returns False on a GPU box. How do you debug it?**
A: Check in order: (1) `nvidia-smi` works — if not, the driver is broken/missing; (2) you installed a CUDA-enabled wheel, not the CPU-only one (`torch.version.cuda` should be non-None); (3) inside a container, that `--gpus` / NVIDIA Container Toolkit is configured; (4) the driver supports the runtime PyTorch was built with; (5) `CUDA_VISIBLE_DEVICES` isn't set to empty.

### Pitfalls & best practices

- **Pitfall:** Installing the default `pip install torch`, which may pull a CPU-only build in some environments — then wondering why the GPU isn't used. Always use the correct `--index-url`.
- **Pitfall:** A stray system CUDA on `LD_LIBRARY_PATH` shadowing the bundled runtime, causing subtle version mismatches.
- **Best practice:** Standardize on containers so the CUDA/cuDNN stack is reproducible and decoupled from host state (only the driver stays on the host).
- **Best practice:** Bake the verification snippet above into a startup healthcheck so misconfigured nodes fail fast.

---

## 3. Containerizing GPU Workloads

### Plain-language explanation

Docker by default isolates containers from host hardware, so a plain container cannot see the GPU. The **NVIDIA Container Toolkit** bridges this gap: it installs a runtime hook that injects the host's GPU devices and driver libraries into the container at launch. The key design principle is that **the driver lives on the host, and the CUDA runtime/cuDNN live in the image**. That's why the same GPU box can run one container built on CUDA 11.8 and another on CUDA 12.4 simultaneously — each brings its own userspace CUDA, sharing the single host driver.

You typically build on NVIDIA's official `nvidia/cuda` base images (choose `base`, `runtime`, or `devel` variants) or the framework images like `pytorch/pytorch`. At runtime you pass `--gpus all` (or specific device IDs) to expose GPUs.

### Why it matters in production

Containerization is what makes GPU environments reproducible. It eliminates the "works on my machine" problem for the entire CUDA stack, lets you run heterogeneous CUDA versions on shared hardware, and is the prerequisite for orchestration with Kubernetes. Without it, you'd be fighting driver/toolkit drift on every host.

### Real-world example / use case

An inference service must serve two models: a legacy TensorFlow model requiring CUDA 11.2 and a new PyTorch model on CUDA 12.1. On bare metal this is a nightmare of conflicting libraries. With the NVIDIA Container Toolkit, each service ships as its own container with its own CUDA runtime, both scheduled onto the same GPU node, sharing only the host driver. Deployment becomes trivial and rollbacks are just image tags.

### Code or config example

```dockerfile
# Dockerfile — GPU-enabled training image
# Use the runtime variant (has CUDA + cuDNN, no compiler) to keep it small.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Install a PyTorch build matching the image's CUDA 12.1 runtime
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .
ENTRYPOINT ["python3", "train.py"]
```

```bash
# Install the NVIDIA Container Toolkit on the host (Ubuntu)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Sanity check: run nvidia-smi inside a container
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Expose only specific GPUs, and set a memory-friendly run
docker run --rm --gpus '"device=0,1"' my-training-image:latest
```

### Interview questions

**Q: Why can't a normal Docker container use the GPU, and what fixes it?**
A: Docker isolates the container from host devices, so `/dev/nvidia*` and the driver libraries aren't visible. The NVIDIA Container Toolkit registers a runtime hook that injects the GPU device nodes and the host driver's userspace libraries into the container at startup. You then request GPUs with `docker run --gpus`.

**Q: If the driver is on the host, what goes in the image?**
A: The CUDA runtime, cuDNN, and any CUDA libraries the app needs — everything in userspace. The kernel-mode driver stays on the host. This split is what allows multiple containers with different CUDA versions to run on one host that has a single driver.

**Q: What's the difference between the `base`, `runtime`, and `devel` CUDA images?**
A: `base` has the minimal CUDA runtime libraries; `runtime` adds math libraries and cuDNN (good for running trained models/frameworks); `devel` additionally includes the compiler (`nvcc`) and headers, needed when you compile CUDA extensions. Use the smallest one that meets your needs to reduce image size and attack surface.

### Pitfalls & best practices

- **Pitfall:** Forgetting `--gpus` — the container runs CPU-only silently.
- **Pitfall:** Mismatching the image's CUDA version with the PyTorch wheel's CUDA build inside it.
- **Best practice:** Use `runtime` images for serving/training and reserve `devel` for build stages; use multi-stage builds to keep final images lean.
- **Best practice:** Pin exact image tags (e.g., `12.1.1-cudnn8-runtime-ubuntu22.04`), never `latest`, for reproducibility.
- **Best practice:** Set `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES` explicitly when you need fine control.

---

## 4. Multi-GPU / Multi-Node Training

### Plain-language explanation

When a model or dataset outgrows one GPU, you parallelize. The most common approach is **data parallelism**: replicate the model on each GPU, give each a different slice of the batch, then average (all-reduce) the gradients so every replica stays in sync.

PyTorch offers two implementations:

- **`DataParallel` (DP)** — single-process, multi-threaded, splits the batch across GPUs on one machine. Simple but slow: it's bottlenecked by the Python GIL and a single-process gather/scatter. Effectively deprecated for serious work.
- **`DistributedDataParallel` (DDP)** — one process per GPU, each with its own Python interpreter, communicating gradients via **NCCL** (NVIDIA's collective communication library, optimized for GPU-to-GPU all-reduce over NVLink/InfiniBand). DDP scales across multiple nodes and is the standard.

For models too large to fit on one GPU even as a single replica, you move to **sharding**: **FSDP** (Fully Sharded Data Parallel, built into PyTorch) and **DeepSpeed ZeRO** shard the model parameters, gradients, and optimizer states across GPUs so each holds only a fraction, enabling training of models far larger than one device's VRAM.

`torchrun` is the launcher that spawns and coordinates the per-GPU processes.

### Why it matters in production

Training time and model scale are directly gated by how well you parallelize. Choosing DP over DDP can silently halve your throughput. Misconfigured NCCL (wrong network interface, no InfiniBand) can make multi-node training slower than single-node. And for large models, FSDP/DeepSpeed is not an optimization but a hard requirement — a 70B model simply won't fit otherwise. These choices determine both cost and feasibility.

### Real-world example / use case

A team scales training from 1 to 8 A100s using `DataParallel` and sees only ~3x speedup instead of ~8x. Switching to DDP with `torchrun` gives near-linear scaling because each GPU runs its own process and NCCL does efficient ring all-reduce over NVLink. Later, when they move to a 4-node, 32-GPU job, they discover NCCL was falling back to Ethernet; setting `NCCL_IB_HCA` and the correct `NCCL_SOCKET_IFNAME` to use InfiniBand restores scaling.

### Code or config example

```python
# train_ddp.py — minimal DistributedDataParallel training
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend="nccl")          # NCCL for GPU collectives
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    for batch in dataloader:                          # use DistributedSampler
        loss = model(batch.to(local_rank)).loss
        loss.backward()                              # gradients all-reduced via NCCL
        optimizer.step()
        optimizer.zero_grad()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

```bash
# Single node, 8 GPUs
torchrun --standalone --nproc_per_node=8 train_ddp.py

# Multi-node: 2 nodes x 8 GPUs = 16 processes
# On node 0 (the rendezvous master):
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
         --rdzv_backend=c10d --rdzv_endpoint=10.0.0.1:29500 train_ddp.py
# On node 1:
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
         --rdzv_backend=c10d --rdzv_endpoint=10.0.0.1:29500 train_ddp.py

# Force NCCL onto the right high-speed fabric for multi-node
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO         # invaluable for diagnosing hangs
```

### Interview questions

**Q: What's the difference between DataParallel and DistributedDataParallel, and which should you use?**
A: `DataParallel` is single-process and multi-threaded, splitting a batch across GPUs on one node; it's limited by the GIL and a single-process gradient gather, so it scales poorly. `DistributedDataParallel` runs one process per GPU, overlaps gradient communication with backprop, uses NCCL for efficient all-reduce, and scales across nodes. Always prefer DDP.

**Q: What is NCCL and what role does it play?**
A: NCCL (NVIDIA Collective Communications Library) implements optimized collective operations — all-reduce, broadcast, all-gather — across GPUs using topology-aware algorithms over NVLink, PCIe, or InfiniBand. In DDP it performs the gradient all-reduce that keeps replicas synchronized. Correct NCCL network configuration is critical for multi-node performance.

**Q: When do you need FSDP or DeepSpeed instead of plain DDP?**
A: When the model itself (parameters + gradients + optimizer states) is too large to fit on a single GPU. DDP replicates the full model on every GPU, so it can't help there. FSDP and DeepSpeed ZeRO shard those states across GPUs so each holds only a slice, enabling training of models that far exceed a single device's VRAM, at the cost of extra communication.

### Pitfalls & best practices

- **Pitfall:** Using `DataParallel` and expecting linear scaling — you won't get it.
- **Pitfall:** Forgetting `DistributedSampler`, so every rank trains on the same data.
- **Pitfall:** Multi-node hangs due to NCCL picking the wrong network interface; set `NCCL_SOCKET_IFNAME` / `NCCL_DEBUG=INFO`.
- **Best practice:** Launch with `torchrun`, not the deprecated `torch.distributed.launch`.
- **Best practice:** Match effective batch size and learning rate (linear scaling rule) when increasing GPU count.
- **Best practice:** Use gradient checkpointing plus FSDP/DeepSpeed for memory-bound large models.

---

## 5. GPU Scheduling in Kubernetes

### Plain-language explanation

Kubernetes doesn't understand GPUs natively — it schedules CPU and memory. The **NVIDIA device plugin** (a DaemonSet running on each GPU node) advertises GPUs to the kubelet as a schedulable **extended resource** named `nvidia.com/gpu`. Pods then request whole GPUs the same way they request CPU:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

A key constraint: GPUs are **not divisible by default** — you request whole integers, and a GPU is exclusively assigned to one container. Two ways to share a physical GPU across workloads:

- **MIG (Multi-Instance GPU)** — hardware-level partitioning available on A100/H100 and newer. One physical GPU is split into up to 7 isolated instances, each with dedicated SMs and memory. Strong isolation; ideal for multi-tenant inference.
- **Time-slicing** — software oversubscription where multiple pods share a GPU by taking turns. No memory isolation (they can OOM each other), but great for bursty, low-utilization dev/inference workloads.

You steer GPU pods onto the right nodes with **node selectors, node affinity, and taints/tolerations** (commonly you taint GPU nodes so only GPU workloads land there).

### Why it matters in production

GPUs are the most expensive resource in an ML cluster, so utilization is money. Naively giving every pod a whole GPU wastes capacity when many workloads only need a fraction. MIG and time-slicing let you pack more work onto fewer cards. Taints/affinity prevent CPU-only jobs from squatting on GPU nodes. Getting scheduling right can cut GPU spend dramatically while keeping isolation guarantees where they matter (e.g., MIG for production inference tenants).

### Real-world example / use case

A platform team runs dozens of low-traffic inference services, each idling on a dedicated A100 at 10% utilization — enormously wasteful. They enable MIG, splitting each A100 into seven `1g.10gb` instances, and reconfigure services to request `nvidia.com/mig-1g.10gb: 1`. Effective GPU capacity increases ~7x for those services with hard memory isolation between tenants, while latency-sensitive production models keep whole GPUs.

### Code or config example

```yaml
# Pod requesting one whole GPU, pinned to GPU nodes via nodeSelector + toleration
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training-job
spec:
  restartPolicy: Never
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB   # label set by GPU Feature Discovery
  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"
  containers:
    - name: trainer
      image: my-registry/train:cu121
      command: ["torchrun", "--nproc_per_node=1", "train_ddp.py"]
      resources:
        limits:
          nvidia.com/gpu: 1        # whole-GPU request (also acts as the request)
```

```yaml
# Requesting a MIG slice instead of a whole GPU
resources:
  limits:
    nvidia.com/mig-1g.10gb: 1
```

```bash
# Deploy the NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml

# Enable MIG on a node (single strategy = all instances same profile)
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -cgi 1g.10gb -C   # create 1g.10gb compute instances
```

```yaml
# Time-slicing config: advertise each physical GPU as 4 shareable replicas
# (ConfigMap consumed by the device plugin)
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
data:
  any: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 4
```

### Interview questions

**Q: How does Kubernetes schedule GPUs, given it doesn't understand them natively?**
A: Via the NVIDIA device plugin, a DaemonSet on each GPU node that registers GPUs with the kubelet as the extended resource `nvidia.com/gpu`. Pods request whole GPUs under `resources.limits`, and the scheduler places them on nodes with available GPU capacity. GPUs are allocated as indivisible whole units unless MIG or time-slicing is configured.

**Q: Compare MIG and time-slicing.**
A: MIG partitions a physical GPU into hardware-isolated instances, each with dedicated compute and memory — strong isolation and predictable QoS, ideal for multi-tenant production. Time-slicing is software oversubscription: pods share a GPU by taking turns with no memory isolation, so one workload can OOM another. Time-slicing maximizes utilization for bursty, non-critical workloads; MIG is for isolation guarantees. MIG requires A100/H100-class hardware.

**Q: How do you keep non-GPU workloads off expensive GPU nodes?**
A: Taint the GPU nodes (e.g., `nvidia.com/gpu=present:NoSchedule`) so only pods with a matching toleration can schedule there, and use nodeSelector/affinity on GPU labels to steer GPU workloads onto them. This prevents CPU-only pods from occupying scarce GPU capacity.

### Pitfalls & best practices

- **Pitfall:** Setting only `requests` and not `limits` for `nvidia.com/gpu` — GPU is a limit-only extended resource; request must equal limit.
- **Pitfall:** Enabling time-slicing for production inference and getting cross-tenant OOMs due to no memory isolation.
- **Best practice:** Use the NVIDIA GPU Operator to manage drivers, device plugin, MIG, and feature discovery as one bundle.
- **Best practice:** Label and taint GPU nodes; use affinity to target specific GPU models when a job needs particular capabilities (e.g., Tensor Cores, VRAM size).
- **Best practice:** Choose MIG for multi-tenant isolation, time-slicing for internal/bursty efficiency.

---

## 6. Driver / Version Compatibility Issues

### Plain-language explanation

This is where most GPU incidents live. The compatibility relationship is a chain:

**GPU driver → CUDA runtime → deep learning framework (PyTorch/TF) → cuDNN.**

The governing rules:

1. **The driver sets the ceiling.** A given driver supports up to a maximum CUDA runtime version (shown in `nvidia-smi`). A container/app built against a *newer* CUDA runtime than the driver supports will fail with `CUDA driver version is insufficient for CUDA runtime version`.
2. **CUDA has backward "minor version compatibility."** Within a major version (e.g., CUDA 12.x), a reasonably recent driver can run applications built against any 12.x runtime. Crossing major versions (11.x → 12.x) generally requires a newer driver.
3. **The framework is compiled against a specific CUDA + cuDNN.** PyTorch `cu121` needs a driver supporting CUDA 12.1+ and ships its matching cuDNN.
4. **GPU architecture (compute capability)** must be supported by the CUDA version — a too-old CUDA won't have kernels for a brand-new GPU (`no kernel image is available for execution`), and a too-new framework may drop support for very old GPUs.

The practical mental model: **upgrade the driver freely (it's backward compatible), pin the framework/CUDA to what your driver supports, and keep them consistent inside containers.**

### Why it matters in production

Version mismatches are silent until they aren't — a base-image bump, a driver rollback, or a new GPU SKU can take down training or serving cluster-wide. Because the failure modes have cryptic error messages, engineers who don't understand the matrix can burn days. Treating the driver/CUDA/framework versions as a governed, tested combination is essential for reliability.

### Real-world example / use case

A cluster is upgraded with new H100 nodes, but the existing training image uses an older PyTorch/CUDA build that predates Hopper (compute capability 9.0) support. Jobs fail with `no kernel image is available for execution on the device`. The fix is to rebuild the image on a CUDA 12.x / newer PyTorch that includes SM 9.0 kernels — the driver was fine; the *CUDA runtime* was too old to know about the new architecture.

### Code or config example

```bash
# Diagnose the full stack quickly
nvidia-smi                              # driver version + max CUDA supported
cat /proc/driver/nvidia/version         # exact driver build
python -c "import torch; print(torch.__version__, torch.version.cuda, \
           torch.backends.cudnn.version(), torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_capability(0))"  # e.g. (9, 0) for H100
```

```text
# Reference compatibility mental model (illustrative)
# Driver >= 525  -> supports CUDA up to 12.0
# Driver >= 535  -> supports CUDA up to 12.2
# Driver >= 550  -> supports CUDA up to 12.4
# Rule: framework's built CUDA must be <= driver's max supported CUDA
# Rule: CUDA runtime must be new enough to include your GPU's compute capability
```

```bash
# Common failure modes and what they mean:
# "CUDA driver version is insufficient for CUDA runtime version"
#   -> driver too old for the CUDA the framework was built with. Upgrade driver
#      or downgrade the framework/CUDA build.
# "no kernel image is available for execution on the device"
#   -> CUDA/framework too old for this GPU architecture (or wrong TORCH_CUDA_ARCH_LIST).
#      Rebuild against a newer CUDA that supports the compute capability.
# "torch.cuda.is_available() == False"
#   -> CPU-only wheel, missing NVIDIA Container Toolkit, or CUDA_VISIBLE_DEVICES unset/empty.
```

### Interview questions

**Q: Explain the compatibility relationship between the GPU driver, CUDA runtime, and PyTorch.**
A: The driver defines the maximum CUDA runtime it can support (visible in `nvidia-smi`). PyTorch is compiled against a specific CUDA runtime and cuDNN (bundled in its wheel). For it to work, the driver's max supported CUDA must be greater than or equal to the CUDA version PyTorch was built with. Within a CUDA major version, minor-version compatibility means a recent driver runs any 12.x app; crossing major versions typically needs a driver upgrade. Drivers are backward compatible, so upgrading the driver rarely breaks older apps.

**Q: You see `no kernel image is available for execution on the device`. What's wrong?**
A: The installed CUDA/framework build doesn't contain compiled kernels for the GPU's compute capability — usually a too-old CUDA on a newer GPU (e.g., old build on H100/Hopper), or a custom build with a `TORCH_CUDA_ARCH_LIST` that omits the target architecture. Fix by rebuilding/installing a framework version compiled against a CUDA that supports that GPU's SM version.

**Q: Is it safe to upgrade the GPU driver on a node running older CUDA containers?**
A: Generally yes. NVIDIA drivers are backward compatible with older CUDA runtimes, so upgrading the driver raises the CUDA ceiling without breaking existing applications built on older CUDA. This is why the recommended pattern is a modern host driver plus per-container CUDA runtimes. The reverse — using a new CUDA runtime on an old driver — is what fails.

### Pitfalls & best practices

- **Pitfall:** Downgrading a driver to "match" CUDA — unnecessary and risky; the driver only needs to be new enough.
- **Pitfall:** Mixing a host CUDA on `LD_LIBRARY_PATH` with a container's bundled runtime.
- **Pitfall:** Assuming a new base image "just works" on old GPUs — check compute-capability support both ways.
- **Best practice:** Treat driver + CUDA + framework as a tested, version-pinned combination; validate in CI before cluster rollout.
- **Best practice:** Keep host drivers modern and uniform across the fleet; vary CUDA only inside containers.
- **Best practice:** Maintain a small compatibility table for your supported GPU SKUs and framework versions.

---

## Summary

GPUs accelerate machine learning because training is dominated by massively parallel dense linear algebra, executed across thousands of CUDA cores and Tensor Cores backed by high-bandwidth VRAM — with host-device transfer as the frequent hidden bottleneck. The software stack that exposes this hardware has three layers you must keep straight: the **host driver** (which sets the maximum CUDA version), the **CUDA Toolkit/runtime** (bundled inside framework wheels for most workflows), and **cuDNN** (deep learning primitives). For reproducibility and heterogeneous CUDA versions on shared hardware, you containerize with the **NVIDIA Container Toolkit**, keeping the driver on the host and the CUDA runtime in the image.

Scaling beyond one GPU means **DistributedDataParallel** with **NCCL** over `torchrun` for data parallelism, and **FSDP/DeepSpeed** sharding when a model exceeds a single device's memory. On **Kubernetes**, the NVIDIA device plugin advertises `nvidia.com/gpu`, and **MIG** (hardware isolation) or **time-slicing** (software sharing) plus taints/affinity let you maximize utilization of your most expensive resource. Finally, the **driver → CUDA → framework → cuDNN** compatibility chain — where the driver sets the ceiling, drivers are backward compatible, and CUDA must be new enough for your GPU's compute capability — is the root cause of most GPU incidents. Master that matrix and you eliminate the majority of GPU production pain.

## Chapter Q&A

**Q1: Why do GPUs outperform CPUs for deep learning, and what limits that advantage?**
A: GPUs have thousands of throughput-oriented cores plus Tensor Cores and high memory bandwidth, perfectly matched to the parallel matrix math in training. The advantage is limited by VRAM capacity (caps model/batch size) and by feeding the GPU fast enough — if the data pipeline or host-device transfer can't keep the SMs busy, the GPU idles and the theoretical advantage is lost.

**Q2: What does the "CUDA Version" in `nvidia-smi` actually mean?**
A: It's the maximum CUDA runtime version the installed **driver** can support — a ceiling. It is not the installed CUDA Toolkit version (that's `nvcc --version`) nor necessarily the CUDA your framework uses. Applications can use any CUDA runtime up to that ceiling.

**Q3: Do you need a system CUDA Toolkit to run PyTorch on GPUs?**
A: No. PyTorch wheels bundle their own CUDA runtime and cuDNN; you only need a compatible host driver. A full toolkit (`nvcc`) is needed only to compile custom CUDA extensions.

**Q4: How does the NVIDIA Container Toolkit let containers use GPUs?**
A: It installs a runtime hook that injects the host's GPU device nodes and driver userspace libraries into the container at launch. The driver stays on the host; the CUDA runtime lives in the image — enabling multiple containers with different CUDA versions on one host with a single driver. You request GPUs with `docker run --gpus`.

**Q5: DataParallel vs DistributedDataParallel — which and why?**
A: DDP. DataParallel is single-process/multi-threaded, GIL-bound, and single-node with poor scaling. DDP uses one process per GPU, overlaps NCCL gradient all-reduce with backprop, and scales across nodes, giving near-linear speedups.

**Q6: When must you move from DDP to FSDP or DeepSpeed?**
A: When the model plus its gradients and optimizer states no longer fit on a single GPU. DDP replicates the full model per GPU, so sharding (FSDP / DeepSpeed ZeRO) is required to distribute those states and train models larger than one device's VRAM.

**Q7: How are GPUs requested and shared in Kubernetes?**
A: The NVIDIA device plugin exposes GPUs as the extended resource `nvidia.com/gpu`, requested via `resources.limits`. Whole GPUs are the default unit; MIG partitions a GPU into hardware-isolated instances (multi-tenant isolation), and time-slicing oversubscribes a GPU across pods (higher utilization, no memory isolation). Taints/affinity keep GPU workloads on GPU nodes.

**Q8: A container reports `CUDA driver version is insufficient for CUDA runtime version`. Diagnose and fix.**
A: The framework in the image was built against a newer CUDA than the host driver supports. Check `nvidia-smi` for the driver's max CUDA, then either upgrade the host driver (backward compatible, safe) or use a framework build compiled against an older, supported CUDA.

**Q9: `torch.cuda.is_available()` is False inside a Kubernetes pod. Walk through debugging.**
A: Verify the node actually has GPUs and the device plugin advertises them; confirm the pod requested `nvidia.com/gpu`; ensure the NVIDIA Container Toolkit / runtime is configured; check the image has a CUDA-enabled PyTorch (`torch.version.cuda` not None); confirm `CUDA_VISIBLE_DEVICES`/`NVIDIA_VISIBLE_DEVICES` aren't empty; and validate the driver supports the framework's CUDA.

**Q10: Is upgrading the host GPU driver safe for existing older-CUDA workloads?**
A: Yes, generally. NVIDIA drivers are backward compatible with older CUDA runtimes, so upgrading raises the supported-CUDA ceiling without breaking older apps. The recommended pattern is a modern, uniform host driver with per-container CUDA runtimes; the failing direction is a new CUDA runtime on an old driver.
