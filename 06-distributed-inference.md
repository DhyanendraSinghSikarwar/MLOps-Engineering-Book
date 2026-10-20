# Chapter 6 — Large-Scale Distributed Inference

Training a model is only half the story. The moment a model has to serve real user traffic — a recommendation feed, a chat completion, a fraud score — it becomes a *systems* problem, not a machine-learning problem. Latency budgets are measured in milliseconds, traffic spikes are unpredictable, GPUs are expensive and scarce, and some modern models simply do not fit on a single accelerator. Distributed inference is the discipline of serving models correctly, quickly, and affordably at scale.

This chapter walks from the fundamentals — putting a single model behind an HTTP endpoint — through batching, sharding across GPUs, purpose-built serving frameworks (Ray Serve and NVIDIA Triton), low-level GPU optimization, autoscaling, and finally the cost-latency tradeoffs that tie every previous decision together. Each section is structured so you can both build the system and articulate the reasoning in an interview.

---

## 1. Single-Node Serving Basics

### Plain-language explanation

Single-node serving means wrapping your model in a process that listens for requests, runs a forward pass, and returns predictions — all on one machine. Clients talk to it over **REST** (JSON over HTTP, human-readable and universal) or **gRPC** (binary Protocol Buffers over HTTP/2, faster and strongly typed). Two metrics dominate everything:

- **Latency**: time to serve a single request (often reported as p50/p95/p99 percentiles).
- **Throughput**: requests (or tokens) served per second across all clients.

These two are in tension. You can lower latency by serving each request immediately and alone, or raise throughput by grouping work — but grouping adds waiting time.

### Why it matters in production

Nearly every serving system starts as a single node, and most latency and reliability bugs are rooted here: blocking the event loop, loading the model per-request, running CPU preprocessing inside the request path, or forgetting that Python's GIL serializes CPU-bound work. Getting the single-node contract right (health checks, warm model, async I/O, timeouts) is the foundation everything else builds on. Tail latency (p99) matters more than average because a single slow dependency in a fan-out request can dominate the user-visible response time.

### Real-world example / use case

A fintech serves a gradient-boosted fraud model behind FastAPI. Each transaction needs a score in under 50 ms p99 so it can block or allow a payment inline. Because the model is small and CPU-bound, they run multiple worker processes (one per core) behind a load balancer rather than using a GPU at all — the right architecture is dictated by the model, not by fashion.

### Code or config example

```python
# app.py — FastAPI single-node serving with a warm model and async endpoint
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import torch

MODEL = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ONCE at startup, not per-request. Move to GPU, set eval mode.
    model = torch.jit.load("model.ts").eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    MODEL["net"] = model
    yield
    MODEL.clear()

app = FastAPI(lifespan=lifespan)

class Request(BaseModel):
    features: list[float]

class Response(BaseModel):
    score: float
    latency_ms: float

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=Response)
def predict(req: Request):
    start = time.perf_counter()
    x = torch.tensor([req.features], dtype=torch.float32)
    if torch.cuda.is_available():
        x = x.to("cuda")
    with torch.inference_mode():          # disables autograd, faster than no_grad
        score = MODEL["net"](x).item()
    return Response(score=score, latency_ms=(time.perf_counter() - start) * 1000)
```

Run with several workers to use multiple cores: `uvicorn app:app --workers 4 --host 0.0.0.0 --port 8080`.

### Interview questions

**Q: What's the difference between latency and throughput, and why can't you always maximize both?**
A: Latency is per-request response time; throughput is aggregate requests per second. They trade off because the main lever for throughput — batching multiple requests into one GPU call — forces early requests to wait for the batch to fill, adding latency. You optimize for a *latency SLO first*, then maximize throughput within that budget.

**Q: When would you choose gRPC over REST for model serving?**
A: gRPC when you control both client and server, need low overhead and high throughput, want streaming (e.g., token-by-token LLM output), and benefit from typed schemas via protobuf. REST/JSON when you need broad interoperability, browser access, or human-debuggable payloads. gRPC's binary encoding and HTTP/2 multiplexing meaningfully cut serialization cost for large tensors.

**Q: Why is p99 latency often more important than average latency?**
A: In fan-out architectures a single request may touch dozens of services; the slowest one determines user-perceived latency, so tail latency compounds. Averages hide the users having a bad experience. SLAs are almost always written against p95/p99.

### Pitfalls & best practices

- **Pitfall**: Loading the model inside the request handler — do it once at startup.
- **Pitfall**: Running blocking CPU code in an async endpoint, stalling the event loop; offload to a thread pool or use `def` (which FastAPI runs in a threadpool) for CPU-bound work.
- **Best practice**: Always expose `/healthz` (liveness) and a separate readiness check that confirms the model is loaded.
- **Best practice**: Use `torch.inference_mode()`, set `eval()`, and pin the model to the device at startup.
- **Best practice**: Set explicit request timeouts and return 503 under overload rather than queueing unbounded.

---

## 2. Batching Strategies

### Plain-language explanation

GPUs are throughput machines: running one request and running sixteen requests often takes almost the same wall-clock time because the GPU is massively parallel. **Batching** groups multiple independent requests into a single tensor so the GPU is fully utilized. **Dynamic (or adaptive) batching** does this automatically at serve time: the server waits a tiny window (a few milliseconds) to collect requests, then dispatches whatever it has. Two knobs govern it: **max batch size** (cap on requests per batch) and **batch timeout / max queue delay** (how long to wait before firing a partial batch).

### Why it matters in production

Without batching, a GPU serving small requests may sit at 5-10% utilization — you're paying for hardware you're not using. Dynamic batching can lift throughput 5-20x. The tradeoff is that the batch-wait window adds latency to every request, so the timeout must be tuned against your SLO. For LLMs, naive batching wastes work because sequences finish at different times, which is why **continuous (in-flight) batching** — swapping finished sequences out and new ones in every decoding step — has become standard in vLLM and TensorRT-LLM.

### Real-world example / use case

An image-embedding service receives bursty traffic from a crawler. With a 5 ms max queue delay and max batch size 32, average GPU utilization rose from 12% to 78%, throughput went up ~6x, and p99 latency rose only ~4 ms — an excellent trade because the SLO was 100 ms. For an LLM chat service, continuous batching let them serve 3x more concurrent users on the same GPUs versus static batching.

### Code or config example

```protobuf
# config.pbtxt — Triton dynamic batching for a single model
max_batch_size: 32
dynamic_batching {
  # Try to form batches of these sizes if possible
  preferred_batch_size: [ 8, 16, 32 ]
  # Wait at most 3ms to accumulate a larger batch
  max_queue_delay_microseconds: 3000
}
```

```python
# Manual micro-batcher illustrating the timeout/size tradeoff
import asyncio, torch

class DynamicBatcher:
    def __init__(self, model, max_batch=32, max_delay_ms=3):
        self.model, self.max_batch, self.max_delay = model, max_batch, max_delay_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()
        asyncio.create_task(self._loop())

    async def submit(self, x):
        fut = asyncio.get_event_loop().create_future()
        await self.queue.put((x, fut))
        return await fut

    async def _loop(self):
        while True:
            x0, fut0 = await self.queue.get()
            batch, futs = [x0], [fut0]
            try:
                # Collect more until we hit size cap or the delay window expires
                deadline = asyncio.get_event_loop().time() + self.max_delay
                while len(batch) < self.max_batch:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break
                    x, f = await asyncio.wait_for(self.queue.get(), timeout)
                    batch.append(x); futs.append(f)
            except asyncio.TimeoutError:
                pass
            out = self.model(torch.stack(batch))     # single GPU call for the whole batch
            for f, o in zip(futs, out):
                f.set_result(o)
```

### Interview questions

**Q: What are the two main knobs in dynamic batching and how do they trade off?**
A: Max batch size and max queue delay (batch timeout). Larger batch size raises throughput and GPU efficiency but increases the chance of longer waits; a longer timeout forms fuller batches but adds latency to every request. You tune the timeout to consume the slack between your achievable latency and your SLO.

**Q: Why is continuous batching preferred over static batching for LLMs?**
A: In autoregressive generation, sequences in a static batch finish at different lengths, so the batch is padded to the longest sequence and GPU cycles are wasted on finished slots. Continuous (in-flight) batching evicts completed sequences and admits new ones at each decode step, keeping the batch full and boosting throughput 2-4x.

**Q: How would you set the batch timeout for a service with a 100 ms SLO?**
A: Measure the model's compute time at various batch sizes and the p99 of downstream steps, then allocate the remaining slack to the queue delay — e.g., if compute is 40 ms and other overhead 20 ms, a 10-20 ms queue delay is safe. Validate empirically under production traffic shape, not synthetic uniform load.

### Pitfalls & best practices

- **Pitfall**: Setting max batch size so high the batch exceeds GPU memory and OOMs under load.
- **Pitfall**: A large timeout under low traffic makes every request wait the full window needlessly — use preferred batch sizes and short timeouts.
- **Best practice**: Batch requests of similar shape/length together; padding wildly different sequence lengths wastes compute.
- **Best practice**: For LLMs use a framework with continuous batching (vLLM, TensorRT-LLM, TGI) instead of rolling your own.
- **Best practice**: Cap the queue depth and shed load rather than letting the batch queue grow unbounded during spikes.

---

## 3. Model Sharding

### Plain-language explanation

Some models are too big to fit in a single GPU's memory (a 70B-parameter model in FP16 needs ~140 GB, far past a single 80 GB GPU). **Sharding** splits the model across multiple GPUs. Two main strategies:

- **Tensor parallelism (TP)**: split individual layers — e.g., a matrix multiply is partitioned column-wise across GPUs, each computes a slice, then results are combined with an all-reduce. Requires very fast interconnect (NVLink) because it communicates every layer.
- **Pipeline parallelism (PP)**: split the model by *layers* — GPU 0 holds layers 1-10, GPU 1 holds 11-20, and activations flow between stages like an assembly line. Communication is lighter (only at stage boundaries) but introduces "pipeline bubbles" (idle time).

They are often combined (TP within a node over NVLink, PP across nodes) along with data parallelism for replicas.

### Why it matters in production

Sharding is what makes large-model serving possible at all. But it changes the cost and latency profile: TP adds per-layer communication overhead, so it's only efficient with high-bandwidth links; going cross-node with TP over slow Ethernet can be catastrophic. Choosing the right parallelism layout is a first-order performance decision — the wrong one can halve throughput or blow the latency budget.

### Real-world example / use case

Serving Llama-3 70B: a common layout is tensor-parallel degree 4 across 4 GPUs on one node connected by NVLink. This keeps the all-reduce communication on fast intra-node links. If they needed to serve a 405B model spanning two nodes, they'd use TP=8 within each node and PP=2 across nodes, so the slow inter-node hop only carries pipeline activations, not per-layer all-reduces.

### Code or config example

```python
# vLLM with tensor parallelism — shards the model across 4 GPUs on one node
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    tensor_parallel_size=4,        # split each layer across 4 GPUs (needs NVLink)
    # pipeline_parallel_size=2,    # add pipeline stages across nodes if needed
    dtype="float16",
    gpu_memory_utilization=0.90,   # reserve 10% headroom for activations/KV-cache
    max_model_len=8192,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain tensor parallelism in one sentence."], params)
print(outputs[0].outputs[0].text)
```

```bash
# Same idea via the OpenAI-compatible server, TP across 4 GPUs
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90
```

### Interview questions

**Q: Contrast tensor parallelism and pipeline parallelism.**
A: Tensor parallelism splits individual operations/layers across GPUs and synchronizes with all-reduce every layer, giving low latency but heavy communication — best within a node over NVLink. Pipeline parallelism splits the model into sequential stages across GPUs, communicating only at stage boundaries (light traffic) but suffering pipeline bubbles that hurt utilization unless you use micro-batching. TP scales latency-sensitive workloads inside a node; PP scales across nodes.

**Q: Why can tensor parallelism perform badly across nodes?**
A: TP requires an all-reduce after essentially every layer. Across nodes the interconnect (Ethernet/InfiniBand) has far lower bandwidth and higher latency than intra-node NVLink, so communication dominates and throughput collapses. Cross-node scaling should prefer pipeline or data parallelism, keeping TP within a node.

**Q: How do you decide the parallelism layout for a given model and cluster?**
A: Start from memory: parameters + KV cache + activations must fit. Use the largest TP degree that stays within a single NVLink domain, add pipeline parallelism only when you must cross node boundaries, and replicate with data parallelism for throughput. Then benchmark — the theoretical layout must be validated against real latency/throughput.

### Pitfalls & best practices

- **Pitfall**: Using TP across nodes over slow interconnect — profile interconnect bandwidth first.
- **Pitfall**: Forgetting the KV cache and activation memory when sizing; parameters alone don't determine fit.
- **Best practice**: Keep tensor-parallel groups within an NVLink domain; use pipeline parallelism to cross nodes.
- **Best practice**: Set `gpu_memory_utilization` below 1.0 to leave headroom for dynamic KV-cache growth.
- **Best practice**: Let a mature framework (vLLM, TensorRT-LLM, DeepSpeed-Inference) manage sharding rather than hand-rolling collectives.

---

## 4. Distributed Inference with Ray Serve and NVIDIA Triton

### Plain-language explanation

Once you outgrow a single process, you need a framework that handles replication, routing, batching, and multi-model composition.

- **Ray Serve** is a Python-native serving library on top of Ray. You write deployments as Python classes, and Ray handles scaling replicas across a cluster, autoscaling, and — crucially — composing multiple models into pipelines (a "deployment graph"). It's flexible and great when serving logic is complex or heterogeneous.
- **NVIDIA Triton Inference Server** is a high-performance, framework-agnostic serving engine (C++ core). It runs models from PyTorch, TensorFlow, ONNX, TensorRT, and Python backends, with built-in dynamic batching, concurrent model execution, model ensembles, and first-class GPU optimization. It's the choice when you want maximum GPU efficiency and standardized deployment.

They are complementary: teams often use Triton as the GPU-optimized model runtime and Ray Serve as the orchestration/business-logic layer around it.

### Why it matters in production

Rolling your own replication, routing, and batching is a large, bug-prone effort. These frameworks give you production features (metrics, health checks, model versioning, canary rollout, multi-model concurrency) out of the box. Choosing correctly avoids re-platforming later: Triton for raw single/multi-model GPU throughput and standardization; Ray Serve for Python-heavy pipelines, custom logic, and elastic scaling within an existing Ray ecosystem.

### Real-world example / use case

A vision team serves a detection model plus a downstream classifier plus non-trivial Python post-processing. They use **Ray Serve** to express the pipeline as composed deployments, autoscaling each stage independently (the detector needs 4 GPUs, the classifier 1). A separate NLP team serving a single high-QPS BERT model chooses **Triton** with dynamic batching and a TensorRT engine for maximum tokens/sec/dollar, exposing it over gRPC.

### Code or config example

```python
# Ray Serve deployment: autoscaling replicas + a composed pipeline
import ray
from ray import serve
from starlette.requests import Request

@serve.deployment(
    num_replicas="auto",
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_ongoing_requests": 8,   # scale when avg in-flight requests exceed this
    },
)
class Classifier:
    def __init__(self):
        import torch
        self.model = torch.jit.load("classifier.ts").eval().cuda()

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.005)  # dynamic batching
    async def __call__(self, requests: list) -> list:
        import torch
        x = torch.stack([r for r in requests]).cuda()
        with torch.inference_mode():
            return list(self.model(x).cpu())

@serve.deployment
class Ingress:
    def __init__(self, classifier):
        self.classifier = classifier

    async def __call__(self, http_request: Request):
        payload = await http_request.json()
        import torch
        x = torch.tensor(payload["features"])
        return {"score": float(await self.classifier.remote(x))}

# Bind the graph: Ingress depends on Classifier
app = Ingress.bind(Classifier.bind())
# serve run module:app
```

```protobuf
# Triton config.pbtxt: ONNX model, GPU, dynamic batching, 2 concurrent instances
name: "bert_classifier"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  { name: "input_ids",      data_type: TYPE_INT64, dims: [ 128 ] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [ 128 ] }
]
output [
  { name: "logits", data_type: TYPE_FP32, dims: [ 2 ] }
]
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 2000
}
instance_group [
  { count: 2, kind: KIND_GPU }   # two model instances share the GPU for concurrency
]
```

### Interview questions

**Q: When would you pick Ray Serve over Triton, and vice versa?**
A: Ray Serve for Python-native, multi-model pipelines with custom business logic, dynamic composition, and elastic autoscaling inside a Ray cluster. Triton for maximum single-node GPU efficiency, framework-agnostic model hosting, and standardized high-throughput serving (dynamic batching, concurrent instances, TensorRT). Many stacks combine them — Ray Serve orchestrating, Triton as the optimized backend.

**Q: What does Triton's `instance_group` count do?**
A: It sets how many concurrent copies (instances) of a model run, letting Triton execute multiple requests/batches in parallel on the same or multiple GPUs. Increasing it improves concurrency and utilization up to the point of GPU memory or compute saturation.

**Q: How does Ray Serve compose multiple models?**
A: Through deployment binding/graphs: each model is a deployment, and one deployment holds handles to others, calling them with `.remote()`. Each stage scales independently, so you can give a heavy stage more replicas/GPUs than a light one — impossible in a single monolithic server.

### Pitfalls & best practices

- **Pitfall**: Treating Ray Serve replicas as free — each GPU replica consumes a full GPU; size `num_gpus` and autoscaling bounds deliberately.
- **Pitfall**: Overprovisioning Triton `instance_group` until instances contend for memory and slow each other down.
- **Best practice**: Use Triton's model analyzer to find the best batch size / instance count for your GPU.
- **Best practice**: Expose Prometheus metrics from both; drive autoscaling and capacity planning from real utilization and queue metrics.
- **Best practice**: Version models and use canary/shadow deployments for safe rollout.

---

## 5. GPU Inference Optimization

### Plain-language explanation

Beyond architecture, you can make each GPU do far more work per dollar with lower-level optimizations:

- **Quantization**: store/compute weights (and sometimes activations) in fewer bits — **FP16/BF16** (half precision), **INT8**, or even INT4. Smaller weights mean less memory and faster math, at some accuracy cost.
- **TensorRT**: NVIDIA's compiler that fuses kernels, selects optimal GPU kernels, and applies precision calibration to produce a highly optimized inference "engine."
- **ONNX / ONNX Runtime**: an interchange format plus a cross-platform optimized runtime; a common step before TensorRT or for CPU/edge deployment.
- **KV cache**: in transformer decoding, cache the key/value tensors of past tokens so each new token doesn't recompute attention over the whole sequence — turning O(n²) work into incremental O(n).
- **Kernel fusion**: combine multiple GPU operations (e.g., matmul + bias + activation) into a single kernel to cut memory traffic and launch overhead.

### Why it matters in production

These optimizations routinely deliver 2-5x throughput and large memory savings, directly lowering cost-per-request. INT8/FP16 can double batch capacity; TensorRT/FP16 can cut latency by half; a well-managed KV cache is the difference between serving 10 and 100 concurrent LLM users on the same GPU. They're often the highest-ROI work in a serving stack.

### Real-world example / use case

A team serving a ResNet-50 image classifier exports to ONNX, then builds a TensorRT FP16 engine: p99 latency drops from 18 ms to 6 ms and throughput triples on the same T4 GPU, with <0.1% accuracy change. An LLM team applies INT8 weight-only quantization (e.g., via GPTQ/AWQ) to fit a 13B model on a single 24 GB GPU with the KV cache managed by PagedAttention, tripling concurrent sessions.

### Code or config example

```python
# 1) Export PyTorch -> ONNX
import torch
model = torch.jit.load("resnet50.ts").eval().cuda()
dummy = torch.randn(1, 3, 224, 224, device="cuda")
torch.onnx.export(
    model, dummy, "resnet50.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
```

```bash
# 2) Build an optimized FP16 TensorRT engine from the ONNX model
trtexec --onnx=resnet50.onnx \
        --saveEngine=resnet50_fp16.plan \
        --fp16 \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:16x3x224x224 \
        --maxShapes=input:32x3x224x224
```

```python
# 3) INT8 weight-only quantized LLM served with a managed KV cache (vLLM)
from vllm import LLM
llm = LLM(
    model="TheBloke/Llama-2-13B-AWQ",  # pre-quantized INT4/INT8 weights
    quantization="awq",
    dtype="float16",
    gpu_memory_utilization=0.90,        # remaining memory used for the paged KV cache
)
```

### Interview questions

**Q: What is the KV cache and why is it essential for LLM serving?**
A: During autoregressive decoding, attention needs the keys and values of all previous tokens. The KV cache stores them so each new token only computes attention against cached K/V instead of recomputing the whole prefix — reducing per-token cost from quadratic to incremental. It dominates memory usage at long context, which is why techniques like PagedAttention (non-contiguous, paged allocation) exist to avoid fragmentation and pack more sequences.

**Q: What are the tradeoffs of INT8 quantization?**
A: INT8 roughly halves memory versus FP16 and speeds up compute on hardware with INT8 tensor cores, raising throughput and batch size. The cost is potential accuracy loss, which calibration (or quantization-aware training) mitigates. Some layers (e.g., certain attention/softmax) are precision-sensitive and kept in higher precision. Always validate accuracy on a held-out set after quantizing.

**Q: Where does TensorRT get its speedups?**
A: Kernel fusion (fewer kernel launches and less memory traffic), auto-tuning to pick the fastest kernels for the specific GPU, precision calibration (FP16/INT8), and layer/tensor optimizations. It compiles a static engine tailored to your shapes and hardware, so it's fast but less flexible than eager execution.

### Pitfalls & best practices

- **Pitfall**: Quantizing without re-validating accuracy — always measure on real eval data.
- **Pitfall**: A TensorRT engine is tied to GPU architecture and (often) input shape ranges; rebuild it when you change hardware.
- **Best practice**: Prefer FP16/BF16 as a near-free default; move to INT8/INT4 when memory or cost demands it.
- **Best practice**: Use paged/managed KV cache (vLLM PagedAttention or TensorRT-LLM) rather than static per-request allocation.
- **Best practice**: Optimize the whole path — fuse preprocessing, avoid host-device copies, and keep tensors on the GPU end to end.

---

## 6. Autoscaling Inference Workloads

### Plain-language explanation

Traffic is rarely constant, and idle GPUs are pure waste. **Autoscaling** adjusts the number of serving replicas to match demand. Common mechanisms in Kubernetes:

- **HPA (Horizontal Pod Autoscaler)**: scales pods based on metrics like CPU/GPU utilization or custom metrics.
- **KEDA (Kubernetes Event-Driven Autoscaling)**: extends HPA to scale on external signals — queue length (Kafka, SQS, RabbitMQ), Prometheus queries, etc. — and uniquely supports **scale-to-zero**.
- **Scale-to-zero**: shut down all replicas when there's no traffic (great for cost on spiky/dev workloads) at the price of cold-start latency when traffic returns.
- **Queue-based scaling**: scale on backlog depth rather than utilization — ideal for async/batch inference where requests land on a queue.

### Why it matters in production

GPUs are the dominant cost line in most inference budgets. Right-sizing capacity to demand can cut cost dramatically while protecting the SLO during spikes. But autoscaling GPU workloads is tricky: model load and CUDA context init make cold starts slow (tens of seconds to minutes), so naive scaling causes SLO violations. The metric you scale on matters — CPU utilization is a poor proxy for a GPU-bound service; queue depth or in-flight requests are far better.

### Real-world example / use case

An async document-processing pipeline pulls jobs from an SQS queue. With KEDA scaling on queue length (one replica per ~30 messages, max 20), the cluster runs a single warm replica overnight and bursts to 20 GPUs during business hours, returning to near-zero at night — cutting GPU spend by ~65%. A separate interactive endpoint keeps `minReplicas: 2` to absorb cold starts and never scales to zero because its SLO can't tolerate a cold start.

### Code or config example

```yaml
# HPA scaling on a custom GPU-utilization metric (via metrics adapter)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Pods
      pods:
        metric:
          name: gpu_utilization        # exported to Prometheus + custom metrics API
        target:
          type: AverageValue
          averageValue: "70"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # avoid flapping; scale down slowly
    scaleUp:
      stabilizationWindowSeconds: 0    # react fast to spikes
```

```yaml
# KEDA: queue-based scaling with scale-to-zero for async inference
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: batch-inference-scaler
spec:
  scaleTargetRef:
    name: batch-worker
  minReplicaCount: 0                    # scale to zero when the queue is empty
  maxReplicaCount: 20
  cooldownPeriod: 120                   # wait before scaling back to zero
  triggers:
    - type: aws-sqs-queue
      metadata:
        queueURL: https://sqs.us-east-1.amazonaws.com/1234/inference-jobs
        queueLength: "30"              # target ~30 messages per replica
        awsRegion: us-east-1
```

### Interview questions

**Q: Why is CPU utilization a poor autoscaling metric for GPU inference, and what's better?**
A: The bottleneck is the GPU, not the CPU — a saturated GPU may show modest CPU use, so CPU-based HPA won't scale when needed and may scale on irrelevant load. Better signals: GPU utilization, in-flight/queued requests, request rate, or latency. For async workloads, queue depth (via KEDA) directly reflects backlog.

**Q: What are the tradeoffs of scale-to-zero?**
A: It maximizes cost savings for spiky or intermittent workloads by running no replicas when idle, but the first request after idle pays a cold start (pull image, init CUDA, load model) that can be tens of seconds — unacceptable for latency-critical endpoints. Mitigations: keep min replicas > 0 for interactive services, use warm pools, pre-load/snapshot models, or provisioned/standby capacity.

**Q: How do you keep autoscaling from flapping?**
A: Use stabilization windows (longer for scale-down than scale-up), sensible cooldown periods, and target values with hysteresis so small fluctuations don't trigger churn. Scale up quickly to protect the SLO, scale down conservatively to avoid thrashing and repeated cold starts.

### Pitfalls & best practices

- **Pitfall**: Scaling on the wrong metric (CPU) for a GPU-bound service.
- **Pitfall**: Ignoring cold-start time so scale-up lags demand and violates the SLO during spikes.
- **Best practice**: Set `minReplicas` based on the latency SLO; reserve scale-to-zero for async/dev/spiky workloads.
- **Best practice**: Scale up aggressively, down conservatively; use queue-based scaling for async pipelines.
- **Best practice**: Keep container images and models cache-warm (node-local model cache, warm pools) to shrink cold starts.

---

## 7. Cost-Latency Tradeoffs

### Plain-language explanation

Every previous decision ultimately resolves into a tradeoff between **cost** (GPUs, memory, engineering) and **latency** (how fast each user is served). You can almost always buy lower latency with more/faster hardware, or lower cost by batching harder, quantizing, and scaling down — at the expense of latency or accuracy. The engineering goal is to hit the latency SLO at the lowest cost, not to minimize latency unconditionally.

### Why it matters in production

Inference cost dominates the lifetime cost of many ML products (a model is trained once but served billions of times). Small per-request savings multiply enormously at scale. Meanwhile, latency directly affects user experience and revenue. Framing decisions as "cheapest configuration that meets the SLO" keeps both finance and product satisfied, and makes tradeoffs explicit and defensible.

### Real-world example / use case

A team benchmarks three configs for an LLM endpoint with a 2 s p99 SLO: (a) A100 FP16, low batch — 700 ms p99, high cost/token; (b) A100 with continuous batching + INT8 — 1.4 s p99, 3.5x cheaper per token; (c) cheaper L4 GPUs with heavy batching — 2.3 s p99, cheapest but *violates* the SLO. They pick (b): it comfortably meets the SLO at far lower cost. This is the essence of cost-latency engineering — choose the cheapest option inside the SLO envelope.

### Code or config example

```python
# Simple cost-per-1k-requests estimator to compare serving configs
def cost_per_1k(gpu_hourly_usd: float, throughput_rps: float) -> float:
    """Given a GPU hourly price and measured throughput, cost per 1000 requests."""
    reqs_per_hour = throughput_rps * 3600
    return gpu_hourly_usd / reqs_per_hour * 1000

configs = {
    "A100 FP16 (batch 4)":       dict(price=3.67, rps=45),
    "A100 INT8 + cont. batching":dict(price=3.67, rps=160),
    "L4 INT8 (batch 32)":        dict(price=0.80, rps=70),
}
for name, c in configs.items():
    print(f"{name:32s} p99-ok? check | ${cost_per_1k(c['price'], c['rps']):.4f} / 1k req")
# Pick the lowest cost/1k that still meets the latency SLO under load tests.
```

### Interview questions

**Q: How do you frame the cost-latency tradeoff when choosing a serving configuration?**
A: Start from the latency SLO as a hard constraint, then minimize cost within it. Benchmark candidate configs (GPU type, batch settings, precision, parallelism) under realistic load, measure p99 latency and throughput, compute cost-per-request, and pick the cheapest configuration that still meets the SLO with headroom for spikes. Never optimize latency below what users can perceive at the expense of large cost increases.

**Q: What levers reduce cost without necessarily hurting latency much?**
A: Dynamic/continuous batching (huge throughput gains, small latency cost), FP16/INT8 quantization, TensorRT/kernel fusion, right-sized autoscaling (including scale-to-zero for async), spot/preemptible GPUs for fault-tolerant batch work, and choosing cheaper GPUs when the SLO allows. Each should be validated against the SLO.

**Q: When is it worth paying for lower latency?**
A: When latency has a measurable business impact — conversion, engagement, or a hard real-time constraint (fraud blocking, ads bidding) — the revenue gain can dwarf the extra hardware cost. Below the threshold of user perceptibility, further latency reduction usually isn't worth the cost.

### Pitfalls & best practices

- **Pitfall**: Optimizing latency far below the SLO at large cost with no user benefit.
- **Pitfall**: Comparing configs on synthetic uniform load instead of production traffic shape.
- **Best practice**: Treat the SLO as a constraint and cost as the objective; benchmark end-to-end under realistic load.
- **Best practice**: Track cost-per-request (or per-token) as a first-class SLO alongside latency, and revisit as traffic and hardware prices change.
- **Best practice**: Use spot instances for interruptible batch inference; keep on-demand/reserved capacity for latency-critical online serving.

---

## Summary

Distributed inference turns a trained model into a reliable, fast, and affordable production service. The journey starts at the **single node**, where the fundamentals — a warm model, async I/O, health checks, and a clear latency-vs-throughput mindset — determine everything downstream. **Batching** (dynamic for general models, continuous for LLMs) is the primary lever for GPU efficiency, trading a few milliseconds of queueing for multiples of throughput. When models outgrow a single GPU, **sharding** via tensor parallelism (fast, intra-node) and pipeline parallelism (lighter communication, cross-node) makes serving possible, with layout chosen against interconnect bandwidth.

Purpose-built frameworks — **Ray Serve** for Python-native, composable pipelines and **NVIDIA Triton** for maximum, framework-agnostic GPU throughput — save you from re-implementing replication, routing, and batching, and are often used together. **GPU optimizations** (FP16/INT8 quantization, TensorRT, ONNX, KV cache, kernel fusion) commonly yield 2-5x gains and are among the highest-ROI work available. **Autoscaling** with HPA and KEDA (including queue-based scaling and scale-to-zero) matches capacity to demand, provided you scale on the right metric and respect cold starts. Finally, every choice resolves into a **cost-latency tradeoff**: hit the latency SLO at the lowest cost, validated by end-to-end benchmarks on realistic traffic. Master these layers and you can serve anything from a tiny fraud model to a 405B-parameter LLM at production scale.

---

## Chapter Q&A

**Q1: Walk me through how you'd take a new model from a notebook to a production inference endpoint.**
A: Package the model (TorchScript/ONNX), wrap it in a serving layer with a warm model load, health/readiness checks, and async endpoints. Establish a latency SLO and load-test to find throughput. Add dynamic batching tuned to the SLO. Choose a framework (Triton for pure GPU throughput, Ray Serve for pipelines). Apply GPU optimizations (FP16/INT8, TensorRT). Deploy on Kubernetes with autoscaling driven by GPU/queue metrics, canary the rollout, wire up metrics/alerts, and track cost-per-request alongside latency.

**Q2: A GPU inference service shows 15% GPU utilization but high latency under load. What's likely wrong and how do you fix it?**
A: Almost certainly no (or ineffective) batching — requests run one at a time, so the GPU idles while the queue backs up. Enable dynamic/continuous batching with an appropriate max batch size and a small queue delay, verify preprocessing isn't blocking on CPU/host-device copies, and confirm you're not bottlenecked on the event loop or a single worker. Re-measure utilization, throughput, and p99.

**Q3: Explain the difference between tensor and pipeline parallelism and when to combine them.**
A: Tensor parallelism splits layers across GPUs with per-layer all-reduce (low latency, heavy communication — keep within NVLink). Pipeline parallelism splits the model into sequential stages communicating only at boundaries (light traffic, but pipeline bubbles). Combine them for very large models: TP within each node over NVLink, PP across nodes, plus data-parallel replicas for throughput.

**Q4: How does continuous batching differ from static batching, and why does it matter for LLMs?**
A: Static batching fixes the batch for the whole forward pass; in generation, sequences finish at different lengths, wasting compute on padded/finished slots. Continuous (in-flight) batching evicts finished sequences and admits new ones each decode step, keeping the batch full and boosting throughput 2-4x — the default in vLLM and TensorRT-LLM.

**Q5: When would you choose Triton over Ray Serve?**
A: Triton for maximum single-node GPU efficiency, framework-agnostic hosting, built-in dynamic batching, concurrent instances, and TensorRT integration — ideal for high-QPS single or multi-model serving. Ray Serve for Python-native multi-model pipelines, custom business logic, and elastic autoscaling in a Ray cluster. They're often combined, with Ray Serve orchestrating Triton backends.

**Q6: What is the KV cache, why does it dominate LLM serving memory, and how is it managed?**
A: It caches keys/values of past tokens so each new token avoids recomputing attention over the whole prefix. It grows with sequence length × batch size × layers, so at long context it dominates GPU memory and caps concurrency. PagedAttention (vLLM) allocates it in non-contiguous pages to eliminate fragmentation and pack more concurrent sequences, dramatically raising throughput.

**Q7: Describe your autoscaling strategy for (a) a latency-critical online endpoint and (b) an async batch pipeline.**
A: (a) HPA/KEDA on GPU utilization or in-flight requests, `minReplicas ≥ 2` to absorb cold starts, no scale-to-zero, aggressive scale-up and conservative scale-down with stabilization windows. (b) KEDA scaling on queue depth (SQS/Kafka), `minReplicas: 0` for scale-to-zero, one replica per N messages, with a cooldown to avoid churn — maximizing cost savings since latency is not user-facing.

**Q8: What quantization and compilation steps would you apply to cut inference cost, and what are the risks?**
A: Move to FP16/BF16 as a near-free baseline, then INT8/INT4 (calibrated or QAT) for memory-bound models; export to ONNX and build a TensorRT engine for kernel fusion and auto-tuning. Risks: accuracy regression (validate on real eval data), precision-sensitive layers needing higher precision, and engines being tied to specific GPU architectures and shape ranges (rebuild on hardware change).

**Q9: How do you decide when a model needs sharding versus fitting on one GPU?**
A: Compute the memory footprint — parameters (bits × count) + KV cache/activations + framework overhead. If it exceeds a single GPU's memory even after quantization, shard. Prefer the largest tensor-parallel degree within an NVLink domain, add pipeline parallelism only to cross nodes, and always leave headroom (e.g., `gpu_memory_utilization ≈ 0.9`) for dynamic KV-cache growth.

**Q10: How do you reason about cost-latency tradeoffs when picking a serving configuration?**
A: Treat the latency SLO as a hard constraint and cost as the objective. Benchmark candidate configs (GPU type, batch settings, precision, parallelism) under realistic production traffic, record p99 latency, throughput, and cost-per-request, then pick the cheapest configuration that meets the SLO with spike headroom. Track cost-per-request as a first-class metric and revisit as traffic and prices evolve.
