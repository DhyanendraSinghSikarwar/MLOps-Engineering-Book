# Chapter 13 — Advanced Observability for ML Infrastructure

Machine learning systems fail in ways that traditional software rarely does. A REST API can be perfectly healthy — 200s across the board, sub-50ms latency, zero exceptions — while quietly serving predictions that are wildly wrong because a feature pipeline went stale, an upstream schema drifted, or the world simply changed. Classic uptime dashboards are blind to this class of failure. Observability for ML infrastructure therefore has to answer two overlapping questions at once: *Is the system running?* (the SRE question) and *Is the system still correct?* (the ML question).

This chapter builds up the discipline from the ground: the three pillars of observability adapted for ML signals, a concrete Prometheus/Grafana stack for GPU and model metrics, distributed tracing across data-to-inference pipelines with OpenTelemetry, SLOs and error budgets tuned for inference services, and finally the root-cause analysis workflows that tie all of it together. The tone throughout is practical and interview-oriented — expect real config, real PromQL, and the questions a hiring panel actually asks.

---

## 1. Metrics, Logs, and Traces for ML Systems

### Plain-language explanation

Observability rests on three pillars. **Metrics** are numeric time series — cheap to store, easy to aggregate, ideal for dashboards and alerts (e.g., requests per second, p99 latency). **Logs** are timestamped, structured or unstructured event records — high cardinality, great for forensic detail (e.g., "prediction for request abc123 used model v4.2"). **Traces** follow a single request as it hops across services, showing where time was spent and where it failed.

ML systems layer domain-specific signals on top of the generic ones. Beyond CPU and latency, you care about **prediction latency** (often split into feature-fetch, preprocessing, and model-forward time), **model accuracy proxies** (you rarely have ground truth in real time, so you watch confidence distributions, prediction class balance, and delayed-label accuracy), **feature freshness** (how old is the newest feature value versus now), and **hardware signals** like **GPU utilization, memory, and temperature**.

### Why it matters in production

The generic pillars tell you the service is *up*; the ML signals tell you it is *right*. A model can degrade silently for weeks. Feature freshness catches a broken batch job before customers do. Confidence-distribution shifts flag concept drift before delayed labels confirm an accuracy drop. GPU metrics catch the expensive, invisible problem of a $30/hr A100 sitting at 8% utilization because your batch size is misconfigured.

### Real-world example / use case

A fraud-detection service showed flat latency and zero errors for three days while fraud losses climbed. The root cause: a nightly Spark job that computed `account_txn_count_7d` had silently failed, so the feature store served the last-good value indefinitely. There was no metric for feature age, so nothing fired. After the incident, the team added a `feature_freshness_seconds` gauge per feature group and a burn-rate alert. The next stale-feature event was caught in 11 minutes instead of 3 days.

### Code or config example

Instrumenting an inference service with the Prometheus Python client, exposing generic and ML-specific signals:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Generic signals
PRED_LATENCY = Histogram(
    "inference_latency_seconds",
    "End-to-end prediction latency",
    ["model", "version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PRED_TOTAL = Counter(
    "inference_requests_total", "Total predictions", ["model", "version", "status"]
)

# ML-specific signals
PRED_CONFIDENCE = Histogram(
    "prediction_confidence", "Model output confidence",
    ["model", "version"], buckets=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99),
)
FEATURE_FRESHNESS = Gauge(
    "feature_freshness_seconds",
    "Seconds since the newest feature value was computed",
    ["feature_group"],
)

def predict(request, model, version="v4.2"):
    start = time.perf_counter()
    status = "ok"
    try:
        features = fetch_features(request)  # returns values + computed_at timestamps
        for group, computed_at in features.freshness.items():
            FEATURE_FRESHNESS.labels(feature_group=group).set(time.time() - computed_at)
        proba = model.predict_proba(features.vector)
        PRED_CONFIDENCE.labels(model=model.name, version=version).observe(proba.max())
        return proba
    except Exception:
        status = "error"
        raise
    finally:
        PRED_LATENCY.labels(model=model.name, version=version).observe(
            time.perf_counter() - start
        )
        PRED_TOTAL.labels(model=model.name, version=version, status=status).inc()

if __name__ == "__main__":
    start_http_server(8000)  # exposes /metrics for Prometheus scrape
```

### Interview questions

**Q: You have 200ms p99 latency and no errors, but users complain about bad recommendations. Which pillar helps and why?**
A: None of the generic pillars will show it — the system is healthy by SRE standards. You need ML-specific metrics: prediction confidence distribution, class-balance of outputs, feature freshness, and delayed-label accuracy. This is the core reason ML observability is a superset of traditional observability: correctness and availability are orthogonal failure modes.

**Q: Why prefer histograms over gauges for latency?**
A: A gauge captures a single instantaneous value and hides the distribution. A histogram bins observations, letting you compute quantiles (p50/p95/p99) with `histogram_quantile()` and aggregate across instances correctly. Averages lie — p99 is where user pain lives, and you can't recover p99 from an averaged gauge.

**Q: How do you monitor accuracy when ground-truth labels arrive days later?**
A: Use proxy signals in real time (confidence distribution, prediction entropy, output class balance, input drift scores) and reconcile with true accuracy asynchronously via a delayed-labels join job that emits a lagged `model_accuracy` metric. Alert fast on proxies, confirm slowly on labels.

### Pitfalls & best practices

- **Cardinality explosions**: never put `user_id`, `request_id`, or raw feature values in metric labels — that's what logs and traces are for. Keep label sets bounded (model, version, status).
- **Pre-aggregation loss**: don't compute averages in your app and export them; export histograms and aggregate in PromQL.
- **Structure your logs** (JSON) with a `trace_id` field so logs, metrics, and traces correlate. Log the model version and feature-set hash on every prediction.
- Emit ML signals from day one — retrofitting freshness and confidence metrics after an incident is the most common regret.

---

## 2. Prometheus / Grafana for GPU and Model Metrics

### Plain-language explanation

**Prometheus** is a pull-based time-series database and monitoring system: it scrapes HTTP `/metrics` endpoints on an interval, stores the samples, and evaluates rules. **Exporters** are small processes that translate some system's state into Prometheus format — `node_exporter` for hosts, and crucially **NVIDIA DCGM exporter** for GPUs (utilization, memory, temperature, power, ECC errors, SM clocks). **PromQL** is the query language. **Grafana** sits on top for dashboards and visualization. **Recording rules** precompute expensive queries; **alerting rules** fire when a condition holds.

### Why it matters in production

GPUs are the single most expensive line item in most ML platforms. Without `DCGM_FI_DEV_GPU_UTIL` you cannot tell whether you're getting value from them — the classic pathology is 15% average utilization masked by occasional spikes, meaning you could serve the same traffic on a third of the fleet. Prometheus also unifies infra and model metrics in one query surface, which is exactly what root-cause analysis needs.

### Real-world example / use case

A team running eight A100 nodes for real-time inference saw cloud spend climb but throughput flat. A Grafana panel of `DCGM_FI_DEV_GPU_UTIL` averaged over the fleet showed a steady 12%. The bottleneck was CPU-side feature preprocessing feeding the GPU one request at a time. After adding dynamic batching, utilization rose to 70% and they decommissioned five nodes — roughly $110k/year saved, discovered purely from a GPU metric.

### Code or config example

Prometheus scrape config wiring in the DCGM exporter and the inference service:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/ml_recording.yml"
  - "rules/ml_alerts.yml"

scrape_configs:
  - job_name: "dcgm"
    static_configs:
      - targets: ["gpu-node-1:9400", "gpu-node-2:9400"]  # DCGM exporter default port
  - job_name: "inference"
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: inference-.*
        action: keep
```

Running the DCGM exporter (typically a DaemonSet on GPU nodes):

```yaml
# dcgm-exporter DaemonSet excerpt
containers:
  - name: dcgm-exporter
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04
    ports:
      - name: metrics
        containerPort: 9400
    securityContext:
      capabilities:
        add: ["SYS_ADMIN"]
```

Useful PromQL queries:

```promql
# p99 inference latency per model over 5m
histogram_quantile(
  0.99,
  sum(rate(inference_latency_seconds_bucket[5m])) by (le, model)
)

# Fleet-average GPU utilization (%)
avg(DCGM_FI_DEV_GPU_UTIL)

# GPU memory used ratio per device
DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)

# Request error ratio per model
sum(rate(inference_requests_total{status="error"}[5m])) by (model)
  /
sum(rate(inference_requests_total[5m])) by (model)
```

Recording and alerting rules:

```yaml
# rules/ml_recording.yml
groups:
  - name: ml_recording
    rules:
      - record: job:inference_latency_p99:5m
        expr: |
          histogram_quantile(0.99,
            sum(rate(inference_latency_seconds_bucket[5m])) by (le, model))
      - record: job:inference_error_ratio:5m
        expr: |
          sum(rate(inference_requests_total{status="error"}[5m])) by (model)
            / sum(rate(inference_requests_total[5m])) by (model)
```

```yaml
# rules/ml_alerts.yml
groups:
  - name: ml_alerts
    rules:
      - alert: GPUUnderutilized
        expr: avg(DCGM_FI_DEV_GPU_UTIL) < 20
        for: 30m
        labels: { severity: warning }
        annotations:
          summary: "GPU fleet under 20% for 30m — possible batching/config issue"
      - alert: GPUTooHot
        expr: DCGM_FI_DEV_GPU_TEMP > 85
        for: 5m
        labels: { severity: critical }
        annotations:
          summary: "GPU {{ $labels.gpu }} on {{ $labels.instance }} over 85C"
      - alert: FeatureStale
        expr: feature_freshness_seconds > 3600
        for: 10m
        labels: { severity: critical }
        annotations:
          summary: "Feature group {{ $labels.feature_group }} stale >1h"
```

A Grafana panel JSON snippet (a time series of p99 latency):

```json
{
  "title": "Inference p99 Latency by Model",
  "type": "timeseries",
  "datasource": { "type": "prometheus", "uid": "prom" },
  "targets": [
    {
      "expr": "job:inference_latency_p99:5m",
      "legendFormat": "{{model}}"
    }
  ],
  "fieldConfig": {
    "defaults": { "unit": "s", "custom": { "drawStyle": "line" } }
  }
}
```

### Interview questions

**Q: Pull vs push — why is Prometheus pull-based, and when do you need the Pushgateway?**
A: Pull lets Prometheus control scrape timing, detect target down (a failed scrape is itself a signal via `up`), and avoids apps needing to know the monitoring endpoint. The exception is short-lived batch jobs that die before a scrape — for those, push final metrics to the Pushgateway, which Prometheus then scrapes. Do *not* use Pushgateway for long-running services.

**Q: What's the difference between a recording rule and an alerting rule?**
A: A recording rule precomputes an expression and stores it as a new time series, cheapening dashboards and alert evaluation. An alerting rule evaluates an expression and, when true for the `for` duration, fires an alert to Alertmanager. Recording rules are commonly the building blocks that alerting rules reference.

**Q: A single GPU shows 100% `DCGM_FI_DEV_GPU_UTIL` but throughput is low. What does that metric actually mean?**
A: `GPU_UTIL` is the fraction of time *at least one kernel was running* — it says nothing about how many SMs or how much compute was used. You can be "100% utilized" running a tiny kernel. To judge real efficiency, also look at SM activity (`DCGM_FI_PROF_SM_ACTIVE`), memory bandwidth, and achieved FLOPs. Utilization is necessary but not sufficient.

### Pitfalls & best practices

- **`rate()` needs counters**, and always use `rate()`/`irate()` before aggregating counters — never `sum()` raw counters across restarts.
- Match `rate()` window to at least 4x scrape interval (e.g., `[5m]` for 15s scrapes) to survive missed scrapes.
- Set retention and use recording rules for expensive quantile queries feeding dashboards — histogram_quantile over many series is costly at query time.
- Give DCGM exporter the right node access (DaemonSet, MIG-aware config for partitioned GPUs) or you'll get gaps.
- Alert on symptoms (latency, error ratio, freshness), not causes (CPU); page humans only for user-facing SLO burn.

---

## 3. Distributed Tracing with OpenTelemetry Across Pipelines

### Plain-language explanation

A **trace** represents one logical request's journey; it is a tree of **spans**, each span a timed unit of work (an HTTP call, a DB query, a model forward pass). Spans carry attributes and are linked by a shared **trace context** propagated across process boundaries. **OpenTelemetry (OTel)** is the vendor-neutral standard: SDKs instrument your code, the **Collector** receives/processes/exports spans, and backends like Jaeger, Tempo, or Zipkin store and visualize them.

For ML, a single prediction often spans: gateway → feature retrieval (online store + maybe on-the-fly compute) → preprocessing → model inference (possibly a remote model server) → post-processing. Tracing shows you exactly which stage ate the latency budget, across service boundaries, for one request.

### Why it matters in production

Metrics tell you *that* p99 is high; traces tell you *where*. When your inference pipeline touches five services, "the model is slow" is usually wrong — it's frequently the feature fetch or a cold cache. Traces also carry the `trace_id` that stitches logs and metrics into one story, which is the backbone of the RCA workflows in section 5.

### Real-world example / use case

A recommendation API breached its 150ms SLO intermittently. Latency histograms showed a bimodal distribution but not the cause. Traces revealed that ~8% of requests hit a fallback code path that synchronously recomputed an embedding when the online feature store missed — a 400ms span that only appeared on cache misses. The fix was pre-warming embeddings; the bimodality vanished. Without span-level visibility across the cache and compute services, the team would have kept "optimizing the model" that was never the problem.

### Code or config example

OpenTelemetry Python instrumentation across the data → feature → inference stages, with context propagation:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor

resource = Resource.create({"service.name": "inference-gateway"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True))
)
trace.set_tracer_provider(provider)
RequestsInstrumentor().instrument()  # auto-propagates trace context on outbound HTTP

tracer = trace.get_tracer(__name__)

def handle_request(req):
    with tracer.start_as_current_span("predict") as span:
        span.set_attribute("model.name", "ranker")
        span.set_attribute("model.version", "v4.2")

        with tracer.start_as_current_span("feature.fetch") as fspan:
            features = fetch_features(req)          # remote call auto-instrumented
            fspan.set_attribute("feature.cache_hit", features.cache_hit)
            fspan.set_attribute("feature.count", len(features.vector))

        with tracer.start_as_current_span("preprocess"):
            x = preprocess(features)

        with tracer.start_as_current_span("model.infer") as ispan:
            result = model.predict(x)
            ispan.set_attribute("prediction.confidence", float(result.confidence))

        span.set_attribute("prediction.class", result.label)
        return result
```

OpenTelemetry Collector config that receives OTLP and exports to Tempo/Jaeger, with sampling:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc: { endpoint: 0.0.0.0:4317 }
processors:
  batch: {}
  tail_sampling:
    decision_wait: 5s
    policies:
      - name: errors
        type: status_code
        status_code: { status_codes: [ERROR] }
      - name: slow
        type: latency
        latency: { threshold_ms: 200 }
      - name: baseline
        type: probabilistic
        probabilistic: { sampling_percentage: 5 }
exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls: { insecure: true }
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [tail_sampling, batch]
      exporters: [otlp/tempo]
```

### Interview questions

**Q: How does trace context propagate across service and process boundaries?**
A: Via context propagation using the W3C Trace Context standard — the `traceparent` HTTP header carries trace ID, parent span ID, and sampling flags. OTel instrumentation libraries inject the header on outbound calls and extract it on inbound, so child spans in the downstream service attach to the same trace. For non-HTTP transports (Kafka, gRPC), propagators inject the same context into message metadata.

**Q: Head-based vs tail-based sampling — which for ML inference and why?**
A: Head-based decides at the trace's start (cheap, but you might drop the rare slow/errored trace). Tail-based buffers all spans and decides after seeing the whole trace, so you can always keep errors and slow requests while sampling the boring majority. For ML services where the interesting failures are rare, tail sampling (as in the Collector config above) is preferred despite its higher memory cost.

**Q: You add tracing and latency gets worse. Why, and how do you avoid it?**
A: Synchronous span export blocks the request path. Use the `BatchSpanProcessor` (async, batched export) not `SimpleSpanProcessor`, sample aggressively, keep attribute payloads small, and export to a local Collector over gRPC rather than a remote backend directly. Instrumentation overhead should be low-single-digit microseconds per span when done right.

### Pitfalls & best practices

- Always propagate context — a broken propagator produces orphan traces that look like separate requests.
- Name spans by operation, not by data (`feature.fetch`, not `fetch_features_for_user_123`) — span names are low-cardinality dimensions.
- Put the `trace_id` into your structured logs so you can jump from a trace to its logs and back.
- Instrument the boundaries first (gateway, feature store, model server) — that 80% of value comes from a handful of spans.
- Never log PII or raw features into span attributes.

---

## 4. SLOs for ML Services

### Plain-language explanation

An **SLI** (Service Level Indicator) is a measured quantity of service quality — e.g., the fraction of inference requests served under 200ms and without error. An **SLO** (Service Level Objective) is a target for an SLI over a window — e.g., "99.5% of requests succeed within 200ms over 30 days." The **error budget** is the allowed failure: `1 - SLO`. If your SLO is 99.5%, your budget is 0.5% of requests — spend it on risk (deploys, experiments) and stop shipping when it's exhausted.

**Multi-window, multi-burn-rate alerts** watch how fast you're consuming that budget over both a short and a long window, giving fast pages for severe burns and slow pages for gradual ones — with far fewer false alarms than naive threshold alerts.

### Why it matters in production

SLOs turn "is it healthy?" into a quantified, budget-driven decision. They align engineering and product on how reliable is reliable enough, and they justify slowing feature velocity when reliability suffers. For ML services you can define SLOs on availability *and* on ML-specific quality — e.g., a freshness SLO ("99% of feature reads under 5 minutes old") or a quality SLO on delayed-label accuracy.

### Real-world example / use case

An inference platform set a latency SLO of 99.9% of requests under 300ms over 28 days. A new model version increased tail latency, and the multi-window burn-rate alert fired within an hour of a fast burn (14.4x) rather than waiting for the monthly SLO to visibly dip. The team rolled back before the budget was meaningfully dented. The error-budget policy — "freeze deploys when >50% of monthly budget is spent" — automatically paused a risky migration the following week.

### Code or config example

Recording the SLI and a multi-window multi-burn-rate alert (the Google SRE pattern):

```yaml
# rules/slo.yml
groups:
  - name: slo_inference
    rules:
      # Good = fast AND non-error. SLI = good / total.
      - record: slo:inference_good_ratio:rate5m
        expr: |
          (
            sum(rate(inference_latency_seconds_bucket{le="0.3"}[5m]))
            - sum(rate(inference_requests_total{status="error"}[5m]))
          )
          / sum(rate(inference_requests_total[5m]))
      # Error-budget burn rate = (1 - SLI) / (1 - SLO). SLO = 0.999
      - record: slo:inference_burn_rate:5m
        expr: (1 - slo:inference_good_ratio:rate5m) / (1 - 0.999)
      - record: slo:inference_burn_rate:1h
        expr: |
          (1 - (
            (sum(rate(inference_latency_seconds_bucket{le="0.3"}[1h]))
             - sum(rate(inference_requests_total{status="error"}[1h])))
            / sum(rate(inference_requests_total[1h]))
          )) / (1 - 0.999)

      # Fast burn: 14.4x over 5m AND 1h => exhausts 28d budget in ~2 days
      - alert: InferenceErrorBudgetFastBurn
        expr: |
          slo:inference_burn_rate:5m > 14.4
          and slo:inference_burn_rate:1h > 14.4
        for: 2m
        labels: { severity: critical }
        annotations:
          summary: "Fast error-budget burn on inference SLO (page immediately)"

      # Slow burn: 3x over 1h AND 6h => gradual erosion
      - alert: InferenceErrorBudgetSlowBurn
        expr: |
          slo:inference_burn_rate:1h > 3
          and slo:inference_burn_rate:6h > 3
        for: 15m
        labels: { severity: warning }
        annotations:
          summary: "Slow error-budget burn on inference SLO (ticket)"
```

Remaining error budget as a PromQL expression for a dashboard:

```promql
# Fraction of 28-day error budget remaining
1 - (
  (1 - (sum(rate(inference_requests_total{status="ok"}[28d]))
        / sum(rate(inference_requests_total[28d]))))
  / (1 - 0.999)
)
```

### Interview questions

**Q: Difference between SLI, SLO, and SLA?**
A: The SLI is the measurement (e.g., success ratio), the SLO is the internal target for that measurement (e.g., 99.9%), and the SLA is the external, contractual promise to customers usually with financial penalties — and is set looser than the SLO so you have internal headroom. You alert on the SLO, not the SLA.

**Q: Why multi-window multi-burn-rate instead of a simple threshold alert?**
A: A single threshold either pages too often on transient blips or too slowly on real degradation. Requiring both a long and a short window to exceed a burn rate confirms the problem is sustained (short window) and real (long window), cutting false positives while still catching fast severe burns quickly. Different burn-rate/window pairs map to how urgently budget is being consumed.

**Q: How do you set SLOs for ML quality, not just availability?**
A: Pick measurable proxies with clear thresholds: feature freshness ("99% of reads < 5 min stale"), calibration or confidence bounds, and delayed-label accuracy over a rolling window ("weekly accuracy ≥ 0.92 for 95% of weeks"). Treat sustained quality-SLO breaches with the same error-budget discipline — freeze model deploys when the quality budget is spent.

### Pitfalls & best practices

- Base SLOs on the user journey, not internal components — users feel end-to-end latency, not one microservice's.
- Don't set SLOs at 100%; that leaves no error budget and makes every blip an emergency. 99.9% is often plenty.
- Measure the SLI where the user experiences it (at the gateway/client), not deep inside.
- Write an explicit error-budget policy (what happens when it's exhausted) before you need it.
- Keep the window realistic (28–30 days is common) and align burn-rate thresholds to that window.

---

## 5. Root-Cause Analysis Workflows

### Plain-language explanation

**Root-cause analysis (RCA)** is the structured process of going from a symptom (an alert) to the underlying cause. In observability terms it means correlating the three pillars: a metric tells you *what* broke and *when*, traces tell you *where* in the request path, and logs give you the *why* at the line level. The `trace_id` and consistent labels (service, model, version) are the join keys that let you pivot between them.

ML adds a distinctive fork in the RCA tree: **is this an infrastructure problem or a model/data problem?** Infra causes look like latency spikes, error rates, OOMs, GPU throttling, or throughput collapse. Model/data causes look like drifting input distributions, falling confidence, changing prediction mix, and — eventually — accuracy decay, often with the infra layer perfectly healthy.

### Why it matters in production

The most expensive incidents are the ones where you fix the wrong layer — restarting pods and scaling nodes for hours when the real problem is a drifted feature. A disciplined RCA workflow that explicitly checks "infra vs data/model" early prevents that. Correlated observability also collapses mean-time-to-resolution: jumping from a burn-rate alert straight to the offending trace to the exact log line is minutes, not hours.

### Real-world example / use case

An image-classification service saw its confidence distribution collapse and its "unknown" class rate triple. Infra was green — latency, GPU, errors all nominal. The RCA workflow: (1) burn-rate alert on the quality SLO fired; (2) drift dashboard showed input pixel-intensity distribution had shifted; (3) traces on affected requests all shared a new upstream client version; (4) logs revealed that client was sending images in a new color space (BGR vs RGB). The cause was a data/contract change, not the model. Fix: normalize color space at ingest. Had the team scaled GPUs (the reflex), they'd have burned hours and money on the wrong layer.

### Code or config example

A decision-oriented RCA workflow and the correlating queries. First, a quick "infra vs model" triage in PromQL:

```promql
# --- Infra suspects ---
# Latency regression?
job:inference_latency_p99:5m > 0.3
# GPU throttling / thermal?
DCGM_FI_DEV_GPU_TEMP > 85 or DCGM_FI_DEV_GPU_UTIL == 100
# OOM / restarts?
increase(kube_pod_container_status_restarts_total{pod=~"inference-.*"}[15m]) > 0

# --- Model / data suspects (infra can be green) ---
# Confidence collapse: mean confidence dropping
histogram_quantile(0.5, sum(rate(prediction_confidence_bucket[15m])) by (le))
# Prediction-mix shift: share of one class spiking
sum(rate(inference_requests_total{status="ok"}[15m])) by (predicted_class)
  / ignoring(predicted_class) group_left
  sum(rate(inference_requests_total{status="ok"}[15m]))
# Feature staleness
max(feature_freshness_seconds) by (feature_group) > 900
```

Then pivot from a firing alert to the traces and logs. Given an alert carrying a representative `trace_id`, a LogQL query (Grafana Loki) joins logs to that trace:

```logql
{app="inference-gateway"} | json | trace_id="4bf92f3577b34da6a3ce929d0e0e4736"
```

A compact runbook encoded as a decision tree (documentation the on-call follows):

```text
ALERT: InferenceErrorBudgetFastBurn
 ├─ Is p99 latency up AND GPU temp>85 / restarts>0?
 │    └─ YES -> INFRA path: check node health, HPA, GPU throttle, rollback infra change
 ├─ Is error ratio up with 5xx in logs?
 │    └─ YES -> code/dep path: check last deploy, dependency health, rollback model server
 ├─ Infra green, but confidence down / prediction-mix shifted / drift score up?
 │    └─ YES -> DATA/MODEL path: check feature freshness, upstream schema/contract,
 │              compare input distribution vs training baseline, consider model rollback
 └─ None clear -> open trace for a slow/errored sample, walk spans, read correlated logs
```

### Interview questions

**Q: Walk me through triaging "the model got worse" when all infra dashboards are green.**
A: Green infra points to a data/model cause. I'd check, in order: feature freshness (broken pipeline?), input drift scores against the training baseline (distribution shift?), prediction confidence and class-mix (model reacting to new inputs?), and any upstream schema/contract change (via traces sharing a new client version). I'd confirm with the delayed-label accuracy metric once labels arrive. Fix depends on cause: re-run the pipeline, patch the ingest contract, or roll back/retrain the model — not scale infra.

**Q: How do metrics, logs, and traces concretely correlate during an incident?**
A: Metrics detect and scope (which model/version/region, since when). I pick an exemplar trace for an affected request — many setups attach trace exemplars directly to histogram buckets — walk its spans to find the failing stage, then filter logs by that `trace_id` for the exact error. Consistent labels (service, model, version) are the join keys; without them the three pillars are three silos.

**Q: How do you distinguish concept drift from a data-pipeline bug? They can look similar.**
A: A pipeline bug is usually abrupt and correlates with a deploy or a job failure — feature freshness jumps, a whole feature goes null/constant, and the change is step-shaped. Concept drift is gradual, affects the input distribution's shape rather than its availability, and has no corresponding code/job event. Checking freshness and null-rates first cheaply rules out the pipeline bug before you invest in the drift hypothesis.

### Pitfalls & best practices

- Standardize labels across metrics, logs, and traces or correlation is impossible — treat `service`, `model`, `version`, and `trace_id` as sacred.
- Check the cheap, common causes first (recent deploy, feature freshness, a failed job) before exotic ones.
- Encode RCA as runbooks tied to each alert; don't rely on the tribal knowledge of whoever's on call.
- Use exemplars to jump metric → trace in one click; it's the single biggest MTTR win.
- Always ask "infra or data/model?" early — the reflex to scale/restart wastes the most time on ML incidents.
- Write blameless postmortems and feed each new failure mode back as a new metric or alert.

---

## Summary

Observability for ML infrastructure is traditional observability plus a correctness dimension that classic dashboards ignore. The **three pillars** — metrics, logs, traces — remain the foundation, but ML systems demand extra signals: prediction latency, confidence and class-mix proxies for accuracy, feature freshness, and GPU utilization. **Prometheus and Grafana**, with the **DCGM exporter** for GPUs, give you a unified query surface over infra and model metrics; PromQL plus recording and alerting rules turn raw samples into actionable dashboards and pages. **OpenTelemetry** brings distributed tracing across the data → feature → inference pipeline, propagating context so you can see exactly which stage spent the latency budget, with tail sampling to keep the rare interesting traces. **SLOs and error budgets** convert "is it healthy?" into a quantified, policy-driven decision, and **multi-window burn-rate alerts** catch both fast and slow degradation with few false alarms. Finally, **RCA workflows** tie it all together by correlating the pillars through shared labels and `trace_id`, always asking the ML-specific question — *infra or data/model?* — so you fix the right layer fast.

Done well, these practices mean your ML platform tells you not just that it is up, but that it is still right — and when it is not, points you straight at why.

## Chapter Q&A

**Q1. Why is ML observability a superset of traditional observability?**
A: Traditional observability answers availability ("is it up, fast, error-free?"). ML systems can be perfectly available while serving wrong predictions due to drift, stale features, or upstream contract changes. So ML observability adds a correctness dimension — freshness, confidence/class-mix proxies, drift scores, and delayed-label accuracy — on top of the standard latency/error/saturation signals.

**Q2. When should you use logs vs metrics vs traces?**
A: Metrics for aggregate trends and alerting (cheap, low-cardinality). Logs for high-detail forensic events (why a specific request behaved as it did). Traces for the path and timing of a single request across services. Correlate them via consistent labels and `trace_id`.

**Q3. Why does Prometheus use a pull model, and what breaks that model?**
A: Pull gives Prometheus control over scrape timing and free liveness detection via `up`, and decouples apps from the monitoring system. Short-lived batch jobs break it because they can die before being scraped — handle those with the Pushgateway, never long-running services.

**Q4. What does `DCGM_FI_DEV_GPU_UTIL` really measure, and why is it insufficient alone?**
A: It's the fraction of time at least one kernel was executing, not how much of the GPU's compute was used. A trivial kernel can peg it at 100%. Combine it with SM activity, memory bandwidth, and achieved FLOPs to judge true efficiency.

**Q5. Explain recording rules vs alerting rules with an example.**
A: A recording rule precomputes and stores an expression (e.g., `job:inference_latency_p99:5m`) to make dashboards and alerts cheaper. An alerting rule evaluates a condition and fires after its `for` duration (e.g., burn-rate > 14.4 for 2m). Alerting rules often reference recording rules.

**Q6. How does trace context propagate across services in OpenTelemetry?**
A: Through the W3C `traceparent` header (and `tracestate`), carrying trace ID, parent span ID, and sampling flags. OTel instrumentation injects it on outbound calls and extracts it on inbound, attaching downstream spans to the same trace; for Kafka/gRPC, propagators embed the same context in message metadata.

**Q7. Head-based vs tail-based sampling for ML inference?**
A: Head-based decides upfront — cheap but may drop rare errors/slow traces. Tail-based buffers the whole trace and decides after, so it can always retain errors and slow requests while sampling the common case. ML services favor tail sampling because the valuable traces are rare.

**Q8. What is an error budget and how does it change engineering behavior?**
A: It's `1 - SLO`, the allowed amount of failure over a window. It quantifies acceptable risk: spend it on deploys and experiments, and when it's exhausted, a policy kicks in (e.g., freeze releases, prioritize reliability). It replaces arguments about "enough reliability" with data.

**Q9. Why multi-window multi-burn-rate alerting?**
A: It requires both a short and a long window to exceed a burn-rate threshold, so alerts confirm the problem is both severe (short window) and sustained/real (long window). This catches fast catastrophic burns quickly while filtering the transient blips that plague single-threshold alerts.

**Q10. In an ML incident with green infra dashboards, how do you find the root cause?**
A: Green infra points at data/model. Check feature freshness (pipeline bug?), input drift vs training baseline (concept drift?), confidence and prediction-mix shifts, and upstream schema/contract changes surfaced via traces. Pipeline bugs are abrupt and event-correlated; drift is gradual. Confirm with delayed-label accuracy, then fix the correct layer — re-run pipeline, patch ingest, or roll back/retrain — rather than reflexively scaling infrastructure.

---

*This is the final chapter of* **MLOps Engineering: From Fundamentals to Production Platforms**. *From the fundamentals of reproducible pipelines through feature stores, training and serving infrastructure, deployment strategies, and now advanced observability, you now hold the full arc of building and operating ML platforms in production. Observability is where all of it converges: the discipline that lets you trust — and prove — that everything you built in the preceding chapters keeps working, and keeps being right, once real traffic and a changing world arrive. Instrument early, alert on what users feel, and always ask whether it's the infrastructure or the data. Thank you for reading.*
