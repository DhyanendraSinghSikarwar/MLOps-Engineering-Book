# Chapter 9 — Helm

Kubernetes gives you powerful primitives — Deployments, Services, ConfigMaps, HorizontalPodAutoscalers — but it hands them to you as flat, static YAML. The moment you need to ship the *same* application to dev, staging, and production, or deploy fifteen model servers that differ only by image tag and replica count, raw manifests collapse under copy-paste duplication. Helm is the answer: it is the de-facto package manager for Kubernetes, letting you template, parameterize, version, and distribute entire applications as reusable **charts**.

For MLOps this matters enormously. A model serving stack is rarely a single manifest — it is a Deployment running your inference container, a Service exposing it, an HPA to absorb traffic spikes, a ConfigMap holding model configuration, and often secrets, ingress, and monitoring rules. Helm bundles all of that into one installable, versioned unit you can promote across environments and roll back in seconds. This chapter takes you from the anatomy of a chart through Go templating, values management, packaging a real ML serving app, versioning and release lifecycle, and finally integrating Helm with ArgoCD for GitOps-driven delivery. Everything here targets **Helm 3**, which removed Tiller and stores release state as Kubernetes Secrets.

---

## 1. Chart Basics

### Plain-language explanation

A **chart** is a Helm package: a directory (or a `.tgz` archive) containing all the Kubernetes manifests needed to run an application, plus metadata and default configuration. When you `helm install` a chart, Helm renders the templates into concrete manifests and applies them to the cluster as a **release** — a named, tracked instance of that chart. Install the same chart twice with different release names and you get two independent deployments.

The canonical chart structure looks like this:

```
model-server/
├── Chart.yaml          # chart metadata (name, version, appVersion)
├── values.yaml         # default configuration values
├── templates/          # templated Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── configmap.yaml
│   ├── _helpers.tpl    # reusable named templates
│   └── NOTES.txt       # post-install message shown to the user
├── charts/             # subcharts / dependencies live here
└── .helmignore         # files to exclude when packaging
```

### Why it matters in production

Charts turn "a pile of YAML someone applied by hand" into a repeatable, versioned artifact. That repeatability is the foundation of reliable deployments: the exact same chart version that passed testing in staging is the one you promote to production. Helm also tracks release history, so a broken deploy is one `helm rollback` away instead of a frantic `kubectl apply` of an old file you hope still exists. In regulated ML environments, being able to point at "release 14, chart version 2.3.1, appVersion 1.8.0" gives you the audit trail auditors and incident reviewers demand.

### Real-world example / use case

Your team ships a fraud-detection model server. Instead of maintaining separate manifest folders for `dev`, `staging`, and `prod`, you build one `fraud-model` chart. Deploying to staging is `helm upgrade --install fraud-staging ./fraud-model -f values-staging.yaml`; production is the same command with `-f values-prod.yaml`. One chart, three releases, zero duplicated YAML.

### Code or config example

```yaml
# Chart.yaml
apiVersion: v2
name: model-server
description: A Helm chart for serving ML models
type: application
version: 0.1.0        # the chart's own version (SemVer)
appVersion: "1.4.0"   # the version of the app/model image being deployed
maintainers:
  - name: MLOps Team
    email: mlops@example.com
```

```bash
# Install a chart as a release named "fraud-detector"
helm install fraud-detector ./model-server

# Idempotent install-or-upgrade — the workhorse command in CI/CD
helm upgrade --install fraud-detector ./model-server \
  --namespace ml-serving --create-namespace

# See what will be rendered WITHOUT touching the cluster
helm template fraud-detector ./model-server

# Inspect a live release
helm list -n ml-serving
helm status fraud-detector -n ml-serving
```

### Interview questions

**Q: What is the difference between a chart and a release?**
A chart is the packaged template — the reusable definition of an application, versioned and distributable. A release is a specific, named instance of that chart deployed into a cluster. Installing one chart three times produces three releases, each with its own configuration and independent lifecycle (history, rollbacks, upgrades).

**Q: What does `helm upgrade --install` do, and why is it preferred in CI pipelines?**
It installs the release if it does not yet exist and upgrades it if it does. This makes deployment steps idempotent — the pipeline does not need to branch on "first deploy vs. subsequent deploy," which eliminates a whole class of "release already exists" or "release not found" failures.

**Q: Where does Helm 3 store release state, and why is that a security improvement over Helm 2?**
Helm 3 stores release metadata as Kubernetes Secrets (by default) in the release's namespace. Helm 2 relied on Tiller, a cluster-wide server component running with broad privileges, which was a notorious attack surface. Helm 3's client-only model uses your existing kubeconfig RBAC, so permissions are scoped to what the user can already do.

### Pitfalls & best practices

- **Do not hand-edit resources Helm manages.** Out-of-band `kubectl edit` changes cause drift that Helm may overwrite or conflict with on the next upgrade.
- **Always run `helm template` or `helm install --dry-run --debug`** before shipping to catch rendering errors early.
- **Keep `NOTES.txt` useful** — it is the first thing an operator sees after install; put the service URL and a quick health check there.
- **Commit `Chart.lock`** when using dependencies so builds are reproducible.

---

## 2. Templating

### Plain-language explanation

Helm templates are Kubernetes manifests with **Go template** directives embedded in `{{ }}`. At render time Helm substitutes values, evaluates logic, and produces plain YAML. The main inputs to a template are:

- `.Values` — configuration from `values.yaml` and overrides.
- `.Release` — release metadata (`.Release.Name`, `.Release.Namespace`, `.Release.Revision`).
- `.Chart` — data from `Chart.yaml` (`.Chart.Name`, `.Chart.Version`, `.Chart.AppVersion`).
- `.Capabilities` — cluster info such as available API versions.
- `.Files` — access to non-template files in the chart.

Helm ships the standard Go template functions plus the **Sprig** library (string, math, date, list helpers) and Helm-specific functions like `include`, `tpl`, `required`, and `toYaml`. **Pipelines** chain functions with `|`, e.g. `{{ .Values.name | upper | quote }}`.

### Why it matters in production

Templating is what makes a chart *reusable*. Without it you would fork YAML for every environment. With it, one deployment template adapts its image, replica count, resource limits, and env vars from values — so the difference between a 1-replica dev pod and a 20-replica GPU production fleet is just a values file. Named templates (`_helpers.tpl`) keep labels, selectors, and naming consistent across every resource, which prevents the subtle mismatches (e.g., a Service selector that no longer matches Deployment labels) that cause silent outages.

### Real-world example / use case

Every resource in your ML chart needs a consistent set of labels for Prometheus scraping and cost allocation (`team`, `model`, `env`). Rather than repeat that label block in five files, you define a `model-server.labels` named template once in `_helpers.tpl` and `include` it everywhere. Change the label schema once, and every manifest updates.

### Code or config example

```yaml
# templates/_helpers.tpl
{{/* Expand the name of the chart. */}}
{{- define "model-server.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Fully qualified app name: <release>-<chart>, capped at 63 chars. */}}
{{- define "model-server.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "model-server.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Common labels applied to every resource. */}}
{{- define "model-server.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
app.kubernetes.io/name: {{ include "model-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/* Selector labels — MUST be stable across upgrades. */}}
{{- define "model-server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "model-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
```

```yaml
# templates/deployment.yaml (excerpt showing range/if/pipelines)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "model-server.fullname" . }}
  labels:
    {{- include "model-server.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "model-server.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "model-server.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          {{- if .Values.env }}
          env:
            {{- range $key, $val := .Values.env }}
            - name: {{ $key }}
              value: {{ $val | quote }}
            {{- end }}
          {{- end }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
```

Note the essentials: `include` (not `template`) so the output can be piped into `nindent`; `nindent` to fix indentation; `| default` to fall back to `appVersion`; `range` over a map with `$key, $val`; `if` to render env only when present; and `toYaml` to serialize an arbitrary values block.

### Interview questions

**Q: What is the difference between `include` and `template`, and why does it matter?**
`template` is a Go action that *writes* output directly and cannot be used inside a pipeline. `include` is a Helm function that *returns* the rendered text as a string, so you can pipe it — most importantly through `indent`/`nindent` to get correct YAML indentation. Because you almost always need to re-indent named-template output, `include` is the idiomatic choice.

**Q: What do `indent` and `nindent` do, and why are they everywhere in Helm charts?**
Both add leading spaces to every line of a string. `nindent` also prepends a newline first. They exist because YAML is whitespace-sensitive: a named template or a `toYaml` block produces text at column zero, and you must shift it to the correct depth. `nindent 8` means "start on a new line and indent 8 spaces."

**Q: What does the `required` function do?**
`required "message" .Values.someField` fails template rendering with your custom error message if the value is empty or missing. It is how you enforce that a chart consumer supplies mandatory values (like an image repository) instead of silently rendering a broken manifest.

### Pitfalls & best practices

- **Whitespace control:** use `{{-` and `-}}` to trim surrounding whitespace, but be deliberate — over-trimming glues lines together and breaks YAML.
- **Quote strings that could be misread as numbers or booleans** (e.g., version tags, `"true"`, `"1.0"`) with `| quote` to avoid YAML type coercion.
- **Never put secrets in templates in plaintext** — integrate with a secrets manager or `helm-secrets`/SOPS.
- **Selector labels must be immutable** across upgrades; changing them forces a Deployment replacement and breaks the existing ReplicaSet linkage.
- Prefer `toYaml` for passing structured blocks (resources, nodeSelector, tolerations) rather than templating each field by hand.

---

## 3. Values Management

### Plain-language explanation

`values.yaml` holds a chart's default configuration as a hierarchy of key-value pairs. Consumers override those defaults at install/upgrade time, and Helm merges everything into the final `.Values` object with a clear precedence. From lowest to highest priority:

1. The chart's own `values.yaml`.
2. A parent chart's values (for subchart values).
3. Files passed with `-f`/`--values` (later files override earlier ones).
4. Individual `--set`, `--set-string`, `--set-file`, `--set-json` flags (highest).

This layering is what enables **per-environment configuration**: a base `values.yaml` for defaults plus `values-prod.yaml` for the overrides that differ in production.

### Why it matters in production

Environment drift is a leading cause of "works in staging, breaks in prod" incidents. Structured values files make the differences between environments *explicit and reviewable* — a diff of `values-staging.yaml` vs `values-prod.yaml` tells you exactly what changes (replicas, resource limits, model version, autoscaling bounds). Keeping those files in Git gives you change history and PR review over configuration, which is essential for ML systems where a wrong model tag or an undersized GPU request has real cost and correctness consequences.

### Real-world example / use case

Your recommender model runs on CPU in dev (1 replica, tiny limits) and on GPU in prod (min 3 / max 30 replicas, 1 GPU each, a larger model artifact). Base values encode the CPU dev defaults; `values-prod.yaml` overrides the image tag, resources, GPU node selector, and HPA bounds. CI runs `helm upgrade --install rec-prod ./model-server -f values-prod.yaml --set image.tag=$GIT_SHA`, injecting the freshly built image tag at deploy time.

### Code or config example

```yaml
# values.yaml (defaults — safe for dev)
replicaCount: 1

image:
  repository: registry.example.com/ml/model-server
  tag: ""              # empty -> falls back to Chart.appVersion
  pullPolicy: IfNotPresent

modelConfig:
  modelName: recommender
  modelVersion: "3"
  maxBatchSize: 8
  timeoutSeconds: 30

resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "1"
    memory: "2Gi"

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 5
  targetCPUUtilizationPercentage: 70

env:
  LOG_LEVEL: info
```

```yaml
# values-prod.yaml (only the deltas from defaults)
replicaCount: 3

image:
  tag: "3.2.0"

resources:
  requests:
    cpu: "2"
    memory: "8Gi"
    nvidia.com/gpu: "1"
  limits:
    nvidia.com/gpu: "1"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 30
  targetCPUUtilizationPercentage: 60

env:
  LOG_LEVEL: warn
```

```bash
# Layered overrides: base defaults < prod file < inline --set
helm upgrade --install rec-prod ./model-server \
  -f values-prod.yaml \
  --set image.tag=1a2b3c4 \
  --set-string modelConfig.modelVersion="4" \
  -n ml-serving

# Subchart values are namespaced under the subchart's alias/name:
helm upgrade --install rec-prod ./model-server \
  --set redis.architecture=standalone \
  --set redis.auth.enabled=false
```

### Interview questions

**Q: What is the precedence order when the same value is set in multiple places?**
Lowest to highest: the chart's `values.yaml`, then parent chart values (for subcharts), then `-f`/`--values` files in the order given (later wins), then `--set` flags. So an inline `--set` always beats a values file, which always beats the chart defaults.

**Q: How do you override a subchart's values from the parent chart?**
You nest the values under a key matching the subchart's name (or its `alias`). For example, to configure a `redis` dependency you set `redis.auth.enabled: false` in the parent's values. Global values can be shared across all subcharts under the top-level `global:` key.

**Q: When should you use `--set` versus a values file?**
Use values files for structured, reviewed, environment-specific configuration that lives in Git. Use `--set` for a small number of dynamic, deploy-time values — most commonly the image tag/Git SHA injected by CI. Overusing `--set` for complex data leads to fragile, unreadable commands and quoting nightmares (arrays and dots need escaping).

### Pitfalls & best practices

- **`--set` type coercion:** `--set` infers types, so a version like `1.0` becomes a float. Use `--set-string` for anything that must stay a string.
- **Keep environment files minimal** — only the deltas. A prod file that duplicates all defaults drifts silently when defaults change.
- **Document every value** with comments in `values.yaml`; it doubles as the chart's API reference.
- **Validate with a JSON schema** (`values.schema.json`) so bad configuration fails fast at install time rather than at pod crash time.
- Beware that `--set` on a list *replaces* the whole list; there is no deep-merge for arrays.

---

## 4. Packaging ML Serving Apps

### Plain-language explanation

A production model server is more than one Deployment. A complete chart typically bundles: a **Deployment** running the inference container, a **Service** to give it a stable in-cluster address, an **HPA** to scale replicas with load, and a **ConfigMap** carrying model configuration (model name/version, batch size, backend settings) that the container mounts or reads from env. Packaging all of these into one chart means "deploy the model server" is a single versioned operation.

### Why it matters in production

Inference workloads have bursty, unpredictable traffic and expensive compute (GPUs). Bundling the HPA with the Deployment ensures autoscaling policy travels with the app and is versioned alongside it. Externalizing model configuration into a ConfigMap lets you change batch size or swap a model version by editing values and upgrading — no image rebuild — while a checksum annotation forces a rolling restart so the new config actually takes effect. This is the packaging pattern behind serving frameworks like KServe, Seldon, and Triton deployments.

### Real-world example / use case

A vision model is served by Triton reading a `config.pbtxt`-style setting from a ConfigMap. The team needs to bump `max_batch_size` from 8 to 16 for a latency/throughput retune. They change one line in `values-prod.yaml`, run `helm upgrade`, and because the Deployment carries a `checksum/config` annotation tied to the ConfigMap contents, the pods roll automatically to pick up the change — no manual pod deletion, no forgotten restart.

### Code or config example

```yaml
# templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "model-server.fullname" . }}-config
  labels:
    {{- include "model-server.labels" . | nindent 4 }}
data:
  model.json: |
    {
      "modelName": {{ .Values.modelConfig.modelName | quote }},
      "modelVersion": {{ .Values.modelConfig.modelVersion | quote }},
      "maxBatchSize": {{ .Values.modelConfig.maxBatchSize }},
      "timeoutSeconds": {{ .Values.modelConfig.timeoutSeconds }}
    }
```

```yaml
# templates/deployment.yaml (serving-specific bits)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "model-server.fullname" . }}
  labels:
    {{- include "model-server.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "model-server.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        # Roll pods automatically when the ConfigMap changes.
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
      labels:
        {{- include "model-server.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          ports:
            - name: http
              containerPort: 8080
          readinessProbe:
            httpGet:
              path: /v2/health/ready
              port: http
            initialDelaySeconds: 15
          livenessProbe:
            httpGet:
              path: /v2/health/live
              port: http
          volumeMounts:
            - name: model-config
              mountPath: /etc/model
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
      volumes:
        - name: model-config
          configMap:
            name: {{ include "model-server.fullname" . }}-config
```

```yaml
# templates/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "model-server.fullname" . }}
  labels:
    {{- include "model-server.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type | default "ClusterIP" }}
  ports:
    - port: {{ .Values.service.port | default 80 }}
      targetPort: http
      name: http
  selector:
    {{- include "model-server.selectorLabels" . | nindent 4 }}
---
# templates/hpa.yaml
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "model-server.fullname" . }}
  labels:
    {{- include "model-server.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "model-server.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
{{- end }}
```

### Interview questions

**Q: Why include the `checksum/config` annotation on the pod template?**
ConfigMap changes do not, by themselves, restart the pods that consume them — a mounted ConfigMap update is eventually reflected in the volume but the process may not re-read it, and env-injected values never change until restart. Hashing the ConfigMap into a pod-template annotation changes the pod spec whenever the config changes, which triggers a rolling update so the new configuration actually takes effect.

**Q: Why guard `replicas` with `if not .Values.autoscaling.enabled`?**
If both the Deployment sets `replicas` and an HPA manages the same Deployment, they fight: Helm reasserts the static replica count on every upgrade, undoing the HPA's scaling decision and causing thrash. Omitting `replicas` when the HPA is enabled lets the autoscaler own that field.

**Q: How would you package a chart that serves multiple models or backends?**
Options include: parameterizing a single chart with a values list and `range` to generate one Deployment/Service per model; using a library chart for shared templates consumed by per-model charts; or a parent chart with each model as a subchart. The `range`-over-a-list approach is simplest for homogeneous models; subcharts suit heterogeneous stacks with independent lifecycles.

### Pitfalls & best practices

- **Always define readiness and liveness probes** for model servers — cold model loading can take tens of seconds, and without a readiness gate the Service routes traffic to pods that cannot serve yet.
- **Set resource requests/limits explicitly**, especially `nvidia.com/gpu`, so the scheduler places pods correctly and the HPA has a baseline.
- **Externalize model config** to a ConfigMap rather than baking it into the image so you can retune without rebuilds.
- **Do not autoscale on CPU alone for GPU inference** — CPU is often not the bottleneck; prefer custom/queue-depth or latency metrics via KEDA or Prometheus adapter.

---

## 5. Versioning and Releases

### Plain-language explanation

A chart carries **two** versions in `Chart.yaml`. `version` is the chart package's own SemVer — bump it whenever the templates or defaults change. `appVersion` is the version of the *application* (your model server image/model) the chart deploys — it is informational and commonly used as the default image tag. They are independent: you can ship chart `2.4.0` deploying appVersion `1.9.3`, then fix a template bug to release chart `2.4.1` still deploying `1.9.3`.

Charts are distributed via **chart repositories** — either classic HTTP repos backed by an `index.yaml`, or **OCI registries** (the modern default, storing charts as OCI artifacts alongside your container images). Every install/upgrade creates a numbered **revision** in the release's history, and `helm rollback` reverts to any prior revision.

### Why it matters in production

Two-version tracking gives you precise change semantics: reviewers instantly know whether a PR changed the *packaging* (chart version) or the *payload* (appVersion). Release history and rollback are your production safety net — when a bad model or config ships, `helm rollback` restores the last-good revision in seconds without needing the old manifests on hand. Publishing charts to an OCI registry means your deployment artifacts live next to your images with the same auth, signing (cosign), and retention policies.

### Real-world example / use case

A canary model upgrade (appVersion 4.0.0, chart 3.1.0) starts returning garbage predictions in production. On-call runs `helm history rec-prod`, sees revision 27 was the good state (appVersion 3.9.1), and executes `helm rollback rec-prod 27`. Traffic recovers in under a minute while the team investigates offline. The chart itself was fine — only the appVersion changed — which the version scheme made obvious in the postmortem.

### Code or config example

```yaml
# Chart.yaml — the two versions are independent
apiVersion: v2
name: model-server
version: 3.1.0        # chart packaging version (bump on template/values changes)
appVersion: "4.0.0"   # the model server app/image version it deploys
```

```bash
# Package the chart into a versioned .tgz
helm package ./model-server              # -> model-server-3.1.0.tgz

# --- OCI registry workflow (modern default) ---
helm registry login registry.example.com
helm push model-server-3.1.0.tgz oci://registry.example.com/charts
helm install rec-prod oci://registry.example.com/charts/model-server --version 3.1.0

# --- Classic HTTP repo workflow ---
helm repo add myrepo https://charts.example.com
helm repo update
helm search repo myrepo/model-server --versions
helm install rec-prod myrepo/model-server --version 3.1.0

# --- Release lifecycle ---
helm history rec-prod -n ml-serving        # list all revisions
helm rollback rec-prod 27 -n ml-serving    # revert to revision 27
helm get values rec-prod --revision 27     # inspect values used in a revision
helm uninstall rec-prod --keep-history     # remove but retain history
```

### Interview questions

**Q: What is the difference between `version` and `appVersion` in `Chart.yaml`?**
`version` is the SemVer of the chart package itself — bump it on any change to templates, values, or dependencies. `appVersion` is the version of the application the chart installs (your image/model), and is typically used as the default image tag and surfaced in labels. They move independently: a template-only fix bumps `version` but leaves `appVersion` unchanged.

**Q: How does `helm rollback` work and what are its limits?**
Helm keeps a history of revisions (as Secrets) with the full rendered state of each. `helm rollback <release> <revision>` re-applies a prior revision's manifests and creates a *new* revision recording the rollback. Limits: it does not undo external side effects (data migrations, deleted PVCs, external state), and history can be pruned via `--history-max`, so a very old revision may no longer be available.

**Q: What are the advantages of OCI-based chart distribution over classic HTTP repos?**
OCI registries store charts as artifacts alongside container images, so you get one registry, one auth mechanism, unified access control, artifact signing (cosign), and retention/GC policies for both images and charts. There is no separate `index.yaml` to maintain, and it integrates cleanly with existing registry infrastructure (ECR, GAR, Harbor, GHCR).

### Pitfalls & best practices

- **Bump the chart `version` on every change** — repos and OCI registries treat versions as immutable; overwriting an existing version breaks reproducibility.
- **Set `--history-max`** (e.g., 10) to prevent unbounded Secret accumulation while keeping enough revisions for realistic rollbacks.
- **Rollbacks are not magic** — pair them with forward-fix discipline and remember they do not reverse data/schema changes.
- **Sign and scan charts** in the OCI registry just like images.
- Treat `appVersion` and image tags as immutable references (pin digests where possible) rather than `latest`.

---

## 6. Helm + ArgoCD Integration

### Plain-language explanation

ArgoCD is a GitOps continuous-delivery controller: it watches a Git repository as the source of truth and continuously reconciles the cluster to match it. ArgoCD understands Helm natively — you point an ArgoCD `Application` at a chart (in Git or an OCI/HTTP repo) and supply values, and ArgoCD renders the chart (via `helm template` under the hood) and applies the result, then keeps it in sync. Crucially, ArgoCD does *not* run `helm install`; it templates the chart and manages the resulting resources itself, so release state lives in Git and ArgoCD, not in Helm's Secrets.

### Why it matters in production

GitOps gives you a declarative, auditable, self-healing deployment model: every change is a Git commit (reviewed, reversible), drift is automatically detected and corrected, and there is no human running `helm upgrade` from a laptop. For ML platforms managing many model servers across clusters, ArgoCD + Helm means one repo describes the desired state of every model, promotion is a PR, and rollback is a `git revert`. You get Helm's templating power with GitOps guarantees.

### Real-world example / use case

Your platform team manages 40 model servers across three clusters. Each is an ArgoCD `Application` referencing the shared `model-server` chart with a per-model values file in the `deploy/` repo. A data scientist bumps their model to a new image tag via PR; on merge, ArgoCD detects the change, syncs the affected Application, and rolls the Deployment. If someone `kubectl edit`s a live Deployment, ArgoCD flags it as OutOfSync and (with self-heal on) reverts it automatically.

### Code or config example

```yaml
# argocd-application.yaml — chart pulled from an OCI/HTTP repo
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rec-model-prod
  namespace: argocd
spec:
  project: ml-platform
  source:
    repoURL: registry.example.com/charts   # OCI: oci://... ; or an HTTP Helm repo URL
    chart: model-server
    targetRevision: 3.1.0                   # the chart version
    helm:
      releaseName: rec-prod
      valueFiles:
        - values-prod.yaml                  # when sourcing chart from Git
      values: |                             # inline overrides
        image:
          tag: "4.0.0"
        autoscaling:
          enabled: true
          maxReplicas: 30
      parameters:                           # equivalent of --set
        - name: replicaCount
          value: "3"
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-serving
  syncPolicy:
    automated:
      prune: true       # delete resources removed from Git
      selfHeal: true    # revert manual drift
    syncOptions:
      - CreateNamespace=true
```

```yaml
# Multi-source pattern: chart from a Helm repo, values from your Git repo
spec:
  sources:
    - repoURL: https://github.com/org/ml-deploy.git
      targetRevision: main
      ref: valuesRepo
    - repoURL: registry.example.com/charts
      chart: model-server
      targetRevision: 3.1.0
      helm:
        valueFiles:
          - $valuesRepo/prod/rec-values.yaml
```

### Interview questions

**Q: How does ArgoCD handle Helm charts differently from the Helm CLI?**
ArgoCD does not run `helm install`/`upgrade`. It calls `helm template` to render manifests, then applies and reconciles them through its own controller. Consequently there are no Helm release Secrets and no `helm history` — Git is the source of truth and ArgoCD tracks sync state and history. This means `helm test` hooks and some post-render behaviors differ, and you roll back by reverting Git or using ArgoCD's history, not `helm rollback`.

**Q: What do `prune` and `selfHeal` do in an ArgoCD sync policy?**
`prune: true` deletes cluster resources that no longer exist in the desired (Git) state, preventing orphaned objects. `selfHeal: true` automatically reverts any manual, out-of-band changes to bring the cluster back to the Git-declared state. Together they enforce Git as the single source of truth and eliminate configuration drift.

**Q: How do you supply environment-specific values to a Helm chart in ArgoCD?**
Via the Application's `spec.source.helm` block: `valueFiles` (paths to values files, when the chart is in Git), inline `values`, and `parameters` (the `--set` equivalent). For separating chart source from config, use the multi-source pattern with a `$ref` so a values file in your Git repo overrides a chart pulled from an OCI/HTTP registry.

### Pitfalls & best practices

- **Do not mix `helm install` and ArgoCD** on the same release — they will fight over ownership. Pick one controller.
- **Avoid Helm hooks that ArgoCD interprets differently;** ArgoCD maps Helm hooks to its own sync/resource hooks, and lifecycle semantics can surprise you. Prefer ArgoCD-native hooks/waves for ordering.
- **Pin `targetRevision`** to explicit chart versions, not floating ranges, so syncs are deterministic.
- **Roll back via Git revert or ArgoCD history**, not `helm rollback` — the latter is invisible to ArgoCD and gets reverted by self-heal.
- Use the multi-source pattern to keep reusable charts separate from environment values, avoiding chart forks per environment.

---

## Summary

Helm is the packaging and release layer that turns raw Kubernetes YAML into reusable, versioned, promotable applications — indispensable for MLOps where a single model server spans a Deployment, Service, HPA, and ConfigMap. We started with **chart structure** (`Chart.yaml`, `templates/`, `values.yaml`) and the install/upgrade lifecycle, then dug into **Go templating** — `.Values`, built-in objects, Sprig functions, pipelines, named templates in `_helpers.tpl`, and control flow with `range`/`if`. **Values management** layered defaults with per-environment overrides and `--set`, giving explicit, reviewable configuration differences. We packaged a realistic **ML serving chart** with probes, GPU resources, a config ConfigMap, a checksum-driven rolling restart, and an HPA guarded against replica conflicts. **Versioning** clarified the chart `version` vs `appVersion` distinction, OCI/HTTP distribution, and the release history/rollback safety net. Finally, **Helm + ArgoCD** brought GitOps: declarative, self-healing, auditable delivery where Git is the source of truth and Helm provides the templating engine. Master these and you can ship any ML workload to any environment repeatably, safely, and at scale.

## Chapter Q&A

**1. What problem does Helm solve that plain `kubectl apply` does not?**
Helm adds templating (parameterize one chart across environments instead of duplicating YAML), packaging and versioning (distributable, SemVer'd artifacts), release management (named releases with history), and lifecycle operations (atomic upgrades, rollbacks). It converts loose manifests into a repeatable, auditable, promotable application unit.

**2. Explain the Helm 3 architecture and how it improved on Helm 2.**
Helm 3 is client-only: it stores release state as Kubernetes Secrets in the release namespace and uses the caller's kubeconfig RBAC. Helm 2 required Tiller, a cluster-wide privileged server that was a major security liability. Removing Tiller means permissions are scoped to the user and there is no extra attack surface.

**3. Walk through the values override precedence.**
From lowest to highest: chart `values.yaml`, parent chart values (for subcharts), `-f`/`--values` files (later files win), then `--set`/`--set-string`/`--set-json` flags. Inline `--set` always overrides everything, which is why CI uses it for dynamic values like the image tag.

**4. Why is `include` preferred over `template`, and what role do `indent`/`nindent` play?**
`template` writes output directly and cannot be piped; `include` returns a string so it can flow through functions — most importantly `nindent`, which prepends a newline and indents every line to the correct YAML depth. Since named templates emit text at column zero, you almost always need to re-indent, making `include ... | nindent N` the standard idiom.

**5. How do you ensure a pod restarts when its ConfigMap changes?**
Add an annotation on the pod template whose value is a hash of the ConfigMap, e.g. `checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}`. When the ConfigMap content changes, the hash changes, the pod spec changes, and Helm triggers a rolling update so the new config is picked up.

**6. Why should the Deployment omit `replicas` when an HPA is enabled?**
Because both would manage the same field. On every `helm upgrade`, a static `replicas` value would reset the count, fighting the HPA's scaling decisions and causing pod thrash. Guarding it with `{{- if not .Values.autoscaling.enabled }}` lets the autoscaler own replica count.

**7. Distinguish chart `version` from `appVersion` with an example.**
`version` is the chart package's SemVer (bump on template/values changes); `appVersion` is the deployed application's version (often the image tag). Example: chart `2.4.0` deploys appVersion `1.9.3`; a template bug-fix yields chart `2.4.1` still deploying `1.9.3` — packaging changed, payload did not.

**8. How would you roll back a bad model deployment, and what are the caveats?**
Run `helm history <release>` to find the last-good revision, then `helm rollback <release> <revision>`, which re-applies that revision's manifests as a new revision. Caveats: it does not reverse external side effects (data migrations, deleted volumes, external state), and pruned history (via `--history-max`) may make old revisions unavailable. Under ArgoCD, roll back via Git revert instead.

**9. How does ArgoCD deploy a Helm chart, and how does that change rollback and release tracking?**
ArgoCD renders the chart with `helm template` and reconciles the resulting resources itself — it does not create Helm release Secrets. So there is no `helm history`/`helm rollback`; Git is the source of truth and you roll back by reverting a commit or using ArgoCD's history. `prune` and `selfHeal` keep the cluster matching Git and auto-correct drift.

**10. What are the top pitfalls when authoring production Helm charts?**
Uncontrolled whitespace breaking YAML; unquoted version-like strings being coerced to numbers/booleans; changing immutable selector labels; embedding secrets in plaintext; letting both a Deployment and HPA manage `replicas`; forgetting the ConfigMap checksum so config changes don't roll pods; mixing `helm install` with ArgoCD ownership; and overwriting an existing chart version instead of bumping it. Mitigate with `--dry-run --debug`, `values.schema.json` validation, linting, and a secrets manager.
