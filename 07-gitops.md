# Chapter 7 — GitOps (ArgoCD / Flux)

By this point in the book we have containerized models, built CI pipelines, and pushed images to registries. The remaining question is deceptively simple: *how does a version of a model or service actually make it onto a Kubernetes cluster — reliably, repeatably, and auditably — every single time?* The traditional answer was "someone runs `kubectl apply`," or "our Jenkins job runs a deploy script." Both approaches share a fatal flaw: the cluster's real state and the state you *think* you deployed slowly drift apart, and nobody can say with certainty what is running where.

GitOps solves this by making Git the authoritative record of everything that should be running, and by putting an agent inside the cluster whose only job is to make reality match Git. This chapter builds from GitOps first principles through ArgoCD's architecture, sync and rollback mechanics, multi-cluster topologies, and finally the specifics of applying GitOps to machine learning deployments — where you are versioning not just YAML but model artifacts, and promoting those artifacts across dev, staging, and production. We use ArgoCD as the primary reference implementation and contrast Flux where it differs meaningfully.

---

## 1. GitOps Principles

### Plain-language explanation

GitOps is an operating model built on four ideas. First, the entire desired state of your system is described *declaratively* — you write what you want, not the steps to get there. Second, that desired state lives in **Git as the single source of truth**; if it is not in Git, it is not supposed to exist. Third, changes are applied through a **pull-based reconciliation loop**: an agent running inside the cluster continuously pulls from Git and applies changes, rather than an external system pushing into the cluster. Fourth, the system performs **drift detection** — if someone manually edits a live resource, the agent notices the divergence between Git (desired) and cluster (actual) and either flags it or corrects it automatically.

The mental model is a thermostat. You set a target temperature (Git), and the thermostat (the GitOps controller) continuously measures the room and acts to close the gap. You never manually turn the furnace on and off.

### Why it matters in production

Pull-based reconciliation changes your security and reliability posture dramatically. With push-based CI deploys, your CI system needs cluster-admin credentials that live outside the cluster — a juicy attack target and a common breach vector. With GitOps, the controller lives *inside* the cluster and reaches *out* to Git; no external system holds cluster credentials. Git becomes a complete, timestamped, signed audit log of every change: who changed what, when, and why (the commit message and PR). Recovery from a destroyed cluster becomes "point a new controller at the same repo." Drift detection catches the 2 a.m. hotfix somebody applied by hand and never told anyone about.

### Real-world example / use case

A fintech platform runs 40 microservices across three clusters. Before GitOps, an on-call engineer patched a live Deployment during an incident to bump a memory limit. It fixed the incident but was never committed. Three weeks later a routine redeploy reverted the fix, and the incident recurred — this time worse, because nobody remembered the earlier patch. After adopting ArgoCD with self-heal enabled, that same manual patch would have been reverted within seconds and shown as an OutOfSync warning, forcing the fix into Git where it belonged.

### Code or config example

A minimal declarative desired state — a Kustomization that a GitOps controller will reconcile:

```yaml
# gitops-repo/apps/payments/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
  - hpa.yaml
images:
  - name: registry.example.com/payments
    newTag: 1.14.3   # the ONLY line a release PR changes
```

The workflow: a CI pipeline builds image `1.14.3`, opens a PR that changes `newTag`, the PR is reviewed and merged, and the in-cluster controller reconciles the change. No human ever runs `kubectl apply`.

### Interview questions

**Q: What is the difference between push-based and pull-based deployment, and why does GitOps favor pull?**
A: In push-based deployment an external system (CI runner, Jenkins) authenticates to the cluster and applies changes. In pull-based deployment an agent inside the cluster watches a Git repo and applies changes itself. Pull is favored because cluster credentials never leave the cluster (smaller attack surface), the controller can continuously reconcile and self-heal rather than acting only at deploy time, and the same mechanism works identically across many clusters without distributing credentials to CI.

**Q: What is drift, and what are the two ways a GitOps tool can respond to it?**
A: Drift is any divergence between the desired state in Git and the actual state in the cluster — usually caused by manual edits, controllers mutating resources, or failed applies. A GitOps tool can either *detect and report* it (mark the app OutOfSync and alert, leaving remediation to a human) or *detect and self-heal* it (automatically re-apply Git state to overwrite the drift). Self-heal is powerful but must be used carefully with resources that are legitimately mutated at runtime.

**Q: Is Git being the source of truth enough to call something GitOps?**
A: No. Storing YAML in Git is just "config in version control." GitOps additionally requires that the state be declarative, that reconciliation be continuous and automated (typically pull-based from inside the target), and that drift be detected. Running `kubectl apply` from a CI job against files in Git is *not* GitOps — there is no continuous reconciliation and no drift detection.

### Pitfalls & best practices

- **Pitfall:** Treating GitOps as "just store YAML in Git." Without a reconciling agent you get none of the benefits. **Best practice:** Deploy a controller (ArgoCD/Flux) and disable manual `kubectl apply` access for humans in production.
- **Pitfall:** Storing secrets in plain text in the Git repo. **Best practice:** Use Sealed Secrets, SOPS, or an external secrets operator so the repo holds only encrypted or referenced secrets.
- **Best practice:** Separate the *application source repo* (code) from the *config/manifest repo* (deployment state). This prevents CI image builds from triggering redeploy loops and keeps a clean audit boundary.

---

## 2. Declarative vs Imperative Deployment

### Plain-language explanation

An **imperative** approach specifies the sequence of commands: `kubectl create deployment`, then `kubectl scale`, then `kubectl set image`. You describe *how* to reach a state, step by step. A **declarative** approach specifies the desired end state in a manifest — "I want 3 replicas of image 1.14.3" — and lets the system figure out the diff and the steps to converge. Kubernetes itself is fundamentally declarative; GitOps leans all the way into that.

The key distinction: an imperative command is a one-time action with no memory. A declarative manifest is a durable statement of intent that a controller can re-evaluate forever.

### Why it matters in production

Declarative state is idempotent: applying the same manifest ten times produces the same result. Imperative scripts are order-dependent and fragile — re-running a half-failed script can leave you in an unknown state. Declarative manifests are diffable in PRs, so reviewers see exactly what will change. They also enable rollback by simply reverting a commit, because the previous desired state is fully captured. Imperative deploys have no such "previous state" to revert to — you'd have to reconstruct the commands.

### Real-world example / use case

A team migrating from a Bash deploy script (`helm upgrade` wrapped in 200 lines of conditionals) to declarative GitOps found that their script had subtle order dependencies: it applied CRDs and the operator that used them in the wrong order roughly 10% of the time, causing flaky deploys. Moving to declarative manifests with explicit sync ordering (covered in section 4) made deploys deterministic — the controller retries until CRDs are established before applying dependent resources.

### Code or config example

Imperative (avoid in production):

```bash
kubectl create deployment fraud-model --image=registry.example.com/fraud:2.1.0
kubectl scale deployment fraud-model --replicas=3
kubectl expose deployment fraud-model --port=8080
kubectl set env deployment/fraud-model MODEL_THRESHOLD=0.82
```

Declarative equivalent (the GitOps way):

```yaml
# fraud-model/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-model
  labels: { app: fraud-model }
spec:
  replicas: 3
  selector:
    matchLabels: { app: fraud-model }
  template:
    metadata:
      labels: { app: fraud-model }
    spec:
      containers:
        - name: server
          image: registry.example.com/fraud:2.1.0
          ports: [{ containerPort: 8080 }]
          env:
            - name: MODEL_THRESHOLD
              value: "0.82"
          resources:
            requests: { cpu: "500m", memory: "1Gi" }
            limits: { cpu: "2", memory: "2Gi" }
```

### Interview questions

**Q: Why is idempotency important for deployment, and how does declarative config provide it?**
A: Idempotency means applying the same operation repeatedly yields the same state, which is essential for a reconciliation loop that runs continuously and for safe retries after partial failures. Declarative config provides it because you specify the desired end state; the controller computes the diff each time and only acts to close the gap, so re-applying an already-correct manifest is a no-op.

**Q: Can you ever mix imperative and declarative approaches, and what problems arise?**
A: You can, but it causes drift and conflicts. If a controller owns a Deployment declaratively and someone runs `kubectl scale` imperatively, the controller will either revert it (self-heal) or report OutOfSync. The imperative change has no home in Git, so it is invisible to review and audit. The rule is: pick declarative as the source of truth and forbid imperative mutations in managed namespaces.

### Pitfalls & best practices

- **Pitfall:** Using `kubectl edit` on a GitOps-managed resource "just this once." It creates invisible drift. **Best practice:** Make every change through a PR.
- **Best practice:** Prefer server-side apply and let controllers own the fields they manage; use `ignoreDifferences` for fields legitimately mutated by other controllers (e.g., HPA-managed `replicas`).

---

## 3. ArgoCD Architecture

### Plain-language explanation

ArgoCD is a Kubernetes controller that continuously compares the desired state in Git against the live state in one or more clusters, and reconciles the difference. Its core abstraction is the **Application** custom resource (CRD): a declarative object that says "take the manifests from *this* Git repo at *this* path/revision, and deploy them to *this* cluster and namespace." ArgoCD is made of a few cooperating components:

- **repo-server:** Clones Git repos and renders manifests. It runs your Helm, Kustomize, or Jsonnet tooling to produce plain Kubernetes YAML. It is stateless and CPU-heavy during rendering.
- **application-controller:** The reconciliation engine. It compares rendered desired state against the live cluster, computes sync status (Synced / OutOfSync) and health status, and applies changes.
- **API server:** Serves the gRPC/REST API, the web UI, and the CLI. Handles auth (SSO/RBAC), and exposes the Application objects.
- **Redis:** A cache for rendered manifests and computed state so the controller does not re-render on every loop.
- **Dex (optional):** An OIDC provider for SSO integration.

The **App of Apps** pattern is a bootstrapping technique: you create one parent Application whose Git path contains *other* Application manifests. ArgoCD deploys the parent, which in turn creates all the child Applications — letting you manage dozens of apps from a single root and bootstrap an entire cluster from one manifest.

**Flux contrast:** Flux (v2) takes a more modular, controller-per-concern approach. Instead of one Application CRD, it uses a set of controllers and CRDs: `GitRepository` / `OCIRepository` (source), `Kustomization` and `HelmRelease` (apply), `ImageRepository` / `ImagePolicy` (image automation), and a notification controller. Flux has no built-in UI (it is CLI/CRD-first, though the Weave GitOps UI exists), leans on Kubernetes RBAC directly, and its image-update automation can commit new image tags back to Git out of the box. ArgoCD ships a rich UI and a first-class Application/ApplicationSet model; Flux is more composable and Kubernetes-native. Both are CNCF graduated projects and both are excellent — the choice is often about whether you want ArgoCD's UI-driven, application-centric model or Flux's toolkit-of-controllers model.

### Why it matters in production

Understanding the components tells you how to scale and debug ArgoCD. If manifest rendering is slow, you scale repo-server replicas. If reconciliation lags across thousands of apps, you shard the application-controller. Knowing that repo-server executes your Helm/Kustomize means you understand where a templating error surfaces. The App of Apps (or its successor, ApplicationSet) is what makes ArgoCD manageable at scale — without it you would hand-create hundreds of Application objects.

### Real-world example / use case

A platform team manages 200 microservices across 5 clusters. They keep a single "root" repo. The App-of-Apps root Application points at `clusters/prod/apps/`, a directory holding one small Application manifest per service. Onboarding a new service is a single PR adding one file. ArgoCD's root app sees the new file, creates the child Application, and the service deploys — no ticket to the platform team, no manual ArgoCD UI clicks.

### Code or config example

A realistic ArgoCD Application manifest:

```yaml
# apps/fraud-model.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: fraud-model
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io   # cascade-delete children on app delete
spec:
  project: ml-platform
  source:
    repoURL: https://github.com/acme/ml-manifests.git
    targetRevision: main
    path: fraud-model/overlays/prod
    # Kustomize is auto-detected from kustomization.yaml at this path
  destination:
    server: https://kubernetes.default.svc
    namespace: fraud
  syncPolicy:
    automated:
      prune: true       # delete resources removed from Git
      selfHeal: true    # revert manual drift
    syncOptions:
      - CreateNamespace=true
      - ApplyOutOfSyncOnly=true
    retry:
      limit: 5
      backoff:
        duration: 10s
        factor: 2
        maxDuration: 3m
```

The App-of-Apps root that manages many such Applications:

```yaml
# root-app.yaml — deploy this ONE object to bootstrap everything
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: root
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/acme/ml-manifests.git
    targetRevision: main
    path: clusters/prod/apps   # a directory full of Application YAMLs
    directory:
      recurse: true
  destination:
    server: https://kubernetes.default.svc
    namespace: argocd
  syncPolicy:
    automated: { prune: true, selfHeal: true }
```

For teams generating many similar apps, `ApplicationSet` is the modern, templated alternative to App-of-Apps:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: ml-models
  namespace: argocd
spec:
  generators:
    - git:                       # generate one App per directory under models/
        repoURL: https://github.com/acme/ml-manifests.git
        revision: main
        directories:
          - path: models/*
  template:
    metadata:
      name: '{{path.basename}}'
    spec:
      project: ml-platform
      source:
        repoURL: https://github.com/acme/ml-manifests.git
        targetRevision: main
        path: '{{path}}'
      destination:
        server: https://kubernetes.default.svc
        namespace: '{{path.basename}}'
      syncPolicy:
        automated: { prune: true, selfHeal: true }
```

A Flux equivalent for one app, for contrast:

```yaml
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: ml-manifests
  namespace: flux-system
spec:
  interval: 1m
  url: https://github.com/acme/ml-manifests.git
  ref: { branch: main }
---
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: fraud-model
  namespace: flux-system
spec:
  interval: 5m
  path: ./fraud-model/overlays/prod
  prune: true
  sourceRef:
    kind: GitRepository
    name: ml-manifests
  targetNamespace: fraud
```

### Interview questions

**Q: Walk me through what happens inside ArgoCD from a merged PR to a running pod.**
A: The application-controller polls (or receives a webhook) and asks repo-server to fetch the new Git revision. Repo-server clones the repo and renders manifests (running Helm/Kustomize as needed), returning plain YAML, which is cached in Redis. The controller diffs the rendered desired state against the live cluster state, marking the app OutOfSync. If sync is automated, it applies the diff (respecting sync waves and hooks), then re-evaluates health until resources are Healthy and Synced.

**Q: What problem does the App-of-Apps pattern solve, and how does ApplicationSet improve on it?**
A: App-of-Apps lets you manage many Applications declaratively from one parent, so bootstrapping and onboarding are single PRs rather than manual object creation. Its limitation is that each child is a hand-written manifest. ApplicationSet improves on it with generators (Git directories, cluster lists, pull requests, matrices) that *template* Applications programmatically, so you can, say, fan one template across every cluster or every model directory without writing each Application by hand.

**Q: How would you scale ArgoCD for thousands of applications and many clusters?**
A: Scale repo-server horizontally for rendering throughput, enable and tune the manifest cache, and shard the application-controller across replicas (each replica owns a subset of clusters). Use ApplicationSets to manage the fleet, put sensible reconciliation timeouts and resource limits in place, and consider multiple ArgoCD instances per environment or a hub-and-spoke topology if a single instance becomes a bottleneck.

### Pitfalls & best practices

- **Pitfall:** Forgetting the `resources-finalizer` so deleting an Application orphans its child resources. **Best practice:** Add the finalizer when you want cascade deletion.
- **Pitfall:** Putting Application manifests and the app's own workload manifests in the same path, causing the controller to try to manage its own Applications recursively in confusing ways. **Best practice:** Keep the App-of-Apps directory separate from workload manifests.
- **Best practice:** Use AppProjects to scope which repos, clusters, and namespaces a set of apps may touch — critical multi-tenant guardrails.

---

## 4. Sync and Rollback Strategies

### Plain-language explanation

A **sync** is the act of making the cluster match Git. ArgoCD gives you fine control over *how* and *when* that happens:

- **Automated vs manual sync:** Automated means the controller applies changes as soon as it detects drift or a new Git revision. Manual means a human (or a pipeline) must click "Sync" or run `argocd app sync`.
- **Self-heal:** When enabled, automated sync also reverts manual changes to live resources, not just Git changes.
- **Prune:** When enabled, resources deleted from Git are deleted from the cluster.
- **Sync waves:** An ordering mechanism. Resources are annotated with a wave number; ArgoCD applies lower waves first and waits for them to be healthy before proceeding. This handles dependencies (e.g., namespaces and CRDs before workloads).
- **Sync hooks:** Jobs that run at specific phases — `PreSync`, `Sync`, `PostSync`, `SyncFail` — for tasks like database migrations, smoke tests, or notifications.
- **Rollback:** Because Git holds every prior state, rolling back is reverting a commit (the GitOps-pure way) or using `argocd app rollback` to a previous synced revision (fast, but creates drift from Git until you also revert Git).

### Why it matters in production

These controls are the difference between a safe deploy and an outage. Sync waves prevent the classic "workload starts before its CRD/config exists" race. PreSync hooks run DB migrations before the new code that depends on them. Manual sync gates give humans a checkpoint for high-risk production changes. Self-heal enforces that production reality can never quietly diverge from Git. And a reliable rollback path — revert the commit, let the controller reconcile — turns a scary incident into a routine, auditable action.

### Real-world example / use case

An ML API required a schema migration to its feature store before the new model server could start. Using a `PreSync` hook Job that ran the migration, and sync waves that placed the ConfigMap (wave 0), migration (PreSync), and Deployment (wave 1) in order, the team eliminated a class of "deployed new server against old schema" failures. When a bad model shipped, they reverted the single tag-bump commit; ArgoCD detected OutOfSync and self-healed the cluster back to the good version within a minute.

### Code or config example

Sync waves and a PreSync migration hook:

```yaml
# migration Job runs BEFORE the main sync, in the PreSync phase
apiVersion: batch/v1
kind: Job
metadata:
  name: feature-store-migrate
  annotations:
    argocd.argoproj.io/hook: PreSync
    argocd.argoproj.io/hook-delete-policy: HookSucceeded
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: migrate
          image: registry.example.com/feature-migrate:1.4.0
          command: ["python", "migrate.py", "--up"]
---
# ConfigMap applied first
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  annotations:
    argocd.argoproj.io/sync-wave: "0"
data:
  threshold: "0.82"
---
# Deployment applied after wave 0 is healthy
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-model
  annotations:
    argocd.argoproj.io/sync-wave: "1"
spec:
  # ... as before
```

Rollback commands:

```bash
# GitOps-pure rollback: revert the commit, let ArgoCD reconcile
git revert <bad-commit-sha> && git push

# Fast operational rollback via ArgoCD (creates drift until Git is reverted too)
argocd app history fraud-model
argocd app rollback fraud-model 42   # roll back to history ID 42

# Ignore HPA-managed replicas so self-heal doesn't fight the autoscaler
# (in the Application spec)
# spec.ignoreDifferences:
#   - group: apps
#     kind: Deployment
#     jsonPointers: ["/spec/replicas"]
```

### Interview questions

**Q: When would you use manual sync instead of automated sync?**
A: For high-risk or regulated production changes where you want an explicit human or pipeline gate — for example, a promotion into production that requires a change-approval sign-off, or a change with a risky migration. Manual sync lets you review the exact diff in the UI/CLI, coordinate timing (maintenance windows), and integrate a deployment approval step. Dev and staging typically use automated sync for speed.

**Q: Explain sync waves and hooks, and give a case where you need both.**
A: Sync waves order resource application within a sync (lower waves first, waiting for health), handling dependencies like CRDs before CRs. Hooks run Jobs at phases around the sync (PreSync, Sync, PostSync, SyncFail) for imperative side-effects. You need both when a deploy requires a DB migration before new code (PreSync hook) *and* internal ordering among the app's own resources (waves) — e.g., migrate schema (PreSync), then apply config (wave 0), then the server (wave 1), then run smoke tests (PostSync hook).

**Q: Compare a Git-revert rollback with `argocd app rollback`. Which is "more GitOps"?**
A: Reverting the commit is the GitOps-pure approach: Git remains the source of truth, the change is audited, and the controller reconciles normally. `argocd app rollback` is a fast operational escape hatch that syncs to a previous revision *without changing Git*, which creates drift — the cluster now differs from `main`, and self-heal or the next sync will try to reapply the bad version unless you also revert Git. Use `argocd app rollback` for emergency speed, but follow up immediately with a Git revert to restore truth.

### Pitfalls & best practices

- **Pitfall:** Enabling self-heal on a Deployment whose `replicas` are managed by an HPA — ArgoCD and the HPA fight, flapping the replica count. **Best practice:** Use `ignoreDifferences` on HPA-managed fields.
- **Pitfall:** Enabling `prune` carelessly and deleting a resource that was only *temporarily* removed from Git, causing an outage. **Best practice:** Review prune diffs; consider `PruneLast=true` and prune propagation policies.
- **Best practice:** Make hooks idempotent (especially migrations) and set `hook-delete-policy` so completed hook Jobs are cleaned up.

---

## 5. Multi-Cluster GitOps

### Plain-language explanation

Multi-cluster GitOps means managing many clusters — regions, environments, or tenants — from a coherent GitOps setup. There are two dominant topologies. In the **hub-and-spoke** (centralized) model, one ArgoCD "hub" instance holds credentials to many remote clusters and reconciles into all of them; the Application's `destination.server` points at a registered remote cluster. In the **standalone** (decentralized) model, each cluster runs its own ArgoCD instance reconciling into itself only, which improves blast-radius isolation and per-cluster autonomy at the cost of more instances to operate.

ArgoCD registers remote clusters as secrets (via `argocd cluster add`), and **ApplicationSet cluster generators** let you fan a single app template across every registered cluster automatically. Flux typically favors the standalone model (an instance per cluster), sometimes with a management cluster committing config that spoke clusters pull.

### Why it matters in production

Real platforms are rarely one cluster. You have prod-us-east, prod-eu-west, staging, and dev; you may isolate tenants per cluster for compliance. Doing this without automation means copy-pasting Applications per cluster and drifting configurations. ApplicationSet generators make "deploy this to all production clusters" a single declarative statement, so a new region cluster gets the full app fleet automatically upon registration. Choosing the right topology also bounds your blast radius: a compromised or broken hub can affect every spoke, so security-sensitive setups often prefer standalone instances.

### Real-world example / use case

A company runs identical inference stacks in three regions for latency and residency. They register all three clusters with a hub ArgoCD and use an ApplicationSet with a cluster generator. Rolling out a new model version to all regions is one PR to the shared overlay. For a canary region rollout, they use the generator's per-cluster values to pin `prod-eu-west` to the new version first, validate, then promote the rest — all declaratively.

### Code or config example

ApplicationSet fanning one app across all registered clusters, with per-cluster overrides:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: inference-all-clusters
  namespace: argocd
spec:
  generators:
    - clusters:
        selector:
          matchLabels:
            env: production      # labels set when registering the cluster
  template:
    metadata:
      name: 'inference-{{name}}'   # {{name}} = cluster name from generator
    spec:
      project: ml-platform
      source:
        repoURL: https://github.com/acme/ml-manifests.git
        targetRevision: main
        path: inference/overlays/{{metadata.labels.region}}
      destination:
        server: '{{server}}'       # remote cluster API URL from generator
        namespace: inference
      syncPolicy:
        automated: { prune: true, selfHeal: true }
```

Registering a remote cluster with the hub:

```bash
# context must exist in your kubeconfig
argocd cluster add prod-eu-west --label env=production --label region=eu-west
argocd cluster list
```

### Interview questions

**Q: Compare hub-and-spoke and standalone multi-cluster ArgoCD. When would you pick each?**
A: Hub-and-spoke centralizes one ArgoCD managing many remote clusters — simpler single pane of glass and easy fleet-wide rollouts via ApplicationSet, but the hub holds credentials to every cluster (larger blast radius and a high-value target). Standalone runs one ArgoCD per cluster reconciling only itself — stronger isolation, no cross-cluster credentials, better tenant/compliance boundaries, but more instances to operate and no single UI. Pick hub-and-spoke for operational simplicity at moderate scale; pick standalone when isolation, compliance, or blast-radius containment dominate.

**Q: How do you roll out a change to 20 clusters without editing 20 Application manifests?**
A: Use an ApplicationSet with a cluster generator that selects clusters by label and a single templated Application. One PR to the shared source path reconciles to all matched clusters. For staged rollouts you use per-cluster parameters or progressive syncs to canary a subset before the rest.

### Pitfalls & best practices

- **Pitfall:** Hardcoding cluster-specific values in a shared base, causing wrong config in some clusters. **Best practice:** Use Kustomize overlays or Helm values per region/cluster, referenced via generator parameters.
- **Pitfall:** A single hub as an unmonitored single point of failure. **Best practice:** Make the hub HA (multiple replicas, sharded controller), back up its config, and monitor reconciliation lag.
- **Best practice:** Label clusters richly at registration (env, region, tier) so generators can select precisely.

---

## 6. Applying GitOps to ML Model Deployments

### Plain-language explanation

Applying GitOps to ML adds a wrinkle: a deployment is not just code and YAML — it references a **model artifact** (weights) that may be hundreds of megabytes to gigabytes, too large to store in Git. The GitOps pattern is to keep manifests in Git and reference the model artifact *immutably by version* — a model registry entry (MLflow, SageMaker), an object-store URI with a content hash, or an OCI artifact tag. Git holds the *pointer* and the *deployment config*; the artifact store holds the *bytes*. Promoting a model across environments (dev → staging → prod) then becomes a Git operation: update the environment overlay to reference the newer, validated model version, via a reviewed PR.

Two common patterns for delivering the artifact to the pod: (1) the container image *bakes in* the model (image tag is the version of record — simplest, immutable, but rebuilds per model); or (2) the container pulls the model at startup from a registry/object store using a version pinned in a ConfigMap (decouples model version from image, enables the same server image to serve many model versions). Either way, GitOps requires the reference to be **immutable and explicit** — never "latest."

### Why it matters in production

Reproducibility and auditability are non-negotiable in ML: you must be able to say exactly which model, trained on which data, with which config, served which prediction — often for regulatory or debugging reasons. GitOps gives you that: the Git history plus the pinned artifact version is a complete, timestamped record of what was serving when. Promotion-by-PR gives you a governance gate (a human approves prod promotion after staging validation passes). And drift detection ensures nobody hot-swaps a model in production without it being recorded in Git.

### Real-world example / use case

A recommender team stores trained models in MLflow, tagged by run ID. CI promotes a model to the "Staging" alias after offline eval passes. A GitOps PR updates `staging/overlays/kustomization.yaml` to set `MODEL_URI` to the new run's artifact URI; ArgoCD deploys it to the staging cluster; automated integration tests (a PostSync hook) run. When they pass, a second PR copies that exact `MODEL_URI` into `prod/overlays/` — promoting *the identical artifact* rather than rebuilding — and a manual sync gate lets the ML lead approve the prod rollout. The prod Application uses a canary rollout (via Argo Rollouts) to shift 10% of traffic first.

### Code or config example

A model-serving Deployment that pulls a pinned model version at startup, with the version living in Git:

```yaml
# prod/overlays/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
configMapGenerator:
  - name: model-ref
    literals:
      # immutable, explicit — the ONLY thing a promotion PR changes
      - MODEL_URI=s3://acme-models/fraud/2026-07-10T14-22-03Z-run-9f3a/model
      - MODEL_VERSION=fraud-v7
images:
  - name: registry.example.com/fraud-server
    newTag: 3.2.0   # the SERVER image, versioned independently of the model
```

```yaml
# base/deployment.yaml (excerpt)
spec:
  template:
    spec:
      initContainers:
        - name: fetch-model
          image: amazon/aws-cli:2
          envFrom: [{ configMapRef: { name: model-ref } }]
          command: ["sh", "-c", "aws s3 cp --recursive $MODEL_URI /models/current"]
          volumeMounts: [{ name: model, mountPath: /models }]
      containers:
        - name: server
          image: registry.example.com/fraud-server:3.2.0
          envFrom: [{ configMapRef: { name: model-ref } }]
          env:
            - name: MODEL_PATH
              value: /models/current
          volumeMounts: [{ name: model, mountPath: /models, readOnly: true }]
      volumes:
        - name: model
          emptyDir: {}
```

A Flux `HelmRelease` promoting a model via a pinned chart value, for contrast:

```yaml
apiVersion: helm.toolkit.fluxcd.io/v2
kind: HelmRelease
metadata:
  name: fraud-model
  namespace: fraud
spec:
  interval: 5m
  chart:
    spec:
      chart: ./charts/model-server
      sourceRef: { kind: GitRepository, name: ml-manifests }
  values:
    image:
      repository: registry.example.com/fraud-server
      tag: "3.2.0"
    model:
      uri: s3://acme-models/fraud/2026-07-10T14-22-03Z-run-9f3a/model
      version: fraud-v7
```

An ApplicationSet that models the promotion pipeline as three environment apps from three overlays:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: fraud-model-envs
  namespace: argocd
spec:
  generators:
    - list:
        elements:
          - { env: dev,     autoSync: "true"  }
          - { env: staging, autoSync: "true"  }
          - { env: prod,    autoSync: "false" }   # manual gate in prod
  template:
    metadata:
      name: 'fraud-model-{{env}}'
    spec:
      project: ml-platform
      source:
        repoURL: https://github.com/acme/ml-manifests.git
        targetRevision: main
        path: 'fraud-model/overlays/{{env}}'
      destination:
        server: https://kubernetes.default.svc
        namespace: 'fraud-{{env}}'
      # note: autoSync gating typically handled via project/policy or templated syncPolicy
```

### Interview questions

**Q: Model files are too big for Git. How do you keep ML deployments GitOps-compliant?**
A: Store the artifact in a purpose-built store (model registry, object storage, or OCI artifact) and keep only an *immutable, explicit* reference in Git — a run ID, versioned URI, or digest. Git holds the deployment config and the pointer; the store holds the bytes. The pod fetches the pinned version at startup (init container) or the model is baked into a versioned image. The rule is no `latest`: the reference must uniquely identify one artifact so the deploy is reproducible and auditable.

**Q: Describe how you'd promote a model from staging to production with GitOps.**
A: After the model passes offline eval and CI promotes it in the registry, a PR updates the staging overlay's model reference; ArgoCD deploys it and PostSync integration tests run. Once staging validation passes, a second PR copies the *exact same* artifact reference into the prod overlay — promoting the identical bytes, not rebuilding. Prod uses a manual sync gate for approval and typically a progressive/canary rollout. Every step is a reviewed, audited Git change, so you always know which model is in prod and why.

**Q: Should the model version and the server image version be coupled or decoupled? Trade-offs?**
A: Baking the model into the image *couples* them: maximally immutable and simple to reason about (one tag = one exact deployment), but every new model requires an image rebuild and the images are large. Decoupling (server image pulls a pinned model URI) lets one server image serve many model versions and makes model promotion a config-only change, at the cost of a runtime fetch dependency and needing to track two versions. Many teams decouple for iteration speed but keep both references explicit and pinned in Git so the *combination* is still fully reproducible.

### Pitfalls & best practices

- **Pitfall:** Referencing a mutable tag or `latest` model alias in Git — you lose reproducibility and drift silently as the alias moves. **Best practice:** Pin immutable versions/digests; let aliases resolve to a concrete version in CI, then commit the concrete version.
- **Pitfall:** Rebuilding a new artifact when promoting to prod, so staging and prod run subtly different bytes. **Best practice:** Promote the *same* artifact reference through environments; never rebuild between staging and prod.
- **Pitfall:** Large init-container downloads slowing pod startup and hammering the object store on every scale-up. **Best practice:** Cache models on nodes (PVC or hostPath cache), use image-baked models for hot paths, or a model-caching sidecar.
- **Best practice:** Combine GitOps with progressive delivery (Argo Rollouts / Flagger) for canary and automated rollback of models based on live metrics.

---

## Summary

GitOps reframes deployment around a single, auditable source of truth — Git — and a controller inside the cluster that continuously reconciles reality toward that truth. Its four principles (declarative state, Git as source of truth, pull-based reconciliation, and drift detection) deliver security (no external cluster credentials), reproducibility, and effortless rollback (revert a commit). Declarative configuration is what makes this possible: idempotent, diffable, and reversible, unlike fragile imperative scripts.

ArgoCD implements GitOps through the Application CRD and a small set of components — repo-server (renders manifests), application-controller (reconciles), API server/UI, and Redis — with App-of-Apps and ApplicationSet patterns for managing fleets at scale. Flux offers the same principles through a composable set of controllers and CRDs. Sync strategies (automated vs manual, self-heal, prune, sync waves, and hooks) give precise control over how and when changes land, while rollback is as simple as reverting Git. Multi-cluster GitOps, via hub-and-spoke or standalone topologies and ApplicationSet generators, extends this to fleets of clusters across regions and tenants.

For ML specifically, GitOps requires keeping oversized model artifacts out of Git while pinning immutable references to them — enabling reproducible, auditable deployments and promotion-by-PR across environments. Promote the *same* artifact through dev, staging, and prod; gate prod with manual sync and canary rollouts; and never reference `latest`. Done well, GitOps turns "what model is running in prod, and how did it get there?" from an anxious guess into a one-line answer backed by Git history.

## Chapter Q&A

**Q1: Define GitOps in one sentence and name its non-negotiable requirements.**
A: GitOps is an operating model where the entire desired system state is stored declaratively in Git and an in-cluster agent continuously reconciles the cluster to match it. Non-negotiables: declarative state, Git as single source of truth, continuous (typically pull-based) automated reconciliation, and drift detection.

**Q2: Why is pull-based reconciliation more secure than push-based CI deploys?**
A: The controller runs inside the cluster and reaches out to Git, so cluster credentials never leave the cluster and no external CI system needs cluster-admin access — shrinking the attack surface and eliminating a common credential-leak vector. It also enables continuous self-healing rather than one-shot deploys.

**Q3: What are the core ArgoCD components and their jobs?**
A: repo-server clones repos and renders Helm/Kustomize/Jsonnet into plain YAML; application-controller diffs desired vs live state and applies changes, computing sync and health status; the API server serves the UI/CLI/API and handles auth/RBAC; Redis caches rendered manifests; Dex optionally provides SSO.

**Q4: Contrast ArgoCD and Flux.**
A: ArgoCD is application-centric with one Application CRD, a rich UI, ApplicationSets, and its own RBAC. Flux is a composable toolkit of controllers and CRDs (GitRepository, Kustomization, HelmRelease, image automation, notifications), is CLI/CRD-first with no built-in UI, uses Kubernetes RBAC directly, and includes image-update-to-Git automation out of the box. Both are CNCF graduated and implement the same principles.

**Q5: Explain sync waves versus sync hooks.**
A: Sync waves order the application of resources *within* a sync (lower waves first, waiting for health) to satisfy dependencies like CRDs before CRs. Hooks are Jobs run at phases *around* the sync — PreSync, Sync, PostSync, SyncFail — for imperative side-effects like migrations, smoke tests, or notifications.

**Q6: How do self-heal and prune work, and what is a common gotcha for each?**
A: Self-heal reverts manual changes to live resources back to Git; the gotcha is fighting controllers that legitimately mutate resources (e.g., HPA on replicas) — fix with `ignoreDifferences`. Prune deletes resources removed from Git; the gotcha is deleting something removed only temporarily, causing an outage — mitigate by reviewing prune diffs and using cautious prune policies.

**Q7: What is the difference between an App-of-Apps and an ApplicationSet?**
A: App-of-Apps is a parent Application whose Git path contains hand-written child Application manifests — good for bootstrapping. ApplicationSet templates Applications programmatically via generators (Git directories, cluster lists, PRs, matrices), so you can fan a single template across many clusters or directories without writing each Application by hand.

**Q8: How do you roll back safely in GitOps, and what's the pitfall of `argocd app rollback`?**
A: The pure approach is `git revert` of the bad commit and letting the controller reconcile — fully audited, Git stays authoritative. `argocd app rollback` syncs to a prior revision fast but *without changing Git*, creating drift; self-heal or the next sync may reapply the bad version. Use it for emergency speed, then immediately revert Git to restore truth.

**Q9: How do you make ML model deployments GitOps-compliant given large artifacts?**
A: Keep manifests and an immutable model reference in Git (run ID, versioned object-store URI, or digest) while storing the bytes in a registry/object store. Fetch the pinned model at startup or bake it into a versioned image. Never use `latest`; the reference must uniquely identify one artifact for reproducibility and audit.

**Q10: Describe an end-to-end model promotion pipeline with GitOps and progressive delivery.**
A: CI trains and evaluates the model, registers it, and opens a PR pinning the new artifact reference in the staging overlay. ArgoCD deploys to staging and PostSync integration tests run. On success, a second PR copies the *identical* reference into the prod overlay; a manual sync gate provides approval, and Argo Rollouts (or Flagger) performs a canary — shifting traffic gradually while watching live metrics and auto-rolling-back on regression. Every step is an audited Git change, so the running model and its provenance are always known.
