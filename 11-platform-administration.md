# Chapter 11 — Platform Administration

A machine learning platform is rarely used by a single team. Once a platform proves useful, it becomes a shared resource: data scientists train models, engineers serve them, analysts run batch inference, and multiple business units all draw from the same pool of expensive GPUs and CPU nodes. The moment a platform becomes multi-tenant, a new discipline emerges — platform administration. This is the operational glue that keeps a shared system fair, secure, cost-controlled, and reliable.

This chapter walks from the fundamentals (who is allowed to do what) up to the advanced (how you keep a shared ML platform reliable under load and respond when it breaks at 3 a.m.). Kubernetes is the assumed substrate because it is the de facto control plane for MLOps, but the principles — isolation, quotas, cost accountability, service-level objectives, and disciplined incident response — apply to any managed platform. Treat this as both a working reference and interview preparation for platform, SRE, and MLOps engineering roles.

---

## 1. RBAC and Multi-Tenancy

### Plain-language explanation

Role-Based Access Control (RBAC) answers a single question: *who is allowed to do what, and where?* In Kubernetes, RBAC is expressed through four core objects. A **Role** is a set of permissions (verbs like `get`, `list`, `create`) scoped to a single namespace. A **ClusterRole** is the same idea but cluster-wide or applicable to cluster-scoped resources (nodes, persistent volumes). A **RoleBinding** grants a Role (or ClusterRole) to a subject — a user, group, or **ServiceAccount** — within a namespace. A **ClusterRoleBinding** grants a ClusterRole across the whole cluster.

A **ServiceAccount** is an identity for workloads (pods) rather than humans. When a training pod needs to read from an S3-backed secret or a serving pod needs to talk to the Kubernetes API, it authenticates as its ServiceAccount.

Multi-tenancy is the practice of hosting multiple teams ("tenants") on one cluster while keeping them logically — and sometimes physically — separated. The common isolation models are: **namespace-per-tenant** (soft multi-tenancy, cheapest), **cluster-per-tenant** (hard isolation, most expensive), and **virtual clusters** (e.g., vCluster) that sit in between.

### Why it matters in production

Without RBAC, any user with cluster access can delete another team's production model server, read another tenant's secrets (which often hold cloud credentials and API keys), or exhaust shared resources. In regulated environments (finance, healthcare) least-privilege access is a compliance requirement, not a nicety. RBAC is also the foundation on which quotas, network policies, and cost attribution are built — every one of those relies on a stable tenant boundary, and in Kubernetes that boundary is almost always the namespace.

### Real-world example / use case

A fintech company runs a shared ML platform with three tenants: `fraud-team`, `risk-team`, and `marketing-team`. Each tenant gets its own namespace. Data scientists in `fraud-team` should be able to launch training jobs and inspect logs in their own namespace but must never touch `risk-team`. Meanwhile, the central platform SRE team needs cluster-wide read access for monitoring plus the ability to manage nodes. This maps cleanly to: namespaced Roles for tenant members, a ClusterRole for the SRE group, and a tightly scoped ServiceAccount for each team's CI/CD pipeline.

### Code or config example

```yaml
# A namespaced Role: data scientists can manage training jobs and read logs,
# but cannot touch secrets or delete deployments.
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: fraud-team
  name: ds-training-role
rules:
  - apiGroups: ["batch", "kubeflow.org"]
    resources: ["jobs", "pytorchjobs", "tfjobs"]
    verbs: ["create", "get", "list", "watch", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list"]
---
# Bind the Role to a group provided by the OIDC identity provider.
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  namespace: fraud-team
  name: ds-training-binding
subjects:
  - kind: Group
    name: "oidc:fraud-data-scientists"   # comes from your SSO / OIDC claims
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: ds-training-role
  apiGroup: rbac.authorization.k8s.io
---
# A dedicated ServiceAccount for the tenant's CI pipeline.
apiVersion: v1
kind: ServiceAccount
metadata:
  namespace: fraud-team
  name: fraud-ci-deployer
automountServiceAccountToken: false   # opt in only where needed
---
# Give that ServiceAccount just enough to roll out model servers.
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: fraud-team
  name: ci-deployer-role
rules:
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  namespace: fraud-team
  name: ci-deployer-binding
subjects:
  - kind: ServiceAccount
    name: fraud-ci-deployer
    namespace: fraud-team
roleRef:
  kind: Role
  name: ci-deployer-role
  apiGroup: rbac.authorization.k8s.io
```

```bash
# Verify permissions the right way: impersonation with `can-i`.
kubectl auth can-i delete deployments \
  --as=system:serviceaccount:fraud-team:fraud-ci-deployer \
  -n fraud-team
# -> yes

kubectl auth can-i get secrets \
  --as=system:serviceaccount:fraud-team:fraud-ci-deployer \
  -n risk-team
# -> no
```

### Interview questions

**Q: What is the difference between a Role and a ClusterRole, and when would you bind a ClusterRole with a RoleBinding?**
A: A Role is namespace-scoped; a ClusterRole is cluster-scoped and can also grant access to cluster-scoped resources (nodes, PVs, namespaces themselves). You bind a ClusterRole with a RoleBinding when you want to reuse a common permission set (say, a "viewer" role) but limit its effect to a single namespace. The RoleBinding constrains the ClusterRole's scope to that namespace — a very common pattern to avoid duplicating the same Role in every tenant namespace.

**Q: How do you achieve strong tenant isolation on Kubernetes, and what are the trade-offs?**
A: Namespace-per-tenant is the cheapest but is "soft" — tenants share the API server, the kernel, and the node pool, so a container escape or a noisy neighbor can cross boundaries. Cluster-per-tenant gives hard isolation (separate control planes, blast radius contained) at high operational and cost overhead. Virtual clusters (vCluster) give each tenant a synthetic API server and CRD space while sharing worker nodes — a middle ground. The choice depends on your threat model: regulated or untrusted workloads push you toward hard isolation; internal trusted teams are usually fine with namespaces plus network policies and quotas.

**Q: Why is `automountServiceAccountToken: false` a good default?**
A: By default every pod gets the namespace's ServiceAccount token mounted, which is an unnecessary credential a compromised container could use to call the API server. Disabling automount and opting in only for workloads that genuinely need API access follows least privilege and shrinks the attack surface.

### Pitfalls & best practices

- **Avoid `cluster-admin` sprawl.** Granting `cluster-admin` "temporarily" is the most common RBAC failure. Audit ClusterRoleBindings regularly with `kubectl get clusterrolebindings -o wide`.
- **Never use wildcard verbs/resources** (`verbs: ["*"]`, `resources: ["*"]`) in tenant roles. Enumerate explicitly.
- **Bind to groups, not individual users.** Manage membership in your IdP (Okta, Entra, Google) so onboarding/offboarding never requires cluster changes.
- **Separate human and machine identities.** Users authenticate via OIDC; workloads use ServiceAccounts. Do not share.
- **Watch for privilege escalation via RBAC itself** — the ability to create RoleBindings is effectively the ability to grant oneself any permission the API allows.

---

## 2. Resource Quotas

### Plain-language explanation

A **ResourceQuota** caps the total amount of resources a namespace may consume — total CPU, memory, GPU count, number of pods, PVCs, or object counts. A **LimitRange** works at the pod/container level: it sets default requests and limits and enforces minimums and maximums for individual containers. The two work together: ResourceQuota controls the aggregate ceiling for a tenant, LimitRange controls per-container sanity and supplies defaults so unbounded pods don't slip through.

For GPUs, quotas use the extended resource name (e.g., `nvidia.com/gpu` or `requests.nvidia.com/gpu`). GPUs are integer, non-overcommittable resources — you either get a whole GPU (or MIG slice) or you don't — which makes quota discipline especially important.

### Why it matters in production

GPUs are the single most expensive line item on an ML platform, often 10–40x the cost of equivalent CPU. Without a quota, one data scientist running an unbounded hyperparameter sweep can grab every GPU in the cluster, starving production serving and everyone else's training. Quotas convert an implicit free-for-all into an explicit, enforceable budget. They also make capacity planning possible: if each tenant's ceiling is known, the platform team can reason about total demand versus supply.

### Real-world example / use case

A research org has 32 A100 GPUs. Three teams share them. The platform team allocates 16 GPUs to `fraud-team`, 8 to `risk-team`, and 4 to `marketing-team`, holding 4 in reserve for bursts and system workloads. A ResourceQuota per namespace enforces this. A LimitRange ensures no single training pod requests more than 8 GPUs and that every pod without explicit requests gets sensible defaults so the scheduler can pack the cluster.

### Code or config example

```yaml
# Aggregate ceiling for one tenant, including GPUs.
apiVersion: v1
kind: ResourceQuota
metadata:
  name: fraud-team-quota
  namespace: fraud-team
spec:
  hard:
    requests.cpu: "200"
    requests.memory: 800Gi
    limits.cpu: "300"
    limits.memory: 1200Gi
    requests.nvidia.com/gpu: "16"    # hard GPU cap for the tenant
    persistentvolumeclaims: "50"
    count/jobs.batch: "100"
    pods: "200"
---
# Per-container defaults and bounds. Ensures no unbounded pods and no
# single pod hogging more than 8 GPUs.
apiVersion: v1
kind: LimitRange
metadata:
  name: fraud-team-limits
  namespace: fraud-team
spec:
  limits:
    - type: Container
      default:                # applied if container omits limits
        cpu: "2"
        memory: 4Gi
      defaultRequest:         # applied if container omits requests
        cpu: "500m"
        memory: 1Gi
      max:
        cpu: "32"
        memory: 128Gi
        nvidia.com/gpu: "8"   # max GPUs per single container
      min:
        cpu: "100m"
        memory: 128Mi
```

```bash
# Inspect current usage against the quota.
kubectl describe resourcequota fraud-team-quota -n fraud-team
# Name:                     fraud-team-quota
# Resource                  Used   Hard
# --------                  ----   ----
# requests.nvidia.com/gpu   12     16
# pods                      47     200
```

### Interview questions

**Q: What happens when a pod requests GPUs but the namespace has no matching ResourceQuota entry, versus when the quota is exhausted?**
A: If the ResourceQuota sets any `requests.*` entry, Kubernetes requires every pod to specify requests for the quota'd resources or it is rejected at admission. If the GPU quota is exhausted, new pods requesting GPUs are rejected at admission time with a `forbidden: exceeded quota` error — they never even get scheduled. This is different from CPU/memory, which can be overcommitted; GPUs cannot, so the quota rejection is a hard stop.

**Q: Why do you need both ResourceQuota and LimitRange?**
A: ResourceQuota controls the namespace *total* but doesn't stop one pod from claiming the entire budget, and if a container omits requests while a quota requires them, the pod is rejected. LimitRange fills both gaps: it injects default requests/limits so pods aren't rejected for omission, and it caps per-container size so a single pod can't monopolize the tenant's allocation. They are complementary — aggregate versus per-object control.

**Q: How would you handle GPU quotas when using MIG (Multi-Instance GPU)?**
A: With MIG, a single physical GPU is partitioned into slices exposed as distinct extended resources (e.g., `nvidia.com/mig-1g.5gb`). You set quotas against those specific resource names rather than the whole-GPU `nvidia.com/gpu`. This lets you allocate fractional GPU capacity to tenants whose workloads don't need a full device, improving utilization.

### Pitfalls & best practices

- **Once you set a `requests.*` quota, every pod in the namespace must declare requests** — apply a LimitRange with defaults simultaneously or you will break existing workloads.
- **Quota GPUs by `requests.nvidia.com/gpu`, not `limits`**, since GPUs must have equal requests and limits (they can't be overcommitted); requesting is what counts.
- **Reserve headroom.** Don't allocate 100% of GPUs to tenants; keep a buffer for system pods, upgrades, and bursts.
- **Monitor quota utilization** and alert when tenants approach their ceiling so you can rebalance before jobs start failing.
- **Priority classes and preemption** complement quotas — let production serving preempt low-priority training when contention hits.

---

## 3. Cost Management

### Plain-language explanation

Cost management for a shared platform is about answering "who spent what, and was it worth it?" **Showback** reports each tenant's consumption without actually billing them — it drives awareness. **Chargeback** goes further and bills the cost back to the team's budget, which drives behavior change. **Tagging** (labels/annotations on namespaces, pods, and cloud resources) is what makes attribution possible. **Right-sizing** means matching requested resources to actual usage so you're not paying for idle capacity. **Spot/preemptible instances** are steeply discounted (60–90%) cloud VMs that can be reclaimed with short notice — ideal for fault-tolerant training. **Kubecost** (and OpenCost, its open-source core) is the standard tool for allocating Kubernetes spend down to namespace, deployment, and label.

### Why it matters in production

ML platforms burn money faster than almost any other infrastructure because of GPUs. Left unattended, a platform accumulates zombie training jobs, over-provisioned serving replicas, and idle GPU nodes running 24/7. Cost visibility turns an opaque cloud bill into per-team accountability, and it's often the difference between a platform that gets renewed and one that gets shut down for being too expensive. GPU cost control specifically — ensuring GPUs are actually being *used* when allocated, not sitting idle in an interactive notebook — is where the biggest savings live.

### Real-world example / use case

A company's monthly GPU bill hit $180k. After deploying Kubecost with namespace-level allocation, they discovered `marketing-team` was running eight A100 nodes at 6% average GPU utilization — data scientists left Jupyter kernels holding GPUs overnight. They introduced idle-GPU reclamation (kill notebooks idle > 2 hours), moved all training to spot instances with checkpointing, and right-sized serving from 10 replicas to 4 based on actual p99 load. The bill dropped to $70k with no loss of throughput, and monthly showback reports kept it there.

### Code or config example

```yaml
# Consistent labels are the foundation of cost attribution. Enforce them
# on every namespace and workload.
apiVersion: v1
kind: Namespace
metadata:
  name: fraud-team
  labels:
    team: fraud
    cost-center: "cc-4471"
    environment: production
    owner: fraud-platform-leads
```

```yaml
# Schedule training on spot nodes; tolerate reclamation with checkpointing.
apiVersion: batch/v1
kind: Job
metadata:
  name: nightly-fraud-train
  namespace: fraud-team
  labels:
    team: fraud
    cost-center: "cc-4471"
    workload-type: training
spec:
  backoffLimit: 6            # retry if a spot node is reclaimed mid-run
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-spot: "true"    # or eks.amazonaws.com/capacityType: SPOT
      tolerations:
        - key: "cloud.google.com/gke-spot"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
      containers:
        - name: trainer
          image: registry.internal/fraud-trainer:2.3.1
          resources:
            requests:
              nvidia.com/gpu: "4"
            limits:
              nvidia.com/gpu: "4"
          # trainer writes checkpoints to object storage every N steps
          env:
            - name: CHECKPOINT_URI
              value: "s3://fraud-checkpoints/nightly/"
      restartPolicy: OnFailure
```

```bash
# Query Kubecost's allocation API for last 7 days by namespace.
curl -s "http://kubecost.monitoring:9090/model/allocation" \
  --data-urlencode "window=7d" \
  --data-urlencode "aggregate=namespace" \
  --data-urlencode "accumulate=true" | jq '.data[] | {name, totalCost, gpuCost, efficiency}'
```

### Interview questions

**Q: What is the difference between showback and chargeback, and when would you use each?**
A: Showback reports costs to teams for visibility without moving money; chargeback actually debits the team's budget. Showback is the right first step — it builds cost awareness without political friction and surfaces waste. Move to chargeback once cost data is trusted and you want real accountability, because a team that literally pays for its GPUs will right-size aggressively. Chargeback requires very accurate attribution or it breeds disputes.

**Q: When are spot instances appropriate for ML workloads, and how do you make them safe?**
A: Spot is ideal for interruptible, fault-tolerant work — batch training, hyperparameter sweeps, batch inference. You make it safe with frequent checkpointing to durable storage, `backoffLimit`/retries so reclaimed jobs resume, and handling the ~2-minute termination notice to checkpoint gracefully. Spot is generally *not* appropriate for latency-sensitive online serving unless you run a spot/on-demand mix behind a load balancer with on-demand as a fallback floor.

**Q: A team's GPU nodes show 90% allocation but 8% utilization. What's going on and how do you fix it?**
A: Allocation is how much is *requested/reserved*; utilization is how much is actually *used*. High allocation with low utilization means GPUs are held but idle — classic causes are interactive notebooks that reserve a GPU and sit idle, or oversized requests. Fixes: idle-GPU reaping, right-sizing requests, GPU sharing (MIG or time-slicing) for interactive work, and moving interactive users off dedicated GPUs onto shared ones.

### Pitfalls & best practices

- **Enforce a tagging/label policy with admission control** (OPA Gatekeeper or Kyverno). Untagged resources are unattributable cost.
- **Distinguish allocation from utilization.** Optimizing allocation without watching utilization just moves the waste around.
- **Right-size from real telemetry**, not guesses — use VPA recommendations or historical p95 usage.
- **Reap zombies:** idle notebooks, completed-but-not-deleted jobs, and orphaned PVCs quietly accumulate cost.
- **Don't over-index on spot for everything** — reclamation storms during regional capacity crunches can stall all training at once; keep a baseline of on-demand.
- **Make cost a first-class metric** in team dashboards, not a quarterly finance surprise.

---

## 4. Namespace Isolation

### Plain-language explanation

Namespaces give logical separation, but by default Kubernetes networking is flat — every pod can reach every other pod across all namespaces. **NetworkPolicies** close that hole by defining allowed ingress/egress at the pod level, effectively acting as a firewall between tenants. True namespace isolation combines several layers: NetworkPolicies for traffic, per-namespace ResourceQuotas for compute, RBAC for API access, and sometimes node-level separation (taints/tolerations, dedicated node pools) so tenants don't even share hardware. The **isolation boundary** is the set of guarantees you make: "workloads in namespace A cannot see, reach, or starve workloads in namespace B."

### Why it matters in production

An ML serving endpoint in one namespace often holds sensitive model artifacts and customer data. If the network is flat, a compromised or buggy pod in a low-trust namespace can scrape a model server or exfiltrate data from another tenant. Default-deny NetworkPolicies are a baseline security control and frequently a compliance requirement. Isolation also protects against accidental cross-talk — a misconfigured service in `marketing-team` shouldn't be able to hammer the `fraud-team` feature store.

### Real-world example / use case

A healthcare ML platform must isolate a namespace processing PHI. The team applies a default-deny-all NetworkPolicy to that namespace, then explicitly allows only: ingress from the API gateway namespace, egress to the internal feature store and the DNS service. All other traffic — including cross-tenant and internet egress — is blocked. This satisfies HIPAA network-segmentation expectations while keeping the workload functional.

### Code or config example

```yaml
# 1) Default deny ALL ingress and egress in the namespace.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: phi-inference
spec:
  podSelector: {}            # selects every pod in the namespace
  policyTypes:
    - Ingress
    - Egress
---
# 2) Allow ingress only from the shared API gateway namespace.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-gateway-ingress
  namespace: phi-inference
spec:
  podSelector:
    matchLabels:
      app: model-server
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              role: api-gateway
      ports:
        - protocol: TCP
          port: 8080
---
# 3) Allow egress to the feature store and to cluster DNS only.
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-featurestore-and-dns-egress
  namespace: phi-inference
spec:
  podSelector:
    matchLabels:
      app: model-server
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              role: feature-store
      ports:
        - protocol: TCP
          port: 6379
    - to:                       # DNS resolution must be explicitly allowed
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
```

```bash
# Test isolation from inside a pod in another namespace.
kubectl run probe --rm -it --image=nicolaka/netshoot -n marketing-team -- \
  curl --max-time 5 http://model-server.phi-inference:8080/health
# expected: connection times out (blocked by default-deny)
```

### Interview questions

**Q: By default, can a pod in namespace A talk to a pod in namespace B? How do you stop it?**
A: Yes — Kubernetes networking is flat by default, so all pods can reach all pods regardless of namespace. You stop it by applying NetworkPolicies, typically a default-deny-all in each tenant namespace followed by explicit allow rules. This requires a CNI that enforces NetworkPolicy (Calico, Cilium, etc.); with a non-enforcing CNI the policies are silently ignored.

**Q: A team applied a default-deny egress policy and their pods suddenly can't resolve DNS. Why?**
A: A default-deny egress blocks *all* outbound traffic, including UDP/TCP port 53 to CoreDNS in kube-system. DNS is easy to forget because it's implicit. The fix is an explicit egress allow rule to the kube-system namespace (or the kube-dns pods) on port 53. This is one of the most common NetworkPolicy gotchas.

**Q: What are the layers of namespace isolation beyond NetworkPolicy?**
A: RBAC (API-level access control), ResourceQuota/LimitRange (compute isolation so one tenant can't starve another), NetworkPolicy (network isolation), node isolation via taints/tolerations and dedicated node pools (hardware separation), and PodSecurity standards (preventing privilege escalation/container escape). Full isolation is defense in depth across all these layers, since network policy alone doesn't stop resource starvation or API access.

### Pitfalls & best practices

- **Confirm your CNI enforces NetworkPolicy.** The flannel default does not; Calico and Cilium do. Policies applied to a non-enforcing CNI are a false sense of security.
- **Always allow DNS egress** when using default-deny egress, or everything silently breaks.
- **Label namespaces consistently** (`kubernetes.io/metadata.name` is auto-added) so `namespaceSelector` rules are reliable.
- **Start default-deny, then add allows** rather than deny-listing — allow-listing is the only safe posture.
- **Test policies actively** with probe pods; NetworkPolicy failures are silent and easy to miss until an audit.
- **Combine with quotas** — network isolation without resource isolation still leaves the noisy-neighbor problem.

---

## 5. Platform SLAs

### Plain-language explanation

A **Service Level Agreement (SLA)** is a promise to your users about the platform's behavior — typically availability ("99.9% uptime") and latency ("p99 inference latency under 200 ms"). An SLA is usually backed by consequences (credits, escalations). An **SLO (Service Level Objective)** is the internal target you actually engineer toward, usually stricter than the SLA. An **SLI (Service Level Indicator)** is the measured metric that feeds the SLO — e.g., the ratio of successful requests to total requests. The **error budget** is the inverse of the SLO: if your SLO is 99.9% availability, you have a 0.1% budget of allowed failure (about 43 minutes per month) to "spend" on risk — deployments, experiments, maintenance.

### Why it matters in production

On a shared ML platform, dozens of teams depend on the same serving infrastructure, feature store, and model registry. Without explicit SLAs, expectations are implicit and every incident becomes a dispute. SLOs give the platform team an objective, data-driven way to balance reliability against velocity: when the error budget is healthy, ship fast; when it's exhausted, freeze risky changes and focus on stability. Error budgets replace "should we deploy?" arguments with a number. For ML platforms specifically, SLAs must also cover ML-specific dimensions — model freshness, feature availability, and sometimes prediction quality — not just raw uptime.

### Real-world example / use case

A platform team commits to a 99.9% availability SLO and a p99 latency SLO of 250 ms for the online inference service, measured monthly. They define SLIs from the load balancer: availability = (non-5xx responses / total responses), latency = 99th percentile server-side duration. They track burn rate: a fast burn (consuming the month's budget in an hour) pages on-call immediately; a slow burn (trending to exhaust the budget by month-end) creates a ticket. When a bad model deployment pushed p99 to 600 ms and burned 40% of the budget in a day, the automated error-budget policy triggered a deployment freeze until reliability work restored headroom.

### Code or config example

```yaml
# Prometheus recording + alerting rules implementing SLO burn-rate alerts.
groups:
  - name: inference-slo
    rules:
      # SLI: success ratio over 5m and 1h windows.
      - record: job:inference_availability:ratio_rate5m
        expr: |
          sum(rate(http_requests_total{job="inference",code!~"5.."}[5m]))
          /
          sum(rate(http_requests_total{job="inference"}[5m]))

      # Fast-burn alert: burning the 30d budget 14.4x too fast => page now.
      # (SLO 99.9% => error budget 0.1%.)
      - alert: InferenceErrorBudgetFastBurn
        expr: |
          (1 - job:inference_availability:ratio_rate5m) > (14.4 * 0.001)
        for: 2m
        labels:
          severity: page
        annotations:
          summary: "Inference SLO fast burn — paging on-call"
          runbook: "https://runbooks.internal/inference-slo-burn"

      # Latency SLI: p99 under 250ms.
      - record: job:inference_latency_p99:5m
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket{job="inference"}[5m])) by (le))
      - alert: InferenceLatencySLOViolation
        expr: job:inference_latency_p99:5m > 0.25
        for: 10m
        labels:
          severity: ticket
        annotations:
          summary: "Inference p99 latency above 250ms SLO"
```

### Interview questions

**Q: Explain the relationship between SLI, SLO, SLA, and error budget.**
A: An SLI is the measured metric (e.g., success ratio). An SLO is the internal target for that SLI (99.9%). An SLA is the external promise, usually looser than the SLO so you have a safety margin (e.g., SLA 99.5%, SLO 99.9%). The error budget is 100% minus the SLO — the amount of unreliability you're allowed to spend on change and risk. Keeping the SLO stricter than the SLA means you detect and correct before contractually breaching.

**Q: What is a burn-rate alert and why is it better than a simple threshold alert?**
A: A burn-rate alert measures how fast you're consuming the error budget relative to the window. A single threshold ("error rate > 1%") is either too noisy or too slow. Multi-window burn-rate alerts (e.g., fast burn over 5m/1h for paging, slow burn over 6h/3d for tickets) page only when the trajectory genuinely threatens the SLO, giving high signal-to-noise and appropriate urgency.

**Q: What SLAs are unique to an ML platform versus a generic web service?**
A: Beyond availability and latency, ML platforms care about model/feature freshness (is the deployed model within its retraining SLA?), feature availability and completeness (are online features being served, or falling back to defaults?), and sometimes prediction-quality SLOs tied to monitored accuracy or drift. A model can be "up" (200 responses, low latency) yet serving garbage due to stale features — so ML SLAs must include correctness/freshness dimensions, not just the request path.

### Pitfalls & best practices

- **Set SLOs from what users actually need**, not what's easy to hit. An over-tight SLO wastes engineering effort; too loose and users suffer.
- **Keep the SLA looser than the SLO** so internal breaches don't become external ones.
- **Measure SLIs at the point the user experiences them** (load balancer/gateway), not deep inside the service.
- **Enforce error-budget policy** — an error budget with no consequences is just a dashboard. Tie budget exhaustion to a deployment freeze.
- **Include freshness/quality SLOs for ML**, not just uptime.
- **Don't SLO everything.** Pick the few user-critical journeys; over-instrumenting dilutes focus.

---

## 6. Incident and On-Call Practices for ML Platforms

### Plain-language explanation

Incident management is the discipline of responding to unplanned service degradation in a structured way: detect, triage by severity, assign an incident commander, mitigate, resolve, and learn. **On-call** is the rotation of engineers responsible for responding to pages. A **runbook** is a step-by-step guide for diagnosing and fixing a known class of problem. **Escalation** defines who gets pulled in when the first responder can't resolve it. A **blameless postmortem** is a written retrospective that focuses on systemic causes and fixes rather than assigning fault. ML platforms add their own incident types — model **drift**, feature-pipeline failures, GPU **OOM** (out-of-memory) errors, and data-quality regressions — on top of the usual infrastructure failures.

### Why it matters in production

A shared platform failing at 2 a.m. affects every dependent team simultaneously, so the blast radius of any incident is large and coordination matters. Good incident practice minimizes mean time to recovery (MTTR) and prevents the same failure from recurring. On-call health matters too: burned-out engineers paged by noisy alerts respond poorly and quit. ML incidents are especially tricky because failures are often *silent* — a model with stale features still returns 200s while quietly degrading business metrics — so detection and runbooks must cover model behavior, not just process health.

### Real-world example / use case

At 02:14 a fraud model's precision drops sharply. There's no 5xx spike — the service is "healthy" by infra metrics — but a drift monitor fires because the input feature distribution shifted after an upstream schema change dropped a feature to null. The on-call engineer follows the drift runbook: confirms via the monitoring dashboard, identifies the broken upstream pipeline, and executes the documented mitigation — roll back to the previous model version and pin the last good feature snapshot. Service is restored to acceptable precision within 30 minutes. The next day, a blameless postmortem produces action items: add a schema-validation gate on the feature pipeline and a null-rate alert so the failure is caught before it reaches the model.

### Code or config example

```yaml
# Alerting rules for ML-specific incidents feeding the on-call pager.
groups:
  - name: ml-platform-incidents
    rules:
      # GPU OOM: container killed with reason OOMKilled on a GPU node.
      - alert: GpuPodOOMKilled
        expr: |
          increase(kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}[10m]) > 0
          and on(pod) kube_pod_container_resource_requests{resource="nvidia_com_gpu"} > 0
        for: 0m
        labels:
          severity: page
          team: ml-platform
        annotations:
          summary: "GPU pod {{ $labels.pod }} OOMKilled"
          runbook: "https://runbooks.internal/gpu-oom"

      # Feature freshness: online feature store lagging its SLA.
      - alert: FeatureStoreStale
        expr: time() - feature_store_last_materialization_timestamp_seconds > 3600
        for: 5m
        labels:
          severity: page
        annotations:
          summary: "Online features stale > 1h — models may serve on defaults"
          runbook: "https://runbooks.internal/feature-staleness"

      # Drift: population stability index above threshold.
      - alert: PredictionDriftHigh
        expr: model_input_psi{model="fraud-v3"} > 0.25
        for: 15m
        labels:
          severity: ticket
        annotations:
          summary: "PSI > 0.25 on fraud-v3 inputs — investigate drift"
          runbook: "https://runbooks.internal/model-drift"
```

```markdown
<!-- Excerpt of a runbook: runbooks/gpu-oom.md -->
# Runbook: GPU Pod OOMKilled

## Symptom
Training/serving pod on a GPU node terminates with reason `OOMKilled`.

## Triage (first 5 minutes)
1. `kubectl describe pod <pod> -n <ns>` — confirm `Last State: Terminated, Reason: OOMKilled`.
2. Check `nvidia-smi` on the node (or DCGM metrics) for GPU memory pressure vs host RAM.
   - Host-RAM OOM  -> raise container memory `limits`.
   - GPU-VRAM OOM  -> reduce batch size / enable gradient checkpointing / use a larger GPU or MIG slice.

## Mitigation
- Serving: roll back to previous replica config; scale down concurrency.
- Training: resubmit with reduced batch size from last checkpoint (`CHECKPOINT_URI`).

## Escalation
- Unresolved in 30 min or repeated across nodes -> page ML Platform L2 (@ml-platform-oncall-l2).
- Suspected node hardware fault -> cordon node, page Infra.
```

### Interview questions

**Q: What makes a postmortem "blameless," and why does it matter?**
A: A blameless postmortem focuses on systemic and process causes — why the system allowed the failure — rather than on individual mistakes. It matters because blame drives people to hide information, which destroys the learning that prevents recurrence. When engineers trust they won't be punished, they share the full timeline honestly, producing better root-cause analysis and durable fixes. The output is action items on systems, not reprimands on people.

**Q: Name ML-specific incidents that don't exist for a typical web service and how you'd detect them.**
A: Model drift (input/output distribution shift) detected via PSI/KL-divergence monitors; feature staleness/pipeline failures detected via materialization-freshness metrics; data-quality regressions (null spikes, schema changes) via validation gates; GPU OOM detected via OOMKilled + GPU resource labels; training-job failures and silent accuracy decay detected via delayed ground-truth evaluation. The theme is that many ML incidents are silent — the service returns 200s while quality degrades — so you monitor model and data behavior, not just the request path.

**Q: How should on-call escalation and severity levels be structured?**
A: Define severity tiers (e.g., SEV1 = platform-wide outage, customer impact; SEV2 = major degradation; SEV3 = minor). Each severity maps to response expectations, who's notified, and whether an incident commander is assigned. Escalation is time-boxed: L1 on-call responds first; if unresolved within a defined window or beyond their runbook, it escalates to L2/subject-matter experts, then engineering leadership for SEV1s. Automate the escalation in the paging tool so it doesn't depend on a stressed human remembering to hand off.

### Pitfalls & best practices

- **Fight alert fatigue.** Every page should be actionable and tied to a runbook; delete or downgrade alerts that don't require human action.
- **Write runbooks for known failure modes** and link them directly in the alert annotation so responders aren't hunting at 3 a.m.
- **Keep postmortems blameless and track action items to completion** — a postmortem with un-actioned items is theater.
- **Monitor the model, not just the process.** Health checks that only verify the pod is up miss the most damaging ML incidents.
- **Protect on-call sustainability** — reasonable rotation size, follow-the-sun where possible, and compensating time off after heavy shifts.
- **Practice with game days / chaos drills** so responders know the runbooks before a real incident.
- **Have a rollback path for models**, not just code — pinning a previous model version and feature snapshot is often the fastest mitigation.

---

## Summary

Platform administration is what turns a collection of ML tooling into a trustworthy shared platform. We started with **RBAC and multi-tenancy** — Roles, ClusterRoles, bindings, and ServiceAccounts that establish who can do what, and the namespace/cluster/virtual-cluster isolation models that define tenant boundaries. On top of that boundary we layered **resource quotas** (ResourceQuota for aggregate ceilings, LimitRange for per-container defaults and caps), with special attention to non-overcommittable GPUs. **Cost management** made consumption visible and accountable through showback/chargeback, consistent tagging, right-sizing, spot instances, and tools like Kubecost — with GPU idle-utilization as the highest-leverage savings. **Namespace isolation** hardened the boundary with default-deny NetworkPolicies and defense-in-depth across RBAC, quotas, and node separation. **Platform SLAs** gave us the language of SLIs, SLOs, SLAs, and error budgets to balance reliability against velocity, extended with ML-specific freshness and quality objectives. Finally, **incident and on-call practices** — runbooks, severity-based escalation, blameless postmortems, and monitoring for ML-specific failures like drift, feature staleness, and GPU OOM — keep the platform recoverable and continuously improving.

The through-line: a good platform administrator makes fairness, security, cost, and reliability *explicit and enforceable* rather than implicit and hoped-for.

## Chapter Q&A

**Q1. You're standing up a new multi-tenant ML platform on a single Kubernetes cluster for five internal teams. What are the first four platform-administration controls you put in place and why?**
A: (1) A namespace per team as the tenant boundary. (2) RBAC bound to IdP groups so each team manages only its own namespace, plus a scoped ServiceAccount for CI. (3) ResourceQuota + LimitRange per namespace, especially GPU caps, to prevent one team starving others. (4) Default-deny NetworkPolicies with explicit allow rules (including DNS) so tenants can't reach each other's workloads. These four establish identity, compute fairness, and network isolation — the minimum viable tenant boundary — before layering cost and SLA tooling.

**Q2. A RoleBinding grants a ClusterRole. What is the effective scope of the granted permissions?**
A: The permissions are limited to the namespace of the RoleBinding, even though the referenced object is a ClusterRole. This lets you define reusable permission sets once as ClusterRoles and apply them namespace-by-namespace. If you'd used a ClusterRoleBinding instead, the permissions would apply cluster-wide.

**Q3. Why can't GPUs be overcommitted the way CPU can, and what does that imply for quotas?**
A: CPU is a compressible resource — the kernel time-slices it, so requests can be overcommitted and throttled under contention. A GPU is assigned to a container as a whole device (or a fixed MIG slice) and isn't transparently time-shared by default, so it can't be overcommitted; requests must equal limits. This means GPU quota exhaustion is a hard admission-time rejection, making accurate GPU quotas and headroom reservation critical.

**Q4. Your monthly cloud bill is dominated by GPU spend and leadership wants a 40% cut without hurting throughput. Walk through your approach.**
A: First get visibility with Kubecost/OpenCost, allocating GPU cost by namespace and label, and separate allocation from utilization. Attack low utilization: reap idle notebooks, enable GPU sharing (MIG/time-slicing) for interactive work, and right-size requests from telemetry. Move interruptible training and batch inference to spot/preemptible nodes with checkpointing and retries. Right-size serving replicas to actual p99 load with autoscaling. Institute showback (then chargeback) so teams sustain the gains. The biggest wins are usually killing idle GPUs and spot-migrating training.

**Q5. After applying a default-deny egress NetworkPolicy, all pods in a namespace break. What's the most likely cause and fix?**
A: The policy blocked DNS (port 53 to CoreDNS in kube-system), so name resolution fails and everything appears broken. Fix: add an explicit egress allow to kube-system on UDP/TCP 53, then add the other required egress destinations. Also verify the CNI actually enforces NetworkPolicy — with a non-enforcing CNI the policy wouldn't take effect at all.

**Q6. Distinguish SLA, SLO, SLI, and error budget, and explain how an error budget changes team behavior.**
A: SLI = measured metric; SLO = internal target for it; SLA = external promise (looser than the SLO); error budget = 100% minus the SLO, the permitted unreliability. The error budget converts reliability-versus-velocity arguments into a number: with budget remaining, teams ship features freely; when it's exhausted, an error-budget policy freezes risky changes until reliability is restored. It aligns dev and ops on a shared, objective trade-off.

**Q7. What ML-specific failure modes must a platform's monitoring and runbooks cover that a generic web platform ignores?**
A: Model drift (input/output distribution shift), feature staleness and feature-pipeline failures, data-quality regressions (null spikes, schema changes), GPU OOM, and silent accuracy decay pending ground truth. The key insight is that many ML failures are silent — the service returns 200s at low latency while quality collapses — so monitoring must observe model and data behavior (PSI, freshness, quality) and runbooks must include model-level mitigations like rolling back to a prior model version and pinning a good feature snapshot.

**Q8. What are the ingredients of an effective blameless postmortem, and what's the anti-pattern?**
A: A clear timeline, impact quantification, contributing/systemic causes (not individual blame), what went well, and concrete action items with owners and due dates tracked to completion. The anti-pattern is naming a culprit ("human error") as the root cause, which suppresses honest information-sharing and guarantees recurrence. Blamelessness plus tracked action items is what turns an incident into durable improvement.

**Q9. How would you enforce that every workload is properly tagged for cost attribution?**
A: Use an admission-control policy engine (OPA Gatekeeper or Kyverno) to reject or mutate resources that lack required labels like `team`, `cost-center`, and `environment`. Enforcing at admission guarantees no untagged, unattributable resources enter the cluster, which is the foundation for accurate showback/chargeback in Kubecost.

**Q10. A single data scientist launches a hyperparameter sweep that consumes every GPU, stalling production serving. Which controls should have prevented this, and how do you protect serving going forward?**
A: A per-namespace GPU ResourceQuota would cap the tenant's total GPUs, and a LimitRange would cap any single pod. To protect production specifically, assign a high PriorityClass to serving with preemption enabled so it can evict low-priority training under contention, keep GPU headroom unallocated for bursts, and separate training and serving into dedicated node pools via taints/tolerations. Together, quotas plus priority/preemption plus node separation ensure a training storm can't starve production.
