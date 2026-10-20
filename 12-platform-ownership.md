# Chapter 12 — Shared ML Platform Ownership

Most organizations reach a point where building one-off ML systems no longer scales. Every team reinvents feature pipelines, hand-rolls deployment scripts, and copies someone else's notebook to production with the bugs intact. The answer is a *shared ML platform* — a common set of tools, paved paths, and services that many teams build on. But a platform is only successful if people choose to use it, and that turns out to be as much an organizational and product challenge as a technical one.

This chapter is deliberately lighter on heavy code and heavier on judgment. Owning a shared platform means thinking like a product manager, an SRE, a policy author, and a technical writer all at once. We'll move from the foundational mindset (treating the platform as a product) through self-service tooling and golden paths, into the harder tradeoffs of governance versus autonomy, and finish with onboarding and documentation — the unglamorous work that decides whether adoption actually sticks.

## Platform-as-a-Product Mindset

### Plain-language explanation

Treating your platform *as a product* means your data scientists and ML engineers are your *customers*, not your captive audience. You have a roadmap, you gather feedback, you measure satisfaction and adoption, and you compete — implicitly — against the alternative of teams building it themselves or reaching for an external SaaS. A platform team that ships features nobody asked for, with no way to measure whether anyone uses them, is running a cost center. A platform team that runs like a product organization earns its budget by making internal customers demonstrably faster.

### Why it matters in production

The failure mode of internal platforms is the "build it and they will come" fallacy. Teams pour a year into a beautiful platform, mandate its use, and discover that adoption is grudging and workarounds are everywhere. Product thinking prevents this: you validate demand before building, you instrument usage so you know what's actually valued, and you treat adoption friction as a bug. In production terms, a platform that customers trust and want to use concentrates ML workloads onto well-tested paths — which is exactly what improves reliability, security posture, and cost control.

### Real-world example / use case

Spotify and Airbnb both famously reframed their internal infrastructure teams around product management, complete with PMs, user research, and adoption dashboards. A concrete pattern: a platform team introduces a quarterly internal NPS survey ("How likely are you to recommend the ML platform to a colleague?") alongside hard adoption metrics — number of models served through the platform, percentage of production models on paved paths, and time-to-first-deployment for a new project. When time-to-first-deployment drops from three weeks to two days, that's the number the platform team takes to the funding conversation.

### Code or config example

A lightweight way to make this concrete is a published, versioned "platform product" definition and a metrics contract:

```yaml
# platform-product.yaml — a living definition of what the platform promises
product:
  name: "Nimbus ML Platform"
  mission: "Get any model from notebook to production in under a day, safely."
  internal_customers:
    - persona: "Data Scientist"
      primary_job: "Train and ship a model without learning Kubernetes"
    - persona: "ML Engineer"
      primary_job: "Own a production service with observability built in"
  slos:
    time_to_first_deploy_p50: "< 1 day"
    platform_api_availability: "99.9%"
  north_star_metric: "percent_of_prod_models_on_paved_path"
  targets:
    percent_of_prod_models_on_paved_path: 0.80   # this quarter
    internal_nps: 30

adoption_dashboard:
  refreshed: "daily"
  panels:
    - models_served_total
    - weekly_active_data_scientists
    - time_to_first_deploy_histogram
    - support_tickets_by_category
```

Pairing a mission statement with a *north star metric* and a public dashboard changes the conversation from "what features shall we build" to "what moves the metric our customers care about."

### Interview questions

**Q: How would you measure the success of an internal ML platform?**
A: I'd resist vanity metrics like "number of services deployed" and anchor on customer outcomes. My north star would be the percentage of production models running on paved paths, because that captures both adoption and the reliability/governance benefits we actually care about. I'd complement it with leading indicators — time-to-first-deployment, weekly active users, and internal NPS — and a diagnostic layer like support tickets by category, which tells me *where* the friction is. Crucially I'd tie these to business value: if the platform halves the time from experiment to production, that's velocity the org can feel.

**Q: A powerful internal team refuses to adopt your platform and keeps building their own. How do you respond?**
A: First I'd treat it as product feedback, not insubordination — they're a lead user telling me my product doesn't meet a real need. I'd interview them to understand the gap: is it a missing capability, performance, or trust? Sometimes the right answer is to absorb their approach into the platform. I'd avoid a blanket mandate early on, because mandates without a compelling product breed shadow infrastructure. Once the platform is genuinely better for 80% of cases, I'd use governance (e.g., production access requires paved-path compliance) to close the long tail — but that's a late-stage lever, not a first move.

### Pitfalls & best practices

- **Pitfall:** Building on assumptions instead of validated demand. **Best practice:** Run discovery interviews and ship a thin slice to a design partner team first.
- **Pitfall:** No feedback loop — you can't tell what's used. **Best practice:** Instrument everything; make an adoption dashboard the team's home screen.
- **Pitfall:** Mandating adoption before the product is good. Mandates convert unhappy customers into resentful ones and drive shadow tooling. **Best practice:** Earn adoption first, gate the tail later.

## Self-Service Tooling for Data Scientists

### Plain-language explanation

Self-service means a data scientist can accomplish common tasks — spin up an experiment environment, register a model, deploy a service — without filing a ticket and waiting for the platform team. The platform team's job shifts from doing the work to *building the tools that let others do the work safely*. The guiding idea is the **paved path**: the easiest, most obvious way to do something is also the well-supported, secure, observable way.

### Why it matters in production

Ticket-driven infrastructure is a throughput ceiling and a morale drain. Every human handoff adds latency and inconsistency, and the platform team becomes a bottleneck that scales linearly with the number of teams. Self-service breaks that coupling. In production, it also enforces consistency: if everyone deploys through the same CLI that wires up logging, metrics, health checks, and rollback, then every service inherits those properties for free. The notebook-to-production gap — where models validated in a notebook never make it to a robust service — closes when the path from one to the other is a single command.

### Real-world example / use case

Consider Netflix's Metaflow: a data scientist writes plain Python, decorates steps, and the framework handles scaling to the cloud, versioning, and scheduling — no infra tickets. A common enterprise version of this is a `mlp` CLI: a data scientist runs `mlp init`, gets a compliant project, iterates in a notebook, and runs `mlp deploy` to promote the trained model to a staged, monitored endpoint. The notebook-to-prod bridge is often a tool like `nbconvert` plus a papermill-executed pipeline, wrapped so the scientist never sees the plumbing.

### Code or config example

A self-service CLI that encodes the paved path:

```bash
# Bootstrap a new, compliant ML project from the golden template
$ mlp init fraud-scoring --template classification-service
✔ Scaffolded fraud-scoring/ (git, CI, model card, tests, Dockerfile)

# Iterate locally; run the training pipeline in a managed environment
$ mlp train --env gpu-small
✔ Run a1b2c3 logged to experiment tracker; model registered as fraud-scoring:0.3.0

# Promote to staging with observability, autoscaling, and rollback wired in
$ mlp deploy fraud-scoring:0.3.0 --stage staging
✔ Endpoint live: https://staging.mlp/fraud-scoring  (canary 10%, dashboards linked)

# Convert an exploratory notebook into a parameterized, reproducible job
$ mlp notebook promote explore.ipynb --schedule "0 6 * * *"
✔ Notebook parameterized via papermill; scheduled as job fraud-scoring-nightly
```

Under the hood, `mlp deploy` might expand to a standard Kubernetes/Knative manifest so no user hand-writes YAML:

```yaml
# Generated by `mlp deploy` — users never edit this directly
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: fraud-scoring
  labels:
    mlp.platform/owner: fraud-team
    mlp.platform/paved-path: "true"
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/target: "50"
    spec:
      containers:
        - image: registry.internal/fraud-scoring:0.3.0
          ports: [{ containerPort: 8080 }]
          readinessProbe: { httpGet: { path: /healthz } }
```

### Interview questions

**Q: What's the difference between a paved path and a mandate, and why does it matter?**
A: A paved path is the easy, supported default — you make the right thing the path of least resistance, so people choose it voluntarily. A mandate forces it. Paved paths scale better because they align incentives: the scientist gets speed, observability, and support "for free," so they *want* the path. Mandates create adversarial dynamics and shadow systems. In practice I'd use paved paths to win the majority and reserve hard gates for genuine safety/compliance boundaries.

**Q: How do you bridge the notebook-to-production gap without frustrating data scientists?**
A: The gap exists because notebooks are great for exploration and terrible for reproducibility and serving. Rather than banning notebooks, I'd meet scientists where they are: tools like papermill let a notebook become a parameterized, versioned job, and nbconvert can extract clean modules. The platform provides a one-command promotion that runs the notebook in a controlled environment, captures inputs/outputs, and wraps the model in the standard serving template. The scientist keeps their preferred workflow; the platform absorbs the reproducibility burden.

### Pitfalls & best practices

- **Pitfall:** Self-service that only covers the happy path, forcing tickets for anything unusual. **Best practice:** Provide well-documented escape hatches so the paved path never becomes a cage.
- **Pitfall:** Leaky abstractions — the CLI works until it breaks and dumps a Kubernetes stack trace. **Best practice:** Invest in error messages and diagnostics that speak the user's language.
- **Pitfall:** Treating self-service as "throw a tool over the wall." **Best practice:** Pair tools with examples, and watch real users struggle before declaring it done.

## Golden Paths / Templates

### Plain-language explanation

A **golden path** is the opinionated, blessed way to build a specific kind of thing — a batch scoring job, a real-time model service, a feature pipeline. Templates make golden paths concrete: instead of a blank repo, a developer gets a fully scaffolded project with CI, tests, a model card, containerization, and observability already wired up. Tools like **cookiecutter** and **Backstage software templates** turn "how do I start?" into a two-minute form.

### Why it matters in production

Scaffolding is where standards get enforced without meetings. If every generated project already has a CI pipeline that runs tests, scans dependencies, and produces a model card, you've made compliance the default rather than an afterthought. This dramatically reduces drift: instead of hundreds of subtly different repos, you have variations on a small number of known-good shapes, which makes them auditable, upgradable, and supportable. When you patch a vulnerability or add a new required check, you update the template and know exactly which projects descend from it.

### Real-world example / use case

Spotify's Backstage popularized the "software template" — an internal developer portal where you pick "Create ML Service," fill in a form (name, owner, model type), and Backstage generates a repo from a template, registers it in the service catalog, and sets up CI. For ML specifically, a template might scaffold a project with a `model_card.md`, a `tests/` folder containing data-validation and inference smoke tests, and a pre-wired deployment config so the very first commit is production-shaped.

### Code or config example

A cookiecutter configuration and the golden-path repo structure it produces:

```json
// cookiecutter.json — prompts and defaults for the ML service template
{
  "project_name": "my-ml-service",
  "owner_team": "ml-platform",
  "model_type": ["classification", "regression", "ranking"],
  "serving_mode": ["realtime", "batch"],
  "python_version": "3.11",
  "include_feature_store": "yes"
}
```

```text
# Generated golden-path repo structure
{{cookiecutter.project_name}}/
├── README.md
├── model_card.md                # governance artifact, required
├── pyproject.toml
├── src/
│   ├── train.py                 # standard training entrypoint
│   └── serve.py                 # standard serving handler
├── tests/
│   ├── test_data_validation.py  # schema/distribution checks
│   └── test_inference.py        # smoke test on sample payload
├── .github/workflows/ci.yaml    # lint, test, scan, build, register
├── deploy/service.yaml          # paved-path serving manifest
└── docs/onboarding.md
```

A Backstage template stitches the same thing into the developer portal:

```yaml
# template.yaml — Backstage software template (abridged)
apiVersion: scaffolder.backstage.io/v1beta3
kind: Template
metadata:
  name: ml-service
  title: Create an ML Service
  tags: [ml, recommended]
spec:
  parameters:
    - title: Service details
      properties:
        name: { type: string }
        owner: { type: string, ui:field: OwnerPicker }
        model_type: { type: string, enum: [classification, regression, ranking] }
  steps:
    - id: fetch
      action: fetch:template
      input:
        url: ./skeleton
        values: { name: "${{ parameters.name }}" }
    - id: publish
      action: publish:github
      input: { repoUrl: "github.com?repo=${{ parameters.name }}&owner=org" }
    - id: register
      action: catalog:register
      input: { catalogInfoUrl: "...catalog-info.yaml" }
```

### Interview questions

**Q: How do you keep golden-path templates from going stale after teams generate projects from them?**
A: This is the central weakness of scaffolding — it's a point-in-time copy, so once you generate a repo it diverges. I'd tackle it on two fronts. First, keep as much *behavior* as possible in versioned shared libraries and reusable CI workflows that projects reference, so improvements propagate without regenerating. Second, for the parts that must live in the repo, use an update tool (like `cruft`, which tracks the template commit a project was generated from) to offer automated update PRs when the template changes. And I'd track "template version in use" as a fleet metric so I can drive upgrades.

**Q: When should you create a new golden path versus telling a team to use an existing one?**
A: A golden path is a bet that a pattern is common enough to be worth standardizing and maintaining. I'd create one when I see the same shape emerging across three or more teams — the "rule of three." Before that, forcing a template can be premature standardization that locks in the wrong abstraction. Every golden path is also a maintenance commitment, so I keep the number small and well-differentiated; too many templates is as confusing as none.

### Pitfalls & best practices

- **Pitfall:** Templates as one-time copies that immediately rot. **Best practice:** Push logic into shared libraries and use update tooling like `cruft`.
- **Pitfall:** Over-templating — a template for every tiny variation. **Best practice:** Keep a small, curated catalog; parameterize instead of proliferating.
- **Pitfall:** Templates that generate broken or non-runnable projects. **Best practice:** Continuously test the template by generating and building a project in CI.

## Governance vs Autonomy Tradeoffs

### Plain-language explanation

Governance is the set of rules that keep the platform safe, compliant, and cost-controlled; autonomy is the freedom that keeps teams fast and motivated. These pull against each other. The key mental model is **guardrails vs gates**. A *guardrail* is a preventive default that stops dangerous actions automatically but lets safe ones flow (like a policy that blocks deploying an image with critical CVEs). A *gate* is a checkpoint that halts progress until a human approves. Guardrails preserve autonomy; gates trade it away for control. **Policy-as-code** is how you implement guardrails at scale — rules expressed as code, evaluated automatically, versioned and testable like any other software.

### Why it matters in production

Manual governance doesn't scale and is inconsistent — a human reviewer catching policy violations by eye is slow and error-prone. Every gate you add is latency injected into every deployment, and latency erodes the velocity that justified the platform. Policy-as-code lets you enforce the non-negotiables (data residency, PII handling, resource limits, approved base images) automatically and uniformly, so you can be strict where it matters and permissive everywhere else. Getting this balance wrong in either direction is costly: too many gates and teams route around you; too few guardrails and you get a security or compliance incident.

### Real-world example / use case

A common setup uses **Open Policy Agent (OPA)** with **Gatekeeper** in Kubernetes: policies are evaluated on every resource admission, so a deployment that lacks an owner label, requests no resource limits, or uses an unapproved registry is rejected automatically — no human in the loop. For ML specifically, a policy might require that any model service carry a link to an approved model card and a data-classification label before it can reach production. Teams stay autonomous within the guardrails; only genuinely high-risk changes (say, deploying a model that touches regulated financial data) trip an explicit approval gate.

### Code or config example

A Gatekeeper constraint template plus a Rego policy that enforces ML governance as a guardrail:

```rego
# require_model_governance.rego — deny prod model services missing governance metadata
package mlplatform.governance

violation[{"msg": msg}] {
  input.review.object.kind == "Service"
  labels := input.review.object.metadata.labels
  labels["mlp.platform/environment"] == "prod"

  not labels["mlp.platform/model-card"]
  msg := "prod model services must reference an approved model card (label mlp.platform/model-card)"
}

violation[{"msg": msg}] {
  input.review.object.kind == "Service"
  labels := input.review.object.metadata.labels
  not labels["mlp.platform/data-classification"]
  msg := "model services must declare a data-classification label"
}
```

```yaml
# constraint.yaml — apply the policy fleet-wide as an admission guardrail
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: RequireModelGovernance
metadata:
  name: require-model-governance
spec:
  enforcementAction: deny        # guardrail: block automatically, no ticket
  match:
    kinds:
      - apiGroups: ["serving.knative.dev"]
        kinds: ["Service"]
```

Contrast that automatic guardrail with an explicit *gate* reserved for high-risk cases:

```yaml
# gate only fires for regulated data — most deploys never see it
approval_gate:
  trigger: "labels['mlp.platform/data-classification'] == 'regulated'"
  required_approvers: ["risk-committee"]
  sla_hours: 24
```

### Interview questions

**Q: Explain the difference between guardrails and gates, and how you decide which to use.**
A: Guardrails are automated preventive controls that let safe actions through and block unsafe ones without human intervention; gates are manual checkpoints that pause everything until someone approves. My default is guardrails, because they preserve velocity and scale — policy-as-code evaluates instantly and consistently. I reserve gates for decisions that genuinely need human judgment or accountability, like deploying against regulated data or an irreversible action with large blast radius. The design goal is to make the risky path require a gate while the common path flows through guardrails untouched, so 95% of work is fast and the 5% that's dangerous is deliberate.

**Q: What is policy-as-code and what does it buy you over a review checklist?**
A: Policy-as-code expresses governance rules as version-controlled, testable code — for example Rego policies evaluated by OPA/Gatekeeper — rather than as prose in a wiki that humans check by hand. The benefits are consistency (the same rule applies identically every time), speed (evaluated in milliseconds at admission), auditability (the policy and its history are in git), and testability (you can unit-test a policy before rolling it out). It shifts governance from "hope people follow the checklist" to "the system enforces it," and it lets you roll out changes gradually with dry-run/audit modes before enforcing.

**Q: How do you introduce a new restrictive policy without breaking existing teams?**
A: I'd never flip a strict policy to `deny` on day one. I'd start in audit/dry-run mode so the policy logs violations without blocking, measure how many existing workloads would fail, and share that impact with affected teams along with a migration path and deadline. I'd provide tooling or auto-fix PRs where possible to make compliance cheap, then move to `deny` for new resources first (grandfathering existing ones), and finally enforce fleet-wide. Communication and a realistic runway are as important as the policy itself.

### Pitfalls & best practices

- **Pitfall:** Governance-by-gate everywhere, turning the platform into a bureaucracy teams route around. **Best practice:** Default to automated guardrails; use gates sparingly and only for true high-risk decisions.
- **Pitfall:** Rolling out strict policies with no runway, breaking production. **Best practice:** Ship policies in audit mode first, measure impact, then enforce.
- **Pitfall:** Policies with cryptic denial messages. **Best practice:** Every deny should explain what's wrong and how to fix it, ideally with a doc link.
- **Pitfall:** Treating governance as static. **Best practice:** Version, test, and review policies like application code.

## Onboarding and Documentation Strategy

### Plain-language explanation

Onboarding is the journey from "I've heard about the platform" to "I shipped my first model." Documentation is the substrate that makes that journey self-serve. The goal is **time-to-first-value**: how quickly a new user gets a real win. Good documentation isn't one giant wiki; it's structured by intent — tutorials for learning, how-to guides for specific tasks, reference for lookup, and explanation for understanding (the Diátaxis framework). For a platform, docs are not an afterthought; they *are* the product's user interface for everything the CLI doesn't cover.

### Why it matters in production

A platform that requires a Slack message to the platform team for every question doesn't scale — the team drowns in support and users stay dependent. Documentation is how you scale yourself. In production, good onboarding also improves safety: users who understand the paved path are far less likely to build fragile workarounds. Time-to-first-value is a leading indicator of adoption; if a new data scientist can deploy a model in an afternoon by following a tutorial, they become an advocate. If they spend a week fighting the docs, they revert to whatever they used before.

### Real-world example / use case

Stripe is often cited as the gold standard for developer documentation, and internal platform teams increasingly borrow that bar. A strong pattern is a "golden onboarding tutorial": a single guided walkthrough that takes a new user from zero to a deployed, monitored toy model in under an hour, using the real paved-path tools. Pair that with docs-as-code (Markdown in the repo, reviewed in PRs, published via a static site generator like MkDocs or Docusaurus) so docs stay close to the code and can be updated in the same change that ships a feature.

### Code or config example

A docs-as-code setup and a structured onboarding checklist template:

```yaml
# mkdocs.yml — docs live in the repo, organized by Diátaxis intent
site_name: Nimbus ML Platform Docs
theme: { name: material }
nav:
  - Home: index.md
  - Tutorials:
      - "Deploy your first model (30 min)": tutorials/first-model.md
  - How-to guides:
      - "Promote a notebook to a job": how-to/notebook-to-job.md
      - "Add a canary rollout": how-to/canary.md
  - Reference:
      - "mlp CLI": reference/cli.md
      - "Serving manifest schema": reference/serving-schema.md
  - Explanation:
      - "Why paved paths": explanation/paved-paths.md
plugins: [search]
```

```markdown
<!-- onboarding-checklist.md — given to every new platform user -->
# Welcome to the Nimbus ML Platform

Goal: ship a monitored model today. Est. time: ~1 hour.

- [ ] Request access: run `mlp login` (SSO)
- [ ] Scaffold a project: `mlp init hello-model --template classification-service`
- [ ] Train the sample model: `mlp train --env cpu-small`
- [ ] Deploy to staging: `mlp deploy hello-model:latest --stage staging`
- [ ] Open your dashboard (link printed by the deploy command)
- [ ] Send a test request: `mlp predict hello-model --sample`
- [ ] Read: "Why paved paths" (5 min) to understand the model
- [ ] Join #ml-platform-support and introduce yourself

Stuck? Every `mlp` command supports `--help`, and errors link to docs.
```

### Interview questions

**Q: How would you structure documentation for an internal ML platform so people actually find what they need?**
A: I'd organize by user intent rather than by system component, following the Diátaxis model: tutorials for someone learning end-to-end, how-to guides for a specific task, reference for precise lookup, and explanation for the "why." Mixing these is the classic failure — a tutorial full of edge-case reference detail overwhelms beginners. I'd also treat docs-as-code so they're versioned and reviewed alongside features, and I'd invest heavily in one flagship onboarding tutorial that delivers a real win fast, because that first success drives adoption more than comprehensive reference ever will.

**Q: What metrics tell you your onboarding is working?**
A: The headline metric is time-to-first-value — median time from a new user's first login to their first successful deployment. I'd also watch onboarding completion rate (do people finish the tutorial or drop off, and where), the volume and category of support questions from new users (repeated questions signal a doc gap), and 30-day retention of new users. Rising support tickets on topics the docs supposedly cover is a strong signal the docs aren't discoverable or aren't clear.

### Pitfalls & best practices

- **Pitfall:** Docs treated as a one-time deliverable that rots as the platform evolves. **Best practice:** Docs-as-code, reviewed in the same PR as the change, with owners.
- **Pitfall:** One monolithic wiki mixing tutorials, reference, and rationale. **Best practice:** Structure by intent (Diátaxis); make search work.
- **Pitfall:** No fast win — onboarding buries users in concepts before any success. **Best practice:** A single tutorial to first value in under an hour.
- **Pitfall:** Support entirely in ephemeral Slack threads. **Best practice:** Convert recurring questions into docs and FAQs so answers compound.

## Summary

Owning a shared ML platform is fundamentally a product-and-people discipline layered on top of solid engineering. The chapter's arc reflects how mature platform teams operate:

- **Platform-as-a-product** reframes internal teams as customers you must win, measured by adoption, time-to-first-value, and internal NPS rather than feature counts. Earn adoption before you mandate it.
- **Self-service tooling** removes the platform team as a bottleneck by encoding paved paths into CLIs and generators, so the easy way is also the safe, observable way — and the notebook-to-production gap closes.
- **Golden paths and templates** (cookiecutter, Backstage) turn scaffolding into the place standards are enforced without meetings, while shared libraries and update tools keep generated projects from rotting.
- **Governance vs autonomy** is best resolved with guardrails (automated policy-as-code via OPA/Gatekeeper) as the default and gates reserved for genuinely high-risk decisions, rolled out through audit mode before enforcement.
- **Onboarding and documentation** scale the team by making the platform self-serve, structured by user intent, with a flagship tutorial that delivers a fast first win.

The connecting thread is that a platform succeeds when using it correctly is the path of least resistance. Every mechanism in this chapter — product metrics, paved paths, templates, guardrails, and great docs — exists to make the right thing the easy thing.

## Chapter Q&A

**Q1: What does "platform-as-a-product" mean, and how does it change how a platform team operates?**
A: It means treating internal engineers and data scientists as customers with a choice, rather than a captive audience. Operationally, the team gains a roadmap driven by user research, adoption and satisfaction metrics (NPS, time-to-first-value, percent of prod models on paved paths), and a bias toward validating demand before building. The team's success is measured by customer outcomes and velocity gains, not by shipping features nobody uses.

**Q2: Why is "build it and they will come" a dangerous assumption for a platform team?**
A: Because internal customers always have alternatives — building it themselves or adopting external SaaS. Without validated demand and low adoption friction, a technically impressive platform can see grudging or zero adoption, spawning shadow infrastructure. Product discovery, design partners, and instrumented usage guard against pouring effort into unwanted capabilities.

**Q3: Distinguish paved paths from mandates. Why prefer paved paths?**
A: A paved path makes the supported, secure, observable option the easiest one, so people choose it voluntarily; a mandate forces compliance. Paved paths align incentives and scale by attraction, whereas premature mandates create adversarial dynamics and workarounds. Mandates (or hard gates) are a late-stage tool for closing the long tail once the platform is genuinely the better choice.

**Q4: How do you bridge the notebook-to-production gap?**
A: Meet scientists in notebooks rather than banning them. Use tools like papermill to parameterize and version notebooks as reproducible jobs and nbconvert to extract clean modules, then provide a one-command promotion (`mlp notebook promote`) that runs the notebook in a controlled environment and wraps the model in the standard serving template. The platform absorbs the reproducibility burden so scientists keep their workflow.

**Q5: What is the biggest weakness of scaffolding templates, and how do you mitigate it?**
A: Templates are point-in-time copies, so generated projects diverge and go stale immediately. Mitigate by pushing behavior into versioned shared libraries and reusable CI workflows that projects reference (so improvements propagate automatically), and by using an update tool like `cruft` that tracks the source template commit and can raise automated update PRs. Track "template version in use" as a fleet metric to drive upgrades.

**Q6: Explain guardrails vs gates and give an ML-specific example of each.**
A: A guardrail is an automated preventive control that blocks unsafe actions while letting safe ones flow — e.g., an OPA/Gatekeeper policy that rejects any prod model service lacking a model-card and data-classification label. A gate is a manual approval checkpoint — e.g., deploying a model that touches regulated financial data requires sign-off from a risk committee. Default to guardrails for scale and velocity; reserve gates for high-risk, judgment-heavy decisions.

**Q7: What is policy-as-code and how do you roll out a new strict policy safely?**
A: Policy-as-code expresses governance rules as versioned, testable code (e.g., Rego evaluated by OPA/Gatekeeper) enforced automatically rather than via manual checklists — giving consistency, speed, auditability, and testability. To roll one out safely, start in audit/dry-run mode to measure impact without blocking, communicate the migration path and deadline, provide auto-fix tooling, enforce on new resources first while grandfathering existing ones, then move to full deny.

**Q8: How should platform documentation be structured, and why?**
A: By user intent, following Diátaxis: tutorials (learning), how-to guides (specific tasks), reference (lookup), and explanation (understanding). This prevents the common failure of mixing modes — like burying a beginner tutorial in edge-case reference. Combined with docs-as-code (versioned and reviewed with features) and a flagship onboarding tutorial that delivers value in under an hour, it maximizes discoverability and time-to-first-value.

**Q9: What metrics best indicate a platform's health and adoption?**
A: A north star like percent of production models on paved paths, plus leading indicators — time-to-first-deployment, weekly active users, internal NPS — and diagnostics like support tickets by category and template version distribution. Together these capture adoption, velocity, satisfaction, and where friction lives, and they tie the platform to business value.

**Q10: A regulated-industry stakeholder wants a manual approval gate on every model deployment. How do you respond?**
A: I'd acknowledge the legitimate risk while pushing back on blanket gates, which inject latency into every deploy and drive teams to route around the platform. I'd propose risk-tiering: automated guardrails (policy-as-code) enforce the non-negotiables on every deployment, and a manual gate fires only for the genuinely high-risk slice — e.g., models touching regulated data or with large blast radius. This satisfies compliance while preserving velocity for the majority, and it's auditable because the guardrails are versioned code.
