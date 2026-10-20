# Chapter 1 — Airflow Orchestration

Apache Airflow is the de facto standard for orchestrating data and machine-learning workflows. At its heart, Airflow lets you express pipelines as code: instead of chaining together cron jobs and brittle shell scripts, you declare *what* work has to happen, in *what order*, and let Airflow's scheduler handle *when* and its executors handle *where*. For MLOps this matters enormously, because a production ML system is rarely a single script — it is a sequence of interdependent steps (ingest, validate, featurize, train, evaluate, deploy, monitor) that must run reliably, retry on failure, be observable, and rerun cleanly over historical time windows.

This chapter builds from the ground up. We start with the mental model of a DAG and the scheduler/executor split, work through operators, scheduling semantics, and dependency management, then move into the advanced territory that separates hobby projects from production platforms: dynamic DAG generation, robust error handling, a complete ML training-to-deployment pipeline, and horizontal scaling with the Celery and Kubernetes executors. Every section pairs plain explanation with production reasoning, realistic code (Airflow 2.x, TaskFlow API where natural), interview questions, and hard-won pitfalls.

---

## 1. DAG Basics

### Plain-language explanation

A **DAG** (Directed Acyclic Graph) is Airflow's unit of workflow. "Directed" means work flows in one direction (task A then task B); "Acyclic" means there are no loops — you can never come back to a task you already ran within the same run. A DAG is made of **tasks**, and each task is an instance of an **operator** (a unit of work like "run this Python function" or "execute this SQL"). When a DAG runs, each task becomes a **task instance** tied to a specific logical date.

Three components make this go:

- **Scheduler** — parses your DAG files, decides which task instances are ready to run (based on schedule and dependencies), and puts them on a queue.
- **Executor** — takes queued tasks and actually runs them (locally, on a Celery worker pool, or as Kubernetes pods).
- **Metadata database** — the source of truth for state (which runs exist, which tasks succeeded/failed). The webserver reads from it to render the UI.

### Why it matters in production

The scheduler/executor separation is what lets Airflow scale and stay observable. Because the DAG is *code parsed into a graph*, Airflow always knows the full dependency structure, so it can retry a single failed task without rerunning upstream work, show you exactly where a pipeline broke, and enforce that downstream steps never run on stale or missing inputs. The metadata DB gives you an auditable history — critical for ML, where you must answer "which run produced the model currently in prod?"

### Real-world example / use case

A daily churn-scoring pipeline: pull yesterday's events from the warehouse, join with the feature store, run batch inference with the current model, and write scores back. Modeled as a four-task DAG, if the inference step fails at 3 a.m. because of an OOM, Airflow retries just that task; the expensive extraction upstream is not repeated, and the on-call engineer sees a red task in the UI with logs attached.

### Code or config example

```python
from datetime import datetime
from airflow.decorators import dag, task

@dag(
    dag_id="daily_churn_scoring",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "batch-inference"],
)
def churn_scoring():
    @task
    def extract() -> str:
        # returns a path/URI to extracted data
        return "s3://bucket/events/2024-01-01.parquet"

    @task
    def join_features(events_uri: str) -> str:
        return f"{events_uri}.joined"

    @task
    def score(features_uri: str) -> int:
        # run batch inference, return row count written
        return 12345

    # TaskFlow wires dependencies automatically from function calls
    features = join_features(extract())
    score(features)

churn_scoring()
```

The `@dag` and `@task` decorators are the **TaskFlow API** (Airflow 2.x). Return values are passed between tasks via XCom automatically, and the call graph *is* the dependency graph.

### Interview questions

**Q: What does "acyclic" guarantee, and why does Airflow require it?**
A: It guarantees the dependency graph has no cycles, so a topological ordering of tasks always exists. Airflow needs this to compute a deterministic execution order and to avoid infinite scheduling loops where task A waits on B which waits on A.

**Q: What is the difference between the scheduler and the executor?**
A: The scheduler decides *what* should run now — it evaluates schedules and dependencies and enqueues ready task instances. The executor is *how/where* those queued tasks actually run (LocalExecutor runs subprocesses, CeleryExecutor dispatches to a worker pool, KubernetesExecutor launches a pod per task). They are decoupled so you can scale execution independently.

**Q: Where does Airflow store task state, and why does that matter?**
A: In the metadata database (Postgres/MySQL). It matters because it is the single source of truth for run history, enables retries and backfills without recomputation, and provides the audit trail linking runs to artifacts — essential for ML reproducibility and debugging.

### Pitfalls & best practices

- **Top-level code runs on every parse.** The scheduler re-parses DAG files frequently (default every ~30s). Never do heavy work (DB queries, API calls, large imports) at module top level — put it inside tasks. Otherwise you slow the scheduler and can cause timeouts.
- **Keep DAG files import-light and deterministic.** A DAG file that occasionally throws during parsing will vanish from the UI intermittently.
- **One DAG object per file** is the clean default; generate multiple only with a deliberate factory pattern (see §5).

---

## 2. Operators and Sensors

### Plain-language explanation

An **operator** is a template for a single task. Airflow ships many: `PythonOperator` / `@task` (run a Python callable), `BashOperator` (run a shell command), `EmailOperator`, plus provider operators like `SnowflakeOperator`, `S3ToRedshiftOperator`, `KubernetesPodOperator`, and many more. You instantiate an operator to create a task.

A **sensor** is a special operator that *waits* for a condition to become true — a file to land in S3, a partition to appear in a table, an external DAG to finish. Sensors are how Airflow handles "don't start until X is ready."

Sensors have two **modes**:
- `poke` — holds a worker slot and checks repeatedly (simple, but occupies a slot the whole time).
- `reschedule` — releases the worker slot between checks and wakes up later (far more efficient for long waits).

### Why it matters in production

Operators give you reusable, well-tested integrations instead of hand-rolled boto3/psycopg2 code in every DAG. Sensors let pipelines react to external readiness without busy-waiting or fragile sleep loops. Choosing `reschedule` mode for long waits is a real cost/stability lever: `poke` sensors waiting hours can exhaust your worker pool and deadlock the whole cluster (the classic "sensor deadlock").

### Real-world example / use case

An upstream data team drops a daily `SUCCESS` marker file to S3 whenever their export finishes — anywhere between 1 a.m. and 5 a.m. Your training pipeline uses an `S3KeySensor` in `reschedule` mode to wait for that marker, then proceeds. No polling scripts, no guessing the arrival time, and no wasted worker slots during the wait.

### Code or config example

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="wait_then_process",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    wait_for_export = S3KeySensor(
        task_id="wait_for_export",
        bucket_name="data-lake",
        bucket_key="exports/{{ ds }}/_SUCCESS",
        mode="reschedule",          # free the worker slot between checks
        poke_interval=300,          # check every 5 min
        timeout=6 * 60 * 60,        # give up after 6 hours
    )

    process = BashOperator(
        task_id="process",
        bash_command="python /opt/jobs/process.py --date {{ ds }}",
    )

    wait_for_export >> process
```

`{{ ds }}` is a Jinja **template** — Airflow renders the run's logical date (YYYY-MM-DD) at execution time.

### Interview questions

**Q: What is the difference between poke and reschedule mode, and when do you use each?**
A: In `poke` mode the sensor holds a worker slot for its entire lifetime, sleeping between checks. In `reschedule` mode it releases the slot after each check and is re-queued later. Use `poke` for short waits (seconds); use `reschedule` for long or uncertain waits (minutes to hours) to avoid occupying worker slots and risking sensor deadlock.

**Q: Why prefer a provider operator over calling an SDK inside a PythonOperator?**
A: Provider operators handle connection management, retries, templating, and hooks consistently; they are tested and maintained by the community; and they keep credentials in Airflow Connections rather than hard-coded in code. It reduces boilerplate and standardizes behavior across DAGs.

**Q: What is a deferrable operator?**
A: A deferrable operator (Airflow 2.2+) offloads its wait to the **triggerer** process using async triggers, so it consumes essentially no worker or scheduler slot while waiting. It is the most efficient option for long waits — even better than reschedule sensors — because the polling happens in an async event loop.

### Pitfalls & best practices

- **Always set a `timeout` on sensors.** A sensor with no timeout can wait forever and mask upstream failures.
- **Prefer `reschedule` or deferrable** for anything longer than a minute.
- **Don't overload one PythonOperator** with an entire pipeline's logic — you lose granular retries and visibility. Split into meaningful tasks.
- **Idempotency:** operators should be safe to rerun. Design writes as upserts or partition-overwrites keyed on the logical date.

---

## 3. Scheduling

### Plain-language explanation

A DAG's schedule tells Airflow how often to create runs. You can use presets (`@daily`, `@hourly`), cron strings (`"0 6 * * *"`), a `timedelta`, or a **Timetable** for complex calendars. Airflow 2.x renamed the confusing `execution_date` to **`logical_date`** — the *start* of the data interval the run represents, not the wall-clock time it actually fires.

Key mental model: **a `@daily` run for 2024-01-01 fires at the end of that interval, i.e. at 2024-01-02 00:00.** The run "covers" the data of Jan 1. That is why the logical date lags real time by one interval.

- **`catchup`** — if `True` (historically the default), Airflow will create runs for every missed interval between `start_date` and now. If `False`, it only runs the latest.
- **`backfill`** — deliberately (re)running a DAG over a past date range, via `airflow dags backfill`.
- **Data interval** — each run has `data_interval_start` and `data_interval_end`; use these to slice data deterministically.

### Why it matters in production

Misunderstanding logical_date vs. wall-clock is the single most common Airflow bug: people query "today" with `datetime.now()` inside a task and get non-reproducible, non-backfillable pipelines. Using the logical date / data interval makes runs **idempotent and backfillable** — you can rerun Jan 1 next month and get the exact same result. `catchup` behavior determines whether deploying a DAG with an old `start_date` triggers a stampede of hundreds of historical runs — a frequent production incident.

### Real-world example / use case

You deploy a new feature-engineering DAG with `start_date=datetime(2023, 1, 1)` and forget `catchup=False`. On activation the scheduler tries to create ~380 daily runs at once, saturating workers and the warehouse. With `catchup=False`, only the latest interval runs, and you backfill history intentionally with a controlled, rate-limited command.

### Code or config example

```python
from datetime import datetime
from airflow.decorators import dag, task

@dag(
    dag_id="feature_engineering",
    schedule="0 6 * * *",              # 6 AM cron
    start_date=datetime(2024, 1, 1),
    catchup=False,                     # do NOT stampede historical runs
    max_active_runs=1,                 # serialize runs to protect the warehouse
    tags=["features"],
)
def feature_engineering():
    @task
    def build_features(data_interval_start=None, data_interval_end=None):
        # Use the interval, NOT datetime.now(), for reproducibility
        print(f"Processing window {data_interval_start} -> {data_interval_end}")
        # SELECT ... WHERE event_ts >= data_interval_start
        #           AND event_ts <  data_interval_end
        return "ok"

    build_features()

feature_engineering()
```

Controlled backfill from the CLI:

```bash
airflow dags backfill \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --reset-dagruns \
  feature_engineering
```

### Interview questions

**Q: Why does a `@daily` DAG for logical date 2024-01-01 actually run on 2024-01-02?**
A: Because Airflow schedules a run at the *end* of the data interval it represents. The 2024-01-01 run covers all of Jan 1's data, which is only complete at 2024-01-02 00:00, so that is when it fires. The logical_date marks the interval start.

**Q: What is the difference between catchup and backfill?**
A: `catchup` is an automatic behavior — when a DAG is enabled, the scheduler creates runs for all missed intervals since `start_date` (unless disabled). `backfill` is a manual, deliberate CLI operation to run a DAG over a specified historical date range. Catchup happens by policy; backfill is on demand.

**Q: How do you make a scheduled task reproducible across reruns?**
A: Never use wall-clock time inside tasks. Derive all date logic from `logical_date` / `data_interval_start` / `data_interval_end`, and make writes idempotent (partition overwrite or upsert keyed on the interval). Then any rerun of a given date produces identical output.

### Pitfalls & best practices

- **Default `catchup=False`** unless you explicitly want historical fill.
- **Set `max_active_runs`** to protect downstream systems from concurrent-run overload during backfills.
- **Never reference `datetime.now()`** for business logic — use interval macros (`{{ ds }}`, `{{ data_interval_start }}`).
- **Don't change `start_date` on a live DAG**; create a new DAG version/id instead — mutating it corrupts the schedule.

---

## 4. Dependency Management

### Plain-language explanation

Dependencies define task order. The classic API uses `a.set_downstream(b)` / `b.set_upstream(a)`; the readable modern form is the bitshift operator: `a >> b` (a then b) and `a << b` (a after b). With TaskFlow, passing one task's output into another creates the dependency implicitly.

For structure and scale:
- **TaskGroups** visually and logically group related tasks (replacing the old SubDAGs, which are deprecated).
- **Cross-DAG dependencies** connect separate DAGs. Two main tools:
  - **`TriggerDagRunOperator`** — DAG A actively *triggers* DAG B (push model).
  - **`ExternalTaskSensor`** — DAG B *waits* for a task/DAG in DAG A to complete (pull model).

### Why it matters in production

Clear dependencies are what make Airflow more than cron. TaskGroups keep large ML DAGs (dozens of tasks) readable and maintainable. Cross-DAG dependencies let you decompose a monolith into independently owned, independently scheduled pipelines — e.g. a feature pipeline owned by the data team and a training pipeline owned by the ML team — while still enforcing correct ordering.

### Real-world example / use case

The data team's `feature_pipeline` runs daily and, on success, triggers the ML team's `retraining_pipeline` via `TriggerDagRunOperator`. Alternatively, if the ML team wants to stay decoupled and pull rather than be pushed, their DAG opens with an `ExternalTaskSensor` waiting for `feature_pipeline`'s final task to succeed for the same logical date.

### Code or config example

```python
from datetime import datetime
from airflow.decorators import dag, task_group, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor

@dag(dag_id="retraining_pipeline", schedule="@daily",
     start_date=datetime(2024, 1, 1), catchup=False)
def retraining():

    # Pull model: wait for the upstream feature DAG's final task
    wait_for_features = ExternalTaskSensor(
        task_id="wait_for_features",
        external_dag_id="feature_pipeline",
        external_task_id="publish_features",
        allowed_states=["success"],
        mode="reschedule",
        timeout=3 * 60 * 60,
    )

    @task_group(group_id="preprocessing")
    def preprocessing():
        @task
        def clean(): return "clean"
        @task
        def split(x): return f"split:{x}"
        split(clean())

    @task
    def train(): return "model.pkl"

    # Push model: kick off a downstream deployment DAG
    trigger_deploy = TriggerDagRunOperator(
        task_id="trigger_deploy",
        trigger_dag_id="deploy_pipeline",
        wait_for_completion=False,
    )

    wait_for_features >> preprocessing() >> train() >> trigger_deploy

retraining()
```

### Interview questions

**Q: When would you use TriggerDagRunOperator vs. ExternalTaskSensor?**
A: Use `TriggerDagRunOperator` when the upstream DAG owns the relationship and should actively launch the downstream one (push, tight coupling, immediate handoff). Use `ExternalTaskSensor` when the downstream DAG should independently wait for an upstream result (pull, loose coupling, downstream team owns the dependency). Push is simpler; pull keeps teams decoupled.

**Q: What replaced SubDAGs and why?**
A: TaskGroups. SubDAGs spawned a separate DAG run with its own scheduling, caused deadlocks (they held a worker slot while their child tasks ran, often on a different executor), and were a frequent source of bugs. TaskGroups are pure UI/logical grouping with no runtime overhead or separate run.

**Q: A common gotcha with ExternalTaskSensor — what date does it check?**
A: By default it looks for the upstream task at the *same logical_date* as the current DAG run. If the two DAGs have different schedules, you must set `execution_date_fn` or `execution_delta` to align the intervals, or the sensor will wait on a run that never existed.

### Pitfalls & best practices

- **Align logical dates** when using ExternalTaskSensor across DAGs with different schedules (`execution_delta` / `execution_delta_fn`).
- **Prefer TaskGroups over SubDAGs** — SubDAGs are deprecated and deadlock-prone.
- **Avoid deep cross-DAG chains** — they create hidden coupling and hard-to-debug cascades; consider Airflow **Datasets** (data-aware scheduling) for a cleaner decoupled trigger.
- **Set `wait_for_completion` deliberately** on TriggerDagRunOperator — leaving it `True` holds a slot until the child finishes.

---

## 5. Dynamic DAGs

### Plain-language explanation

Sometimes you don't know the exact set of tasks or DAGs until parse time or run time. Two techniques:

- **Factory pattern (dynamic DAG generation):** a Python function that builds and returns DAG objects, called in a loop over a config (e.g. one DAG per client, per region, per table). The *number of DAGs/tasks* is decided at **parse time**.
- **Dynamic Task Mapping (`.expand()`):** introduced in Airflow 2.3, this creates a variable number of parallel task instances at **run time**, based on the output of an upstream task — the map/reduce pattern. You don't know the count until the pipeline runs.

### Why it matters in production

ML platforms are inherently multiplicative: N models, M datasets, K hyperparameter sets, per-partition processing. Hand-writing a DAG per combination is unmaintainable. Factory patterns generate them from a single source of truth (a YAML/DB config), so onboarding a new model is a config change, not a code change. Dynamic task mapping lets you fan out over "however many files landed today" or "however many hyperparameter configs" without knowing the count in advance — impossible with static task definitions.

### Real-world example / use case

- **Factory:** A config lists 40 tenants, each needing an identical daily scoring pipeline. One factory function loops over the config and registers 40 DAGs. Adding tenant 41 is a one-line config edit.
- **Dynamic mapping:** A tuning DAG reads a list of 12 hyperparameter dicts from an upstream task and `.expand()`s a `train_one` task into 12 parallel training runs, then a reduce task picks the best.

### Code or config example

Factory pattern:

```python
# dags/tenant_factory.py
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

TENANTS = ["acme", "globex", "initech"]  # in practice: load from YAML/DB

def make_dag(tenant: str) -> DAG:
    with DAG(
        dag_id=f"score_{tenant}",
        schedule="@daily",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["tenant", tenant],
    ) as dag:
        BashOperator(
            task_id="score",
            bash_command=f"python score.py --tenant {tenant} --date {{{{ ds }}}}",
        )
    return dag

# Register each DAG in the module's global namespace so the scheduler finds it
for t in TENANTS:
    globals()[f"score_{t}"] = make_dag(t)
```

Dynamic task mapping:

```python
from datetime import datetime
from airflow.decorators import dag, task

@dag(dag_id="hpo_sweep", schedule=None,
     start_date=datetime(2024, 1, 1), catchup=False)
def hpo_sweep():
    @task
    def get_param_grid() -> list[dict]:
        return [{"lr": 0.1}, {"lr": 0.01}, {"lr": 0.001}]

    @task
    def train_one(params: dict) -> dict:
        # train with these params, return a metric
        return {"params": params, "auc": 0.9}

    @task
    def pick_best(results: list[dict]) -> dict:
        return max(results, key=lambda r: r["auc"])

    results = train_one.expand(params=get_param_grid())  # fan out at runtime
    pick_best(results)                                    # fan in (reduce)

hpo_sweep()
```

### Interview questions

**Q: What is the difference between the factory pattern and dynamic task mapping?**
A: The factory pattern decides the number of *DAGs or tasks* at **parse time** by generating them in Python from config — the graph is fixed once parsed. Dynamic task mapping (`.expand()`) decides the number of *task instances* at **run time** based on an upstream task's output — the fan-out count can differ every run.

**Q: What are the risks of dynamic DAG generation via factory?**
A: Parse-time cost (too many/too-heavy DAGs slow the scheduler), non-determinism if the config source is slow or flaky at parse time, and debugging difficulty because the DAG doesn't exist as literal code. Best practice: keep generation logic cheap and deterministic, and cache/version the config.

**Q: Can you nest or chain dynamic task mapping?**
A: Yes — you can `.expand()` over the output of another mapped task, and combine with `.partial()` to fix constant arguments while mapping others. Airflow also supports mapping over multiple arguments (cross-product) and reducing mapped results into a single downstream task.

### Pitfalls & best practices

- **Cap the fan-out** with `max_active_tis_per_dag` or pool limits — an upstream returning 10,000 items can create 10,000 concurrent tasks and melt the cluster.
- **Keep factory config loading cheap and safe** — a DB call at top level that times out will drop all generated DAGs from the UI.
- **Use `.partial()`** to pass constants to mapped tasks instead of baking them into the mapped list.
- **Prefer dynamic mapping over factory** when the count varies per run; prefer factory when the set is stable and config-driven.

---

## 6. Error Handling & Retries

### Plain-language explanation

Tasks fail — networks blip, warehouses throttle, spot instances vanish. Airflow gives you layered controls:

- **`retries`** — how many times to re-attempt a failed task.
- **`retry_delay`** — wait between attempts.
- **`retry_exponential_backoff=True`** — grow the delay geometrically (delay, 2×, 4×, …) up to `max_retry_delay`.
- **`on_failure_callback` / `on_retry_callback` / `on_success_callback`** — hooks to run custom code (send a Slack/PagerDuty alert) on state changes.
- **`execution_timeout`** — kill a task that runs too long.
- **SLAs** (`sla=timedelta(...)`) — flag when a task doesn't finish within an expected window (SLA misses are recorded and can trigger `sla_miss_callback`).

### Why it matters in production

Retries with backoff absorb transient failures automatically, so a warehouse hiccup at 3 a.m. doesn't page anyone. Callbacks route *real* failures to the right channel with context. SLAs surface silent slowness — a training DAG that "succeeds" but now takes 9 hours instead of 2 is a problem you want flagged before it delays deployment. Together these turn a fragile pipeline into an operable one.

### Real-world example / use case

A training task talks to a rate-limited GPU cluster. You set `retries=3` with exponential backoff (2 min, 4 min, 8 min) to ride out throttling. An `on_failure_callback` posts to Slack with the DAG id, task id, logical date, and a direct link to the log. An `sla=timedelta(hours=3)` on the training task alerts the team if training runs long, before it blocks the downstream deploy.

### Code or config example

```python
from datetime import datetime, timedelta
from airflow.decorators import dag, task

def slack_alert(context):
    ti = context["task_instance"]
    msg = (f":red_circle: {ti.dag_id}.{ti.task_id} failed "
           f"for {context['logical_date']}\nLog: {ti.log_url}")
    # send_to_slack(msg)  # integrate your notifier here
    print(msg)

default_args = {
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=20),
    "on_failure_callback": slack_alert,
    "execution_timeout": timedelta(hours=4),
}

@dag(
    dag_id="robust_training",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
)
def robust_training():
    @task(sla=timedelta(hours=3))     # flag if training runs long
    def train():
        ...
    train()

robust_training()
```

### Interview questions

**Q: Why use exponential backoff instead of a fixed retry delay?**
A: Fixed delays can hammer an already-struggling dependency in lockstep (retry storms). Exponential backoff spreads retries out over increasing intervals, giving a throttled or recovering service time to heal, and reduces the chance that many tasks retry simultaneously.

**Q: What is the difference between an SLA and execution_timeout?**
A: `execution_timeout` actively **kills** a task that runs longer than the limit (it fails). An SLA is a soft **expectation**: if a task isn't done within the SLA window, Airflow records an SLA miss and can fire `sla_miss_callback`, but it does *not* stop the task. Timeout enforces; SLA alerts.

**Q: How do you send a rich failure alert with context?**
A: Use `on_failure_callback`, which receives the `context` dict containing `task_instance` (with `log_url`, `dag_id`, `task_id`, `try_number`) and `logical_date`. Build a message from those fields and push it to Slack/PagerDuty. Set it in `default_args` to apply cluster-wide.

### Pitfalls & best practices

- **Only retry idempotent tasks.** Retrying a non-idempotent write can double-insert data. Make tasks safe to rerun first.
- **Don't set huge `retries` to mask real bugs** — retries are for *transient* failures, not logic errors.
- **Cap backoff** with `max_retry_delay`, or delays can grow to days.
- **SLAs are per-task and tied to schedule** — they don't work well on manually triggered or `schedule=None` DAGs; prefer explicit timeout + alerting there.
- **Alert fatigue:** route to the right severity. Not every retry needs a page; use `on_retry_callback` sparingly.

---

## 7. Airflow for ML Pipeline Orchestration

### Plain-language explanation

An ML pipeline is a DAG like any other, but with domain-specific stages and gates: **ingest → validate data → featurize → train → evaluate → (quality gate) → register → deploy → smoke test**. The distinctive part is the **conditional gate**: you only promote a model if it beats a baseline / passes validation thresholds. Airflow expresses this with **branching** (`@task.branch` / `BranchPythonOperator`) so the DAG deploys on pass and alerts/stops on fail.

### Why it matters in production

Automating the full train-to-deploy loop with a hard quality gate is the core of continuous training (CT) in MLOps. It prevents a worse model from silently replacing a good one, records every run's metrics and artifacts for lineage, and makes retraining a routine, auditable, one-click (or fully scheduled) operation rather than a risky manual ritual. XCom carries small artifacts (paths, metrics); large artifacts (models) live in object storage / a model registry, with only URIs passed between tasks.

### Real-world example / use case

Nightly retraining: pull the last 90 days of labeled data, run Great Expectations-style validation, engineer features, train a gradient-boosted model, evaluate AUC on a holdout. If AUC ≥ the current production model's AUC + margin, register the model in MLflow and trigger a canary deployment; otherwise, skip deploy and notify the ML team that the challenger did not beat the champion.

### Code or config example

```python
from datetime import datetime, timedelta
from airflow.decorators import dag, task

default_args = {"retries": 2, "retry_delay": timedelta(minutes=5)}

@dag(
    dag_id="ml_training_pipeline",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ml", "continuous-training"],
)
def ml_training_pipeline():

    @task
    def ingest(data_interval_end=None) -> str:
        return "s3://lake/training/latest.parquet"

    @task
    def validate_data(uri: str) -> str:
        # run data-quality checks; raise to fail the run if invalid
        assert uri, "no data"
        return uri

    @task
    def featurize(uri: str) -> str:
        return uri + ".features"

    @task
    def train(features_uri: str) -> dict:
        # train, log to MLflow, return artifact URI + metrics
        return {"model_uri": "s3://models/candidate.pkl", "auc": 0.88}

    @task
    def evaluate(model: dict) -> dict:
        prod_auc = 0.85  # fetch current production model's AUC
        model["passed"] = model["auc"] >= prod_auc + 0.005
        return model

    @task.branch
    def gate(model: dict) -> str:
        return "deploy" if model["passed"] else "notify_no_promo"

    @task
    def deploy(model: dict) -> str:
        # register in model registry + trigger canary rollout
        return f"deployed {model['model_uri']}"

    @task
    def notify_no_promo(model: dict):
        print(f"Challenger AUC {model['auc']} did not beat champion; skipping deploy")

    data = validate_data(ingest())
    model = evaluate(train(featurize(data)))
    branch = gate(model)
    branch >> [deploy(model), notify_no_promo(model)]

ml_training_pipeline()
```

### Interview questions

**Q: How do you implement a model quality gate in Airflow?**
A: Add an evaluate task that computes challenger metrics and compares against the incumbent (with a margin), then a branch task (`@task.branch` / `BranchPythonOperator`) that returns the id of the `deploy` path or the `skip/notify` path. Only the returned branch runs; the rest are skipped. This enforces "promote only if better."

**Q: Should you pass a trained model through XCom?**
A: No. XCom is backed by the metadata DB and meant for small values (paths, metrics, flags). Store large artifacts (models, datasets) in object storage or a model registry (MLflow, S3) and pass only the URI/reference through XCom. Passing big blobs bloats the DB and can break serialization.

**Q: How do you ensure reproducibility of an ML training run?**
A: Pin the data window via `logical_date`/data interval, version data and code (dataset snapshot + git SHA), log params/metrics/artifacts to a registry keyed by the run, and make each task idempotent. Then any run — including a backfill — can be traced to exact inputs and reproduced.

### Pitfalls & best practices

- **Keep artifacts out of XCom;** pass URIs.
- **Use branching for gates,** and mind `trigger_rule` on join tasks after a branch (skipped branches propagate skips; use `trigger_rule="none_failed_min_one_success"` where needed).
- **Separate concerns:** consider a training DAG and a deployment DAG linked by `TriggerDagRunOperator` or Datasets, so retraining and rollout can be owned/operated independently.
- **Log lineage** (data version, code SHA, metrics) at every stage for auditability.
- **Don't put heavy compute in the scheduler's process** — offload training to KubernetesPodOperator or an external compute service (see §8).

---

## 8. Scaling with Celery and Kubernetes Executors

### Plain-language explanation

The **executor** determines where tasks run. Beyond `SequentialExecutor` (SQLite, testing) and `LocalExecutor` (subprocesses on one machine), production uses:

- **CeleryExecutor** — a fixed pool of long-running worker processes pulls tasks from a message broker (Redis/RabbitMQ). Workers are always on, so task startup is fast, but you provision and pay for the pool continuously.
- **KubernetesExecutor** — Airflow launches a **new pod per task**, then tears it down. No standing worker pool; each task can request its own CPU/memory/GPU and image. Great isolation and elasticity, but per-task pod startup latency.
- **CeleryKubernetesExecutor** — hybrid: route short/light tasks to Celery workers and heavy/isolated tasks to Kubernetes, via queues.

Separately, **`KubernetesPodOperator`** runs a single task as a K8s pod *regardless of which executor you use* — handy for running a containerized training job with custom dependencies on top of, say, a CeleryExecutor cluster.

### Why it matters in production

Executor choice is the main scaling and cost lever. Celery gives low latency and steady throughput for many small, homogeneous tasks (typical ELT). Kubernetes gives per-task resource isolation and elasticity — essential for ML, where one task needs a GPU and 64 GB RAM while another needs a tiny CPU pod, and where dependency conflicts between models demand different container images. Getting this wrong means either a saturated Celery pool that queues for hours, or a Kubernetes cluster paying pod-startup tax on thousands of trivial tasks.

### Real-world example / use case

A platform runs hundreds of small SQL/transform tasks (perfect for Celery's warm workers) plus a nightly GPU training job. Rather than sizing every Celery worker with a GPU, they use `KubernetesPodOperator` for the training task: it spins up a GPU pod from a CUDA image, trains, writes the model to S3, and exits — GPU billed only during training. Small tasks stay on Celery for speed.

### Code or config example

`KubernetesPodOperator` for a GPU training task:

```python
from datetime import datetime
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

with DAG("gpu_training", schedule="@daily",
         start_date=datetime(2024, 1, 1), catchup=False) as dag:

    train = KubernetesPodOperator(
        task_id="train_model",
        name="train-model",
        namespace="ml-jobs",
        image="myregistry/trainer:1.4.0",     # custom image w/ CUDA + deps
        cmds=["python", "train.py"],
        arguments=["--date", "{{ ds }}", "--out", "s3://models/{{ ds }}/"],
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "1"},
            limits={"cpu": "8", "memory": "64Gi", "nvidia.com/gpu": "1"},
        ),
        get_logs=True,
        is_delete_operator_pod=True,           # clean up the pod when done
    )
```

Executor config sketch (`airflow.cfg` / Helm values):

```yaml
# Helm values for the official Airflow chart
executor: "CeleryKubernetesExecutor"

workers:
  replicas: 6            # Celery worker pool for light tasks

redis:
  enabled: true          # Celery broker

config:
  celery:
    worker_concurrency: 16
  kubernetes_executor:
    namespace: airflow-tasks
    delete_worker_pods: true
```

Routing a heavy task to Kubernetes under the hybrid executor is as simple as `queue="kubernetes"` on that task.

### Interview questions

**Q: Compare CeleryExecutor and KubernetesExecutor — when do you pick each?**
A: Celery uses a persistent worker pool behind a broker: low per-task latency, efficient for many small homogeneous tasks, but fixed capacity you pay for continuously and coarse resource isolation. Kubernetes launches one pod per task: perfect resource isolation, per-task images/GPUs, elastic scale-to-zero, but pod-startup latency per task. Choose Celery for high-volume light ELT with steady load; choose Kubernetes for heterogeneous, resource-hungry, or isolation-sensitive ML workloads. Use the hybrid to get both.

**Q: What's the difference between KubernetesExecutor and KubernetesPodOperator?**
A: KubernetesExecutor is a cluster-wide *executor* that runs *every* task as its own pod automatically. KubernetesPodOperator is a single *operator* you place on specific tasks to run them as pods, independent of the executor — so you can run most tasks on Celery/Local and only certain tasks (e.g. GPU training) as custom pods.

**Q: What are the trade-offs of one-pod-per-task?**
A: Pros: isolation, custom resources/images per task, no noisy-neighbor, elastic cost. Cons: pod scheduling/startup latency (seconds to tens of seconds) that hurts thousands of tiny tasks, image-pull overhead, and more moving parts (cluster autoscaler tuning, resource quotas). Mitigate with warm node pools, image caching, and the hybrid executor.

### Pitfalls & best practices

- **Match executor to workload shape:** don't run swarms of sub-second tasks on KubernetesExecutor (startup tax dominates), and don't force GPU/heterogeneous ML onto a uniform Celery pool.
- **Set resource requests/limits** on pods explicitly; unset requests cause scheduling chaos and OOM kills.
- **`is_delete_operator_pod=True`** (or `delete_worker_pods=True`) to avoid pod/log buildup.
- **Use pools and `worker_concurrency`** to cap parallelism against downstream limits (warehouse connections, API quotas).
- **Keep the scheduler light:** never run compute on the scheduler; offload to workers/pods.
- **Prefer KubernetesPodOperator on a Celery cluster** as a pragmatic middle ground before migrating the whole cluster to KubernetesExecutor.

---

## Summary

Airflow orchestrates workflows as **DAGs** — directed acyclic graphs of tasks — with a clean separation between the **scheduler** (decides what runs), the **executor** (runs it), and the **metadata database** (records state). That architecture is what makes Airflow reliable and observable: it can retry a single task without recomputing upstream work, render exactly where a pipeline broke, and reproduce any historical run. Tasks are built from **operators**, and **sensors** wait for external readiness — using `reschedule` or deferrable modes to avoid hogging worker slots.

Scheduling in Airflow 2.x is interval-based: a run's **`logical_date`** marks the *start* of the data interval it covers and fires at the interval's *end*. Treating that logical date (not wall-clock time) as the source of truth is the key to **idempotent, backfillable** pipelines, and disabling `catchup` by default prevents accidental historical stampedes. Dependencies are expressed with `>>`, organized with **TaskGroups**, and connected across pipelines with **TriggerDagRunOperator** (push) or **ExternalTaskSensor** (pull) — or, more cleanly, with Datasets.

At scale, **dynamic DAGs** (factory pattern at parse time, `.expand()` dynamic task mapping at run time) eliminate copy-paste and handle variable fan-out; **layered error handling** (retries with exponential backoff, callbacks, timeouts, SLAs) turns fragile jobs into operable ones. For ML specifically, a full **train → validate → evaluate → gate → deploy** DAG with a branching quality gate implements continuous training, keeping large artifacts in a registry and passing only URIs through XCom. Finally, **executor choice** is the decisive scaling lever: **Celery** for high-volume light tasks with warm workers, **Kubernetes** for heterogeneous, resource-hungry, isolated ML workloads, and the **hybrid** (or a targeted `KubernetesPodOperator`) to get the best of both.

## Chapter Q&A

**Q1. Explain the scheduler/executor/metadata-DB architecture and why the separation matters.**
The scheduler parses DAGs and enqueues task instances whose dependencies and schedule are satisfied. The executor pulls from that queue and runs tasks (locally, on Celery workers, or as K8s pods). The metadata DB stores all state. Separation lets you scale execution independently of scheduling, retry individual tasks using recorded state without recomputation, and maintain a full audit history — essential for reproducibility and debugging in production ML.

**Q2. A colleague deploys a DAG with `start_date` a year ago and the cluster grinds to a halt. What happened and how do you fix it?**
`catchup` was left enabled, so the scheduler created a run for every missed interval (~365) at once, saturating workers and downstream systems. Fix: set `catchup=False`, then backfill the needed history deliberately with `airflow dags backfill` over a controlled range, and set `max_active_runs` to throttle concurrency.

**Q3. Why is using `datetime.now()` inside a task an anti-pattern?**
It makes runs non-reproducible and non-backfillable — rerunning a past date would use *today's* time instead of the intended window. Use `logical_date` / `data_interval_start` / `data_interval_end` (or `{{ ds }}`) so each run deterministically processes its own interval and any rerun produces identical output.

**Q4. You need to wait up to 6 hours for an upstream file. How do you implement this efficiently?**
Use a sensor with `mode="reschedule"` (or a deferrable sensor) so it releases the worker slot between checks instead of holding it for six hours. Always set a `timeout` and a reasonable `poke_interval`. A poke-mode sensor here risks exhausting the worker pool and causing sensor deadlock.

**Q5. Design a nightly retraining pipeline that only deploys a model if it beats production.**
DAG: ingest → validate_data → featurize → train (log to registry) → evaluate (compare challenger vs. champion with a margin) → `@task.branch` gate → deploy *or* notify. Store the model in a registry/object store and pass only its URI via XCom. Add retries with backoff, an SLA on training, and an `on_failure_callback` for alerting. This is a continuous-training loop with a hard quality gate.

**Q6. Compare TriggerDagRunOperator and ExternalTaskSensor with a concrete scenario.**
If the feature-pipeline team wants to actively launch retraining when features are ready, they add a `TriggerDagRunOperator` (push, tight coupling). If the ML team prefers to stay decoupled and start only when features exist, they open their DAG with an `ExternalTaskSensor` waiting on the feature DAG's final task (pull). With differing schedules, the sensor needs `execution_delta`/`execution_date_fn` to align logical dates.

**Q7. When would you choose Kubernetes over Celery executor for an ML platform?**
When workloads are heterogeneous and resource-hungry — one task needs a GPU and 64 GB RAM, another a tiny CPU pod; when tasks need different container images (conflicting model dependencies); and when you want elastic scale-to-zero cost. Celery's fixed, uniform worker pool can't isolate resources per task and would over- or under-provision. For mostly light, high-volume tasks with a few heavy jobs, use the hybrid executor or a `KubernetesPodOperator` on a Celery cluster.

**Q8. How do you handle transient failures without paging on every blip, but still catch real problems and slow runs?**
Set `retries` with `retry_exponential_backoff` (capped by `max_retry_delay`) so transient failures self-heal quietly; use `on_retry_callback` sparingly (or not at all) to avoid noise, and `on_failure_callback` to page with rich context only when all retries are exhausted. Add `execution_timeout` to kill hung tasks and an `sla` to flag runs that succeed but are abnormally slow. Ensure tasks are idempotent so retries are safe.

**Q9. You have to run the same pipeline for 200 tenants. Static or dynamic, and how?**
Dynamic. Use a **factory pattern**: store the tenant list in a config (YAML/DB), loop over it in a DAG file, and register one DAG per tenant in `globals()`. Onboarding a new tenant becomes a config change. Keep the config load cheap and deterministic to avoid slowing the scheduler or dropping DAGs on parse errors. If the *count of items within a run* varies (e.g., files per day), use dynamic task mapping (`.expand()`) instead.

**Q10. What's the right way to run a heavy, dependency-conflicting training step inside an otherwise Celery-based cluster?**
Use `KubernetesPodOperator` for that one task. It launches a dedicated pod from a custom image with the exact dependencies and resources (GPU, memory) it needs, runs the job, writes artifacts to object storage, and tears the pod down (`is_delete_operator_pod=True`) — all without disturbing the warm Celery workers handling your light tasks. It gives Kubernetes-style isolation for select tasks without migrating the whole cluster to the KubernetesExecutor.
