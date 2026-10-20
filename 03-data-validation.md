# Chapter 3 — Data Validation Frameworks

Machine learning systems fail silently. A model can keep serving predictions with perfect uptime while its inputs quietly rot — a upstream team renames a column, a currency field switches from cents to dollars, a nullable field starts arriving 40% null, or a categorical feature gains a value the model has never seen. None of these throw exceptions. They just degrade accuracy until someone notices the business metric slipping weeks later. Data validation is the discipline of catching these problems at the boundary, before bad data reaches training jobs or inference servers.

This chapter builds from the fundamentals of schema validation up through statistical drift detection, anomaly detection, and the operational practices that make validation a first-class part of your pipelines: failing CI builds on bad data and enforcing data contracts between the teams that produce data and the teams that consume it. The two workhorse open-source frameworks — Great Expectations and TensorFlow Data Validation — appear throughout, and we are careful to use their real APIs so the examples are near-runnable.

## Schema Validation Basics

### Plain-language explanation

A schema is a formal description of what your data should look like: which columns exist, what type each column holds (integer, float, string, timestamp), whether a column is allowed to be null, what range numeric values may fall in, and which discrete values a categorical column may contain (its "domain"). Schema validation is the act of checking incoming data against that description and rejecting or flagging anything that violates it. It is the cheapest, highest-leverage form of data validation because most catastrophic data bugs are actually schema violations in disguise.

The four pillars are:

- **Types** — `age` must be an integer, not the string `"32"`.
- **Nullability** — `user_id` must never be null; `middle_name` may be.
- **Ranges** — `age` must be between 0 and 120; `probability` between 0.0 and 1.0.
- **Categorical domains** — `country_code` must be one of the ISO-3166 two-letter codes; `status` must be in `{active, churned, trial}`.

### Why it matters in production

Type coercion bugs are among the most common and most damaging failures in ML pipelines. When a numeric column silently arrives as a string, feature engineering may not error — it may just produce garbage (e.g., string concatenation instead of addition, or a hashed categorical where a scaled float was expected). A model trained or scored on that garbage produces confidently wrong predictions. Schema validation converts a silent, hard-to-diagnose accuracy regression into a loud, immediate, easy-to-locate failure. It also documents intent: a schema is executable documentation of what the team believes the data means.

### Real-world example / use case

A fraud-detection team ingests transaction data hourly. An upstream payments service deploys a change that alters `transaction_amount` from integer cents (`1099`) to a float in dollars (`10.99`). No pipeline errors — the column is still numeric. But the model, trained on cents, now sees every transaction as ~100x smaller and stops flagging large fraudulent purchases. A range expectation (`transaction_amount` typically between 100 and 10,000,000 cents) would have failed the very first batch, paging the on-call engineer before a single fraudulent transaction slipped through.

### Code or config example

Great Expectations expresses schemas as **Expectation Suites** — collections of individual expectations. Here is a modern (0.18.x) Fluent Datasource example building a suite programmatically:

```python
import great_expectations as gx

context = gx.get_context()

# Connect to data via a pandas datasource
datasource = context.sources.add_pandas("transactions_src")
data_asset = datasource.add_dataframe_asset(name="transactions")
batch_request = data_asset.build_batch_request(dataframe=transactions_df)

# Create (or load) an expectation suite
suite = context.add_or_update_expectation_suite("transactions.schema")

validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="transactions.schema",
)

# --- Types ---
validator.expect_column_values_to_be_of_type("transaction_amount", "int64")
validator.expect_column_values_to_be_of_type("user_id", "object")  # string

# --- Nullability ---
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_not_be_null("transaction_amount")

# --- Ranges ---
validator.expect_column_values_to_be_between(
    "transaction_amount", min_value=1, max_value=10_000_000
)

# --- Categorical domain ---
validator.expect_column_values_to_be_in_set(
    "status", value_set=["active", "churned", "trial"]
)

# --- Column presence / order ---
validator.expect_table_columns_to_match_set(
    column_set=["user_id", "transaction_amount", "status", "created_at"]
)

# Persist the suite and run
validator.save_expectation_suite(discard_failed_expectations=False)
results = validator.validate()
print("Success:", results.success)
```

The equivalent in TFDV is inferred from statistics, then curated:

```python
import tensorflow_data_validation as tfdv

# Generate statistics from a CSV (works on Beam under the hood)
stats = tfdv.generate_statistics_from_csv(data_location="transactions.csv")

# Infer an initial schema from the statistics
schema = tfdv.infer_schema(statistics=stats)

# Tighten the inferred schema manually
amount = tfdv.get_feature(schema, "transaction_amount")
amount.presence.min_fraction = 1.0        # never null
amount.type = 2                            # 2 = INT in the schema proto

# Constrain a categorical feature's domain
tfdv.get_domain(schema, "status").value[:] = ["active", "churned", "trial"]

tfdv.display_schema(schema)
tfdv.write_schema_text(schema, "schema.pbtxt")
```

### Interview questions

**Q: What is the difference between validating types and validating domains, and why do you need both?**
A: Type validation ensures the storage representation is correct (int vs. string vs. float). Domain validation constrains the *set of allowed values* within a correct type — a `status` column can be a perfectly valid string while still containing an illegal value like `"deleted"`. Types catch structural bugs (bad casts, schema drift); domains catch semantic bugs (new enum values the model was never trained on). A value can pass type checks and still be nonsensical, so both layers are needed.

**Q: A column is defined as non-nullable but 0.5% of rows are null. How do you handle this in a schema?**
A: You rarely want a hard 100% non-null rule in production because real data is messy. Instead of `not_be_null` you use a tolerance: in Great Expectations, most expectations accept a `mostly` parameter (e.g., `mostly=0.99`), which passes if at least 99% of values satisfy the rule. In TFDV you set `presence.min_fraction`. This distinguishes "occasional expected nulls" from "the upstream pipeline broke and everything is null."

### Pitfalls & best practices

- **Don't hand-write dozens of expectations from scratch.** Use profiling (GE's `onboarding data assistant` or TFDV's `infer_schema`) to bootstrap, then curate. Inferred schemas are a starting point, not the final answer.
- **Avoid over-tight ranges.** Ranges derived from a single day of data will produce false positives when normal seasonal variation arrives. Base ranges on historical distributions with headroom.
- **Version your schema alongside your code.** A schema is source code; check it into Git and review changes in PRs.
- **Use `mostly` / `min_fraction` tolerances** rather than absolute rules to avoid alert fatigue.

## Great Expectations and TensorFlow Data Validation (TFDV)

### Plain-language explanation

Great Expectations (GE) and TensorFlow Data Validation (TFDV) are the two dominant open-source data-validation frameworks, and they come from different philosophies.

**Great Expectations** is a general-purpose validation framework centered on human-readable "expectations" (`expect_column_values_to_be_between`, `expect_column_mean_to_be_between`, and ~50 others). It is backend-agnostic — the same suite runs against pandas, Spark, or SQL databases. Its killer feature is **Data Docs**: auto-generated HTML reports that document what your data should look like and whether it currently conforms, making it a communication tool as much as a testing tool.

**TFDV** is part of the TensorFlow Extended (TFX) ecosystem. It is built for scale (runs on Apache Beam), and its core loop is *generate statistics → infer/curate a schema → validate new data against the schema → detect drift and skew*. It excels at automatically inferring schemas from data, computing rich feature statistics, and comparing datasets (train vs. serving, or day-over-day).

### Why it matters in production

Choosing the right tool prevents wasted effort. If your stack is SQL/Spark/pandas heterogeneous and validation is a cross-team communication concern, GE's readable expectations and Data Docs shine. If you are on TFX/TensorFlow and processing huge datasets where you need built-in statistics visualization and training-serving skew detection, TFDV is the natural fit. Using GE where TFDV's Beam scaling is needed (or vice versa) leads to painful re-platforming later.

### Real-world example / use case

A retail analytics team runs a nightly Spark job producing a feature table consumed by three downstream models. They adopt **Great Expectations** because (a) validation runs directly on the Spark DataFrame, (b) the Data Docs give non-engineer stakeholders a browsable report of data health, and (c) checkpoints integrate with their Airflow DAG. A separate computer-vision team training on billions of records in a TFX pipeline uses **TFDV** because its Beam backend scales horizontally and its schema-based skew detection catches training-serving mismatches natively.

### Code or config example

Great Expectations **Checkpoint** — the production-run abstraction that ties a suite to data and can trigger actions on failure:

```python
import great_expectations as gx

context = gx.get_context()

checkpoint = context.add_or_update_checkpoint(
    name="transactions_nightly_checkpoint",
    validations=[
        {
            "batch_request": batch_request,
            "expectation_suite_name": "transactions.schema",
        }
    ],
)

result = checkpoint.run()
context.build_data_docs()          # regenerate the HTML report
assert result["success"], "Data validation failed — see Data Docs"
```

TFDV full validate-against-schema loop, including drift/skew configuration:

```python
import tensorflow_data_validation as tfdv

train_stats = tfdv.generate_statistics_from_tfrecord("train.tfrecord")
schema = tfdv.infer_schema(train_stats)

# Configure drift/skew thresholds directly on the schema
country = tfdv.get_feature(schema, "country_code")
country.skew_comparator.infinity_norm.threshold = 0.01   # train vs serving
country.drift_comparator.infinity_norm.threshold = 0.01  # day over day

# Validate a new batch of serving data against the schema
serving_stats = tfdv.generate_statistics_from_tfrecord("serving.tfrecord")
anomalies = tfdv.validate_statistics(
    statistics=serving_stats,
    schema=schema,
    serving_statistics=serving_stats,   # enables skew_comparator
    previous_statistics=train_stats,    # enables drift_comparator
)
tfdv.display_anomalies(anomalies)
```

### Interview questions

**Q: When would you choose TFDV over Great Expectations?**
A: Choose TFDV when you are in the TFX/TensorFlow ecosystem, need to process very large datasets (it runs on Apache Beam and scales to Dataflow/Spark), want automatic schema inference plus built-in training-serving *skew* detection, and value its statistics visualization (Facets). Choose GE when you need backend-agnostic, human-readable expectations across pandas/Spark/SQL, want Data Docs for cross-team communication, and need flexible custom expectations and rich integrations (Airflow, dbt, CI). They are not mutually exclusive — some shops use both.

**Q: What are Great Expectations "Data Docs" and why are they valuable?**
A: Data Docs are auto-generated, versioned HTML sites that render both the expectation suites (what the data *should* be) and validation results (what it *actually was*). They serve as living documentation and a shared source of truth between data producers, data scientists, and stakeholders. Their value is organizational: they turn validation from an invisible test into a browsable, shareable artifact that builds trust in data quality.

**Q: How does TFDV infer a schema and what are the risks of trusting inference blindly?**
A: TFDV computes per-feature statistics (counts, value distributions, min/max, presence) over a dataset and derives a schema — types, expected domains for categoricals, and presence constraints. The risk is that inference reflects only the sample it saw: rare-but-valid categories may be excluded (causing false anomalies later), and ranges may be too tight or based on already-corrupt data. Best practice is to infer, then review and relax constraints, and re-run `infer_schema` only on known-good data.

### Pitfalls & best practices

- **GE has undergone major API changes** between the V2/V3 "Batch Kwargs" era and the Fluent Datasources of 0.15+ and the 1.0 release. Pin your version and use APIs matching it; mixing tutorials from different eras causes confusion.
- **TFDV schemas are protos** — edit them via the helper functions (`get_feature`, `get_domain`), not by hand-parsing text.
- **Don't regenerate statistics on tiny samples**; statistical checks need enough rows to be meaningful.
- **Store GE suites and TFDV schemas in version control** and treat updates as reviewed changes.

## Statistical Drift Checks

### Plain-language explanation

Schema validation tells you whether each value is *individually* legal. Drift checks tell you whether the *distribution* of a feature has changed relative to a reference (typically the training set or a prior stable window). Even when every value is valid, the shape of the distribution can shift — the mean creeps up, a category that was 5% of traffic becomes 40%, a bimodal distribution collapses to unimodal. This is **data drift**. When the relationship between features and the target changes, that is **concept drift** (harder to detect without labels). Common statistical tools:

- **Population Stability Index (PSI)** — a single number summarizing how much a distribution has moved; >0.1 warns, >0.25 is significant.
- **Kolmogorov–Smirnov (KS) test** — for continuous features, compares two empirical CDFs.
- **Chi-squared test** — for categorical features, compares observed vs. expected counts.
- **Jensen–Shannon divergence / infinity norm (L∞)** — distance metrics between distributions (TFDV uses L∞ for categoricals).

### Why it matters in production

Models are trained on a snapshot of the world and assume the future resembles the past. When input distributions drift, the model extrapolates into regions it never learned, and accuracy silently degrades — often without any schema violation. Drift detection is your early-warning system that a retrain may be needed *before* the business KPI drops. It is also essential for post-incident analysis: "did the model get worse, or did the data change?"

### Real-world example / use case

A loan-approval model trained pre-2020 sees applicant income and employment-tenure distributions shift dramatically during an economic shock. Every value is still schema-valid — incomes are still positive numbers — but the distribution has moved far outside the training range. A PSI check on `income` fires at 0.31, well above the 0.25 threshold, triggering a retraining pipeline and a human review of approval thresholds. Without drift monitoring, the model would keep approving/denying based on a stale view of "normal."

### Code or config example

A pragmatic PSI implementation plus a KS check with SciPy:

```python
import numpy as np
from scipy.stats import ks_2samp

def population_stability_index(reference, current, bins=10):
    """PSI between a reference and current 1-D distribution."""
    quantiles = np.quantile(reference, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=quantiles)[0] / len(reference)
    cur_pct = np.histogram(current, bins=quantiles)[0] / len(current)
    # avoid div-by-zero
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    return np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

psi = population_stability_index(train_df["income"], today_df["income"])
print(f"PSI = {psi:.3f}")            # >0.25 => significant drift

# KS test for a continuous feature
ks_stat, p_value = ks_2samp(train_df["income"], today_df["income"])
if p_value < 0.01:
    print("Distributions differ significantly (reject H0)")
```

Great Expectations has a purpose-built distributional expectation:

```python
# Compares the observed distribution against a stored partition object
validator.expect_column_kl_divergence_to_be_less_than(
    column="income",
    partition_object=reference_income_partition,  # bins + weights from training
    threshold=0.1,
)
```

TFDV drift comparator (declarative, in the schema) as shown earlier:

```python
tfdv.get_feature(schema, "country_code").drift_comparator.infinity_norm.threshold = 0.01
anomalies = tfdv.validate_statistics(
    statistics=today_stats, schema=schema, previous_statistics=reference_stats
)
```

### Interview questions

**Q: What is the difference between data drift and concept drift?**
A: Data drift (covariate shift) is a change in the distribution of the *inputs* P(X) — e.g., your users skew younger this quarter. Concept drift is a change in the relationship P(Y|X) — the same inputs now map to different outcomes (e.g., spending patterns that used to signal fraud no longer do). Data drift is detectable from features alone; concept drift generally requires labels (ground truth) to detect, which often arrive with delay, making it harder to catch in real time.

**Q: How do you interpret a PSI value, and what are its limitations?**
A: PSI < 0.1 means no significant shift, 0.1–0.25 a moderate shift worth watching, and > 0.25 a major shift usually warranting action. Limitations: it depends on binning strategy (too few/many bins distort it), it is univariate (misses correlated multivariate drift), and it doesn't tell you whether the drift actually harms model performance — a feature can drift a lot yet be unimportant to the model. Pair PSI with feature-importance weighting and, when possible, direct performance monitoring.

**Q: A feature shows significant statistical drift but model accuracy is stable. What do you do?**
A: Don't retrain reflexively. First check the feature's importance — low-importance drift may be harmless. Confirm the drift isn't a data-quality artifact (an upstream change) versus a genuine population shift. Continue monitoring for lagging effects and set up alerting tied to performance, not just distribution. Drift alarms are a signal to investigate, not an automatic trigger to retrain.

### Pitfalls & best practices

- **Fix a stable reference window**, not a rolling one, or slow gradual drift will hide because the baseline moves with it.
- **Weight drift by feature importance** so alerts focus on features that actually affect predictions.
- **Beware small samples** — KS and chi-squared p-values are sensitive to sample size; with millions of rows, trivial differences become "statistically significant."
- **Segment your drift checks** (by region, device, cohort); aggregate distributions can look stable while a critical segment drifts sharply.

## Anomaly Detection

### Plain-language explanation

Anomaly detection finds data points or dataset properties that deviate from expected patterns and aren't caught by static schema rules. It operates at two levels:

- **Row-level anomalies** — individual records that are outliers: a transaction 500x the median, a sensor reading physically impossible, a user with a negative age. These are found with rules (bounds, z-scores) or unsupervised models (Isolation Forest, Local Outlier Factor, autoencoders).
- **Dataset-level anomalies** — properties of the whole batch that are off: row count dropped 90% versus yesterday, null rate tripled, a new categorical value appeared, cardinality of an ID column collapsed. These are "did the whole feed break?" checks.

Drift (previous section) is a specific kind of dataset-level anomaly focused on distribution shape; here we broaden to volume, freshness, and structural anomalies too.

### Why it matters in production

Row-level anomalies poison training data and produce absurd individual predictions; dataset-level anomalies indicate broken pipelines — a partial load, a failed upstream job, a duplicated batch. Dataset-level checks are often the *first* alarm when an upstream system breaks, precisely because they are cheap and holistic. Without volume and freshness checks, a pipeline can process an empty or half-loaded file "successfully" and you'll train tomorrow's model on a fraction of the data.

### Real-world example / use case

An e-commerce recommender ingests a daily clickstream export. One morning an upstream job fails halfway, producing a file with 8% of the usual rows. Every row is schema-valid, so schema checks pass. But a dataset-level row-count anomaly check (`row_count within ±30% of trailing 7-day median`) fails immediately, halting the training pipeline and paging the data team. Separately, a row-level check flags a batch of records with `session_duration = 2^31 - 1` (an integer-overflow sentinel) that would have skewed the "average session" feature.

### Code or config example

Great Expectations dataset-level (volume) and row-level (outlier) expectations:

```python
# Dataset-level: row count within an expected band
validator.expect_table_row_count_to_be_between(min_value=90_000, max_value=150_000)

# Dataset-level: null rate stays low (tolerance via mostly)
validator.expect_column_values_to_not_be_null("user_id", mostly=0.999)

# Dataset-level: no unexpected new categories
validator.expect_column_values_to_be_in_set(
    "device_type", value_set=["ios", "android", "web"], mostly=0.995
)

# Row-level: values within physically plausible bounds
validator.expect_column_values_to_be_between("session_duration", 0, 86_400)
```

Unsupervised row-level anomaly detection with Isolation Forest:

```python
from sklearn.ensemble import IsolationForest
import numpy as np

features = df[["amount", "session_duration", "clicks"]].to_numpy()

iso = IsolationForest(contamination=0.01, random_state=42)
labels = iso.fit_predict(features)      # -1 = anomaly, 1 = normal

anomaly_rows = df[labels == -1]
anomaly_rate = (labels == -1).mean()
if anomaly_rate > 0.05:                 # far more anomalies than expected
    raise ValueError(f"Anomaly rate {anomaly_rate:.1%} exceeds threshold")
```

TFDV surfaces both new-value and volume anomalies automatically via `validate_statistics`, reporting anomaly types such as `SCHEMA_NEW_VALUE`, `COMPARATOR_LOW_NUM_EXAMPLES`, and `FEATURE_TYPE_MISMATCH`:

```python
anomalies = tfdv.validate_statistics(statistics=today_stats, schema=schema)
for name, anomaly in anomalies.anomaly_info.items():
    print(name, anomaly.description)
```

### Interview questions

**Q: Distinguish row-level from dataset-level anomaly detection with an example of each.**
A: Row-level detection flags individual bad records — e.g., a purchase amount of −$5 or an age of 900. Dataset-level detection flags aggregate properties of the batch — e.g., row count fell 90%, the null rate for `email` jumped from 1% to 60%, or a brand-new category appeared. Row-level protects against poisoned individual samples and absurd predictions; dataset-level protects against broken pipelines and partial loads. Production systems need both.

**Q: How would you choose between rule-based and model-based anomaly detection?**
A: Use rule-based checks (bounds, allowed sets, row-count bands) when you have clear domain knowledge and want interpretable, deterministic, fast checks — this covers the majority of production needs. Use model-based methods (Isolation Forest, autoencoders) when anomalies are multivariate and hard to express as simple rules, e.g., a combination of features that is individually normal but jointly suspicious. Model-based methods add complexity, require tuning (e.g., `contamination`), and are harder to explain, so reserve them for cases rules can't capture.

**Q: Why is a row-count / volume check often the most valuable single data check?**
A: Because a huge class of failures — failed upstream jobs, partial file loads, duplicated batches, empty exports — all manifest as an abnormal row count while leaving individual rows schema-valid. It's cheap to compute, catches "the feed broke" holistically, and often fires before any other check. It's a high-signal, low-cost tripwire.

### Pitfalls & best practices

- **Account for seasonality** in volume checks — Monday traffic differs from Sunday; use day-of-week-aware baselines or trailing medians.
- **Tune `contamination` carefully** in Isolation Forest; setting it wrong either floods you with false positives or misses real anomalies.
- **Don't silently drop anomalies** — quarantine and log them for investigation; the anomaly rate itself is a monitoring signal.
- **Combine cheap deterministic checks first**, then model-based checks only where needed, to keep pipelines fast and debuggable.

## Integrating Validation into CI/CD Pipelines

### Plain-language explanation

Data validation is only useful if it actually *stops* bad data. Integrating validation into CI/CD means running your expectation suites or TFDV checks as automated gates in your pipelines — and *failing the build or halting the DAG* when validation fails, exactly the way a failing unit test blocks a code merge. There are two contexts: (1) CI on your code/config repo (validating on sample data when someone changes a transformation), and (2) orchestration pipelines (Airflow, Argo, GitHub Actions, GitLab CI) that validate real data before training or serving steps proceed.

The core principle: **validation must be a blocking gate, not an advisory log line.** A non-zero exit code fails the job.

### Why it matters in production

If validation results only land in a dashboard nobody watches, bad data still flows through. Making validation a hard gate enforces a fail-fast, fail-loud discipline: a broken batch stops before it corrupts a model or reaches production. This shifts data-quality problems left — caught in a pipeline run instead of discovered weeks later through a business metric. It also creates an auditable record: every training run is provably preceded by a passing validation.

### Real-world example / use case

A team runs training weekly via GitHub Actions. Before the training job, a step runs a Great Expectations checkpoint on the freshly assembled feature set. When a data provider ships a malformed batch (a categorical column with an unexpected value), the checkpoint returns `success: False`, the step exits non-zero, the workflow goes red, training is skipped, and Slack is notified. The bad model is never produced. The previous good model keeps serving. Once fixed, a re-run passes and training proceeds.

### Code or config example

A GitHub Actions workflow that fails on bad data:

```yaml
name: data-validation
on:
  workflow_dispatch:
  schedule:
    - cron: "0 6 * * 1"   # Monday 06:00 UTC, before weekly training

jobs:
  validate-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install deps
        run: pip install "great_expectations>=0.18,<1.0" pandas
      - name: Run Great Expectations checkpoint
        run: python scripts/run_checkpoint.py   # exits 1 if validation fails
      - name: Upload Data Docs
        if: always()                            # publish even on failure
        uses: actions/upload-artifact@v4
        with:
          name: data-docs
          path: gx/uncommitted/data_docs/
```

The gate script that turns validation into an exit code:

```python
# scripts/run_checkpoint.py
import sys
import great_expectations as gx

context = gx.get_context()
result = context.run_checkpoint(checkpoint_name="transactions_nightly_checkpoint")
context.build_data_docs()

if not result["success"]:
    print("::error::Data validation FAILED — see Data Docs artifact")
    sys.exit(1)          # <-- fails the CI job
print("Data validation passed")
```

In Airflow, the same gate uses the Great Expectations operator (or a PythonOperator that raises):

```python
from airflow.decorators import task

@task
def validate():
    result = context.run_checkpoint(checkpoint_name="transactions_nightly_checkpoint")
    if not result["success"]:
        raise ValueError("Validation failed — halting DAG before training")

validate() >> train_model()   # train only runs if validate() succeeds
```

### Interview questions

**Q: Why should a data-validation failure fail the build rather than just log a warning?**
A: Because warnings get ignored and bad data keeps flowing. Treating validation like a unit test — a hard, blocking gate that returns a non-zero exit code — enforces fail-fast discipline: the pipeline stops before corrupt data reaches training or serving. This shifts detection left (caught in the pipeline, not in production metrics weeks later) and creates an audit trail proving every model was trained on validated data. Advisory-only validation provides false comfort.

**Q: How do you prevent data-validation gates from becoming flaky and eroding trust?**
A: Use tolerances (`mostly`, `min_fraction`) instead of brittle absolutes; base ranges and volume bands on historical distributions with headroom; account for seasonality; separate hard-fail checks (schema, volume, nulls) from soft-warn checks (mild drift); and version suites so changes are reviewed. Flaky gates that cry wolf get disabled, so calibrate thresholds against real historical data and monitor the false-positive rate of the checks themselves.

**Q: Where in a training pipeline should validation gates live?**
A: At every data boundary: after ingestion (raw-data validation), after feature engineering (transformed-data validation), and, for online systems, at inference input. The most critical single gate is immediately before training/model-production so a bad batch can't produce a bad model. For serving, validate inputs at request or micro-batch time to catch training-serving skew.

### Pitfalls & best practices

- **Publish reports even on failure** (`if: always()` in CI) so engineers can diagnose without re-running.
- **Fail closed for critical checks**, but consider fail-open with alerting for non-critical enrichment data so one flaky source doesn't block everything.
- **Keep validation fast** — run heavy statistical checks on samples or in parallel so the gate doesn't dominate pipeline runtime.
- **Alert with context** — link straight to the failing expectation and Data Docs, not just "validation failed."

## Contract Testing Between Data Producers and Consumers (Data Contracts)

### Plain-language explanation

A **data contract** is an explicit, versioned, machine-readable agreement between the team that *produces* a dataset (or event stream) and the teams that *consume* it. It specifies the schema, semantics, quality guarantees (SLAs on freshness, completeness, allowed null rates), ownership, and how changes will be communicated and versioned. **Contract testing** is the practice of automatically verifying that the producer still honors the contract — ideally in the producer's own CI, so a breaking change is caught *before* it ships and breaks downstream consumers.

The key shift is ownership: instead of consumers defensively validating data they can't control (and being surprised by upstream changes), the producer commits to a contract and is tested against it. This is the data analog of API contract testing (e.g., Pact).

### Why it matters in production

Most damaging data incidents originate upstream: a producer renames a field, changes units, drops a column, or loosens a constraint — with no idea who depends on it. Data contracts invert responsibility so producers can't silently break consumers. They make dependencies explicit, enable safe schema evolution (via compatibility rules), and provide SLAs consumers can rely on. In a large org, contracts are what let dozens of teams share data without constant firefighting. They turn "the pipeline broke because someone upstream changed something" into "CI blocked the breaking change and forced a versioned migration."

### Real-world example / use case

A platform team publishes a `user_signup` event to Kafka consumed by the growth model, the billing service, and the analytics warehouse. They define a contract (schema + compatibility policy: backward-compatible changes only). When an engineer tries to rename `signup_ts` to `created_at`, the schema-registry compatibility check in their CI rejects the change because it breaks backward compatibility. The engineer instead *adds* `created_at` alongside `signup_ts`, deprecates the old field over a migration window, and bumps the contract version. No consumer breaks. Contrast with the no-contract world, where the rename ships Friday and three teams debug null columns over the weekend.

### Code or config example

A declarative data contract (YAML, in the style of the open Data Contract Specification / ODCS):

```yaml
# contracts/user_signup.yaml
dataContractSpecification: "0.9.3"
id: user_signup
info:
  title: User Signup Events
  version: "2.1.0"
  owner: platform-team
  contact: { name: Platform On-Call, email: platform@company.com }

servers:
  production:
    type: kafka
    topic: user.signup.v2

models:
  user_signup:
    type: table
    fields:
      user_id:   { type: string,    required: true, unique: true }
      signup_ts: { type: timestamp, required: true }
      plan:      { type: string,    required: true, enum: [free, pro, enterprise] }
      country:   { type: string,    required: true, pattern: "^[A-Z]{2}$" }

quality:
  - type: freshness
    description: "Events delivered within 5 minutes"
    threshold: 5m
  - type: completeness
    field: user_id
    mustBe: 100
  - type: schema_compatibility
    policy: BACKWARD          # new versions must not break existing consumers
```

Confluent Schema Registry compatibility check enforced in producer CI:

```yaml
# .gitlab-ci.yml (producer repo)
validate-schema-compatibility:
  stage: test
  script:
    # Fails the pipeline if the new schema breaks BACKWARD compatibility
    - |
      curl -s -X POST \
        -H "Content-Type: application/vnd.schemaregistry.v1+json" \
        --data "@schemas/user_signup.avsc" \
        "$SCHEMA_REGISTRY_URL/compatibility/subjects/user.signup.v2-value/versions/latest" \
        | tee compat.json
    - grep -q '"is_compatible":true' compat.json || (echo "Breaking change!" && exit 1)
```

A consumer-side contract test that asserts the promised guarantees on real data:

```python
# consumer CI: verify the producer honors the contract
def test_user_signup_contract(sample_batch):
    assert set(["user_id", "signup_ts", "plan", "country"]).issubset(sample_batch.columns)
    assert sample_batch["user_id"].notna().all()          # completeness: 100%
    assert sample_batch["plan"].isin(["free", "pro", "enterprise"]).all()
    assert sample_batch["country"].str.match(r"^[A-Z]{2}$").all()
```

### Interview questions

**Q: What is a data contract and how does it differ from ordinary schema validation?**
A: Schema validation is typically a *consumer-side, defensive* check on data the consumer receives but doesn't control. A data contract is an *explicit, versioned agreement* owned by the producer, covering not just schema but semantics, quality SLAs (freshness, completeness), ownership, and change/compatibility policy — and it's enforced in the *producer's* CI so breaking changes are caught before they ship. The difference is ownership and prevention: contracts stop breakage at the source rather than detecting it after the fact downstream.

**Q: What are backward, forward, and full schema compatibility, and why do they matter for contracts?**
A: *Backward* compatibility means new schema versions can read data written with old schemas (safe to upgrade consumers first) — you can add optional fields or remove fields with defaults. *Forward* compatibility means old schemas can read data written with new ones (safe to upgrade producers first). *Full* compatibility means both. They matter because they define what evolution is safe without coordinated lockstep deploys; a schema registry enforces the chosen policy so producers can't ship a change that breaks the agreed contract.

**Q: How do you evolve a data contract without breaking consumers?**
A: Follow the compatibility policy (usually backward-compatible): add new fields as optional/defaulted rather than renaming or removing; when a breaking change is truly needed, publish a new *version* of the dataset/topic, run old and new in parallel during a deprecation window, communicate to consumers, and retire the old version only after all consumers migrate. Never mutate a field's meaning or units in place — that's a silent breaking change even if the schema type is unchanged.

### Pitfalls & best practices

- **Make contracts machine-readable and enforced in CI** — a contract that lives only in a wiki is documentation, not a contract.
- **Enforce compatibility at the source** (producer CI + schema registry), not just via consumer defensive checks.
- **Version explicitly and deprecate gracefully**; never rename/repurpose fields in place.
- **Cover semantics and SLAs, not just types** — units, timezones, freshness, and completeness cause as many incidents as type mismatches.
- **Assign clear ownership** — every contract needs a named owning team and on-call contact.

## Summary

Data validation is the immune system of an ML platform. We moved from the foundational layer — **schema validation** of types, nullability, ranges, and categorical domains — to the tools that operationalize it: **Great Expectations**, a backend-agnostic, human-readable framework with Data Docs for cross-team communication, and **TFDV**, a Beam-scale component of TFX with automatic schema inference and native training-serving skew detection. Choose GE for heterogeneous pandas/Spark/SQL stacks and communication; choose TFDV for large-scale TensorFlow pipelines.

Beyond static schemas, **statistical drift checks** (PSI, KS, chi-squared, L∞/JS divergence) catch distribution shifts even when every value is individually valid, giving early warning that a retrain may be needed — while distinguishing data drift from concept drift. **Anomaly detection** operates at the row level (outliers via bounds or Isolation Forest) and the dataset level (volume, freshness, null-rate, new-value checks), where a simple row-count tripwire is often the highest-value single check. None of this matters unless it *acts*: **integrating validation into CI/CD** as a hard, blocking gate — failing the build on bad data — shifts detection left and guarantees models train only on validated data. Finally, **data contracts** invert responsibility to producers: explicit, versioned, machine-readable agreements enforced in producer CI (often via a schema registry's compatibility rules) that prevent breaking changes from ever shipping. Together these layers turn silent data failures into loud, early, actionable signals.

## Chapter Q&A

**Q1. Walk me through the layers of data validation you'd build for a production ML pipeline, from cheapest to most sophisticated.**
A: Start with **schema validation** (types, nullability, ranges, domains) — cheap, deterministic, catches most catastrophic bugs. Add **dataset-level checks** (row count/volume, freshness, null rates, new categorical values) to catch broken pipelines. Add **row-level anomaly detection** (bounds, then Isolation Forest for multivariate cases) to catch poisoned records. Layer on **statistical drift detection** (PSI/KS/chi-squared) against a fixed reference to catch distribution shift. Wire all of it into **CI/CD as blocking gates**. At the organizational level, formalize **data contracts** enforced in producer CI. Cheap deterministic checks first, model-based and statistical checks where rules can't reach.

**Q2. A model's accuracy dropped last week. How does your validation infrastructure help you diagnose whether it's a data problem or a model problem?**
A: I'd check validation history first. If schema/volume/anomaly gates passed but **drift metrics spiked** (PSI > 0.25 on important features), it points to a genuine population shift — likely needing retraining. If an anomaly or schema check *would have* fired (e.g., a producer changed units), it's a data-quality/contract breach upstream. If all data checks are clean but accuracy dropped, it suggests **concept drift** (P(Y|X) changed) or a model/serving bug. The validation history plus drift dashboards let me separate "the world changed," "the data broke," and "the model/code broke."

**Q3. When would you use Great Expectations versus TFDV, and can they coexist?**
A: GE for backend-agnostic, human-readable expectations across pandas/Spark/SQL, rich integrations, custom expectations, and Data Docs as a communication tool. TFDV for TFX/TensorFlow ecosystems, very large datasets (Beam scaling), automatic schema inference, statistics visualization (Facets), and built-in training-serving skew detection. They coexist fine — e.g., TFDV inside a TFX training pipeline and GE for warehouse/Spark data-quality gates in Airflow.

**Q4. Explain PSI and how you'd act on it.**
A: Population Stability Index quantifies how far a current distribution has moved from a reference by comparing binned proportions with a log-ratio sum. <0.1 = stable, 0.1–0.25 = watch, >0.25 = significant. I act by weighting PSI by feature importance (ignore drift on unimportant features), confirming it's a real population shift versus a data-quality artifact, checking for lagging performance impact, and triggering retraining or threshold review when important features drift materially — not reflexively on any drift.

**Q5. What's the difference between data drift and training-serving skew?**
A: Data drift is a temporal change — the distribution today differs from the reference/training window. Training-serving skew is a *point-in-time* discrepancy between the data used at training and the data seen at serving *for the same period*, usually caused by different code paths, feature computation, or preprocessing between offline and online. TFDV models both: a `drift_comparator` (vs. previous stats) and a `skew_comparator` (vs. serving stats). Skew is often a bug; drift is often the world changing.

**Q6. How do you make data validation a true quality gate rather than noise?**
A: Turn results into exit codes that fail the build/DAG (fail-fast), but calibrate to avoid flakiness: use tolerances (`mostly`, `min_fraction`), historically derived ranges with headroom, seasonality-aware volume bands, and a split between hard-fail checks (schema, volume, nulls) and soft-warn checks (mild drift). Publish reports even on failure, alert with direct links to the failing expectation, version suites in Git, and monitor the checks' own false-positive rate so they keep the team's trust.

**Q7. Design a data contract for an event stream shared by three teams. What goes in it and how is it enforced?**
A: The contract (machine-readable YAML/Avro) specifies: schema (fields, types, required, enums, patterns), semantics (units, timezones, meaning), quality SLAs (freshness, completeness, null-rate limits), ownership and on-call contact, and a **compatibility policy** (e.g., BACKWARD). Enforcement lives in the **producer's CI** via a schema registry compatibility check that rejects breaking changes before merge, plus consumer-side contract tests asserting the guarantees on real samples. Evolution is versioned: add optional fields, never rename/repurpose in place, and run old+new in parallel during deprecation.

**Q8. What are the most common false-positive sources in drift/anomaly detection and how do you mitigate them?**
A: (1) **Seasonality** — Monday vs. Sunday volume — mitigate with day-of-week/trailing-median baselines. (2) **Large-sample statistical significance** — KS/chi-squared flag trivial differences at scale — use effect-size metrics (PSI, L∞) and practical thresholds, not just p-values. (3) **Rolling baselines** hiding gradual drift — fix a stable reference. (4) **Aggregate blindness** — segment checks by cohort. (5) **Rare-but-valid categories** excluded by inferred schemas — curate schemas and use `mostly`/tolerances.

**Q9. Where should validation live in an online (real-time) serving system, and what are the constraints?**
A: Validate inference inputs at request or micro-batch time to catch training-serving skew and malformed payloads. Constraints are latency and blast radius: checks must be cheap (schema, bounds, presence — not heavy distributional tests inline). Do lightweight per-request validation synchronously, and compute drift/statistics asynchronously on sampled logged inputs. Decide fail-open vs. fail-closed per feature: reject clearly invalid requests, but for non-critical enrichment features prefer graceful degradation with alerting rather than dropping the request.

**Q10. If you could implement only one data check on day one, what would it be and why?**
A: A **row-count / volume check against a trailing baseline**. A large fraction of real incidents — failed upstream jobs, partial loads, empty exports, duplicated batches — all show up as an abnormal row count while individual rows stay schema-valid. It's trivially cheap, holistic, and usually the first alarm when a feed breaks. I'd pair it immediately with a non-null check on primary keys, but if it's truly one check, it's volume.
