
docs/PRD.md
v2.0 · draft
Copy markdown
Download .md
SMARTGRID AI — Product Requirements Document
Field	Value
Version	2.0 (supersedes 1.0)
Status	Draft for build

Assumed capacity	1 developer, 12 weeks, ~15 h/week
Hard constraint	Must demo all five syllabus modules
Last revised	2026-11-11
0. Revision rationale
v1 was a feature list. It could not be built as written and could not be defended, for four reasons.

No baselines. A forecasting model with R² = 0.94 is worthless if seasonal-naive gets 0.95. v1 had no comparison floor.
No data contract. The smart-meter files have mixed delimiters, carry per-3-minute energy rather than power, and use ambiguous outage zeros. v1 described none of it.
No failure paths. Every module was specified as working. Nothing said what happens when a meter has 60% attrition or a segmentation subset has no test split.
Scope with no cut lines. Five modules plus a dashboard plus deployment, with no ordering and no MVP boundary that survives scrutiny.
v2 keeps the concept, fixes the engineering, and adds per-module exit criteria so "done" is measurable.

1. Problem
Three separable problems, kept separate because they have different data and different failure modes.

ID	Problem	Consequence today
P1	Household and grid demand is volatile and seasonal; planning runs on historical averages	Poor day-ahead allocation, no headroom planning
P2	Distribution infrastructure degrades visually — lean, tilt, vegetation encroachment — and inspection is manual and infrequent	Defects found late, after failure
P3	Unusual consumption is invisible below billing granularity	Anomalies surface only as complaints or losses
Product thesis: these three are one workflow for an analyst — forecast, inspect, flag — and a shared feature store plus one dashboard makes them cheaper together than apart.

Non-goals
Not a certified electrical safety assessment.
Not theft detection. Anomalies are flags for investigation.
Not real-time SCADA. Batch and on-demand inference only.
Not a billing system, not multi-tenant, no mobile app.
No live meter integration. Static dataset replay.
No transmission-network models.
2. Users and the decision each makes
User	Decision	Needs	Success looks like
Energy analyst	How much headroom for tomorrow's peak?	Forecast, interval, error history	Forecast beats baseline; error bars shown
Discom engineer	Which assets to send a crew to?	Ranked inspection queue	Defect crops surfaced with confidence
Researcher	Which architecture generalises?	Reproducible runs, seed variance	Config hash plus exportable metric table
Examiner	Does the student understand the models?	Live experiments, gradient plots, ablations	Every claim has a baseline
3. Data foundation
This section is normative. Every module depends on it.

3.1 Sources
ID	Dataset	License	Role	Volume
D1	CEEW Mathura/Bareilly smart meter, Harvard Dataverse doi:10.7910/DVN/GOCHJH	CC0 1.0	Household demand	~5.4M rows, ~100 meters
D2	India demand/solar/wind, Mendeley y58jknpgs8	CC BY 4.0	Grid demand	Hourly, 2021-2025
D3	PD-Defect, huggingface.co/datasets/EPDCL/pd-defect	CC BY 4.0	Infrastructure vision	20,476 examples, 14.5 GB
D4	IRES 2020, doi:10.7910/DVN/U8NYUP	Verify on deposit	Household profiling (stretch)	Survey
D5	REDI, doi:10.7910/DVN/1ZNLUY	Verify on deposit	Rural profiling (stretch)	Survey
Attribution for D2 and D3 must appear in the UI footer and docs/CREDITS.md. CC BY is not decorative.

3.2 D1 physical contract
columns : x_Timestamp, meter, t_kWh, z_Avg Voltage (Volt), z_Avg Current (Amp), y_Freq (Hz)
delim   : TAB for 2019 files, both Aggregated files, Mathura 2021; COMMA for 2020/2021 CSVs
tz      : IST, no DST, store UTC
grid    : 3 minutes
Non-obvious rules, all mandatory.

t_kWh is energy per 3-minute bucket. Convert to average power: P_kW = t_kWh × 20. Never model raw 3-minute kWh.
Outage is not zero demand. If voltage_v ≤ 5 or current_a ≤ 0.01, supply is off. Set kwh_3min = NaN, keep is_outage = True. Unflagged outage zeros teach the model the discom's load-shedding schedule instead of household behaviour.
Attrition is real. Mathura 2021 holds 550,560 rows against Bareilly 2019's 2,919,315. A meter needs ≥ 30 continuous days at ≥ 70% coverage to enter the training pool.
Aggregated files are daily totals — the fallback series, not the training source.
3.3 Quality gates
A meter-series is admitted only if all pass.

Gate	Threshold
Continuous span	≥ 30 days
15-minute bin coverage	≥ 80% of bins
Outage fraction	≤ 40%
Negative or impossible values	0 after cleaning
Duplicate timestamps	0 after dedupe
Rejected meters go to quality_report.json with the failing gate named. Report the admission rate; expect 60-70%.

3.4 Splitting
Strictly chronological, per meter, 70/15/15, boundaries written to a versioned split_manifest.json alongside a SHA-256 of each source file. No random shuffling of time-series rows anywhere. Any run whose manifest hash differs from its registry entry is invalid.

For D3, PD-Defect's own splits are authoritative, with one addition: run perceptual-hash dedupe across train/valid, because the same field photograph recurs across subsets and will inflate accuracy if it lands on both sides.

3.5 Storage
Raw files stay in object storage. PostgreSQL holds metadata, predictions, experiments, and quality reports — never raw series. Processed series are partitioned Parquet at city=/year=/meter=/.

4. Architecture
Three planes, hard-separated. The API process never imports TensorFlow at boot.

job lifecycle:  queued → running → succeeded
                     ↳ failed (exception, attempts < 2)
                     ↳ failed (timeout > 20 min)
                failed → queued (manual retry)
Frontend — React 18 + Vite + Tailwind. Vercel.
API — FastAPI, thin. Render. Auth, validation, orchestration, persistence.
Inference and training service — FastAPI + TensorFlow + Ultralytics, separate container. HF Spaces Docker (CPU, 16 GB). Render's 512 MB cannot hold TF plus YOLO.
Queue — Redis + RQ. /api/training/compare returns 202 with a job ID; the dashboard polls /api/jobs/{id}.
Registry — weights in object storage, metadata in model_registry with config hash, dataset hash, seed, and metric row. Promotion is a manual flag, never automatic.
5. Modules
Each module carries a primary metric, a mandatory baseline, a numeric Definition of Done, and a cut-line fallback. A model that misses DoD is reported as "does not beat baseline" — a valid, defensible result.

M1 — Shallow networks (perceptron, XOR, MLP)
Purpose: demonstrate single-layer limits and a supervised baseline for demand.
Method: perceptron on two-feature toy data; XOR via 2-2-1 MLP with sigmoid/tanh/ReLU comparison; MLP regressor on 15-minute features.
Baseline: ridge regression on identical features.
Metrics: MAE, RMSE, R²; XOR convergence rate over 20 seeds.
DoD: XOR solves in ≥ 19/20 seeds with a hidden layer and ≤ 2/20 without; MLP beats ridge MAE on the test split.
Cut line: XOR plus MLP only; drop the activation sweep.
M2 — Training dynamics (DNN, optimisers, init, regularisation, gradients)
Purpose: show that optimisation choices move the loss, with the receipts.
Method: fixed-architecture DNN; SGD vs Adam vs RMSprop; random vs Xavier vs He init; dropout vs BatchNorm vs neither; per-layer gradient-norm logging.
Baseline: every configuration against the default (Adam, He, no regularisation).
Metrics: MAE, RMSE, epochs-to-convergence, wall-clock, final gradient-norm profile.
Protocol: 3 seeds per cell, report mean ± std. Single-seed comparisons are rejected.
DoD: vanishing gradients demonstrated reproducibly (5-layer sigmoid stack, gradient norm decaying more than 2 orders of magnitude by layer 5); exploding reproduced with high learning rate; clipping shown to fix it.
Cut line: optimisers only, 3 seeds, fixed init.
M3 — Vision (CNN, YOLO, U-Net)
Purpose: assist inspection triage.
Corrected task map: Pole Lean Classification is the CNN task (3-class). Voltage Level is object detection, not classification — v1 had this wrong. Vegetation–Conductor is COCO segmentation with train/valid only and no test split.
Baseline: majority-class predictor for classification; published YOLO12m and Mask2Former numbers as reference ceilings, not as your results.
Metrics: macro-F1 and per-class recall for classification (accuracy lies under imbalance); mAP@0.5 and mAP@0.5:0.95 per class for detection; mIoU and per-class IoU for segmentation.
DoD: CNN macro-F1 ≥ 0.70; YOLO mAP@0.5 ≥ 0.85 on Pole Detection; U-Net mIoU ≥ 0.55 on a 15% carve-out from train; p95 CPU latency under 4 s.
Cut line: CNN plus YOLO only. Drop U-Net to a documented stub.
M4 — Sequence forecasting (RNN, LSTM, GRU, BiLSTM)
Purpose: day-ahead household and grid demand.
Method: 7 days in, 24 hours out, direct multi-step. Per-meter normalisation; one global model with a meter embedding. Outage bins masked from the loss.
Baselines, in order: persistence (ŷ = y[t-24h]), seasonal-naive (ŷ = y[t-168h]), then LightGBM on calendar and lag features. A sequence model that cannot beat seasonal-naive is not a forecasting result.
Metrics: MAE, RMSE, MASE (primary — scale-free and defined at zeros). MAPE reported but caveated for sparse meters where it explodes.
Protocol: report per-horizon error, not one aggregate. A model can win at h=1 and lose at h=24.
DoD: LSTM and GRU both beat seasonal-naive MASE on the test split across ≥ 70% of admitted meters.
Cut line: LSTM plus GRU only; drop RNN and BiLSTM. Note in the write-up that BiLSTM needs future context and is therefore not a pure online forecaster — that distinction earns marks.
M5 — Generative (GAN, VAE)
Purpose: synthetic series for augmentation; reconstruction-based anomaly scoring.
GAN evaluation: three tests — discriminator score on held-out real data, MMD against real, and a train-on-synthetic utility test (train the M4 LSTM on synthetic, evaluate on real; utility is the headline number).
VAE anomaly evaluation: the dataset has no anomaly labels, so inject them. Protocol: take clean windows, apply five labelled perturbations (magnitude scaling, load shift, flatline, dropout burst, phase shift), score all, report precision/recall/F1 at the validation-derived threshold. Without injection there is no measurable anomaly result.
Baseline: z-score on daily totals for anomaly detection; random-window GAN comparison.
DoD: GAN synthetic data yields LSTM utility within 15% of real-data training; VAE F1 ≥ 0.65 at the p99 threshold.
Cut line: VAE only. Anomaly detection is the higher-value half.
Dashboard
Overview KPIs, consumption analytics, per-module labs, inspection queue, forecast console, anomaly monitor, model performance table, export. Recharts for series, Plotly for distributions. Dark charcoal, electric blue and green accents, sidebar navigation. Every chart shows its baseline series — no naked predictions.

6. Evaluation protocol
Test split touched once per model family. Re-tuning on test is recorded.
Minimum 3 seeds for anything stochastic; mean ± std in every table.
Every metric row stores model, config hash, dataset hash, seed, hardware, wall-clock, timestamp.
All baselines live in the same table as the models, never in prose.
Reported errors use a fixed, pre-registered horizon and mask.
Negative results are published in the dashboard. Hiding them is the fastest way to lose a viva.
7. API contract
All endpoints return {"data": ..., "meta": {"model_version", "dataset_hash", "latency_ms"}} or {"error": {"code", "message", "detail"}}. Training endpoints return 202 with a job_id.

Endpoint	Method	Auth	Body / params
/api/analytics/overview	GET	viewer	—
/api/analytics/series	GET	viewer	meter_id, from, to, resolution
/api/ann/predict	POST	analyst	features[]
/api/ann/xor	GET	viewer	seed, hidden
/api/training/compare	POST	admin	optimizer[], init, regularisation, seeds
/api/jobs/{id}	GET	analyst	—
/api/infrastructure/classify	POST	analyst	multipart image
/api/infrastructure/detect	POST	analyst	multipart image
/api/infrastructure/segment	POST	analyst	multipart image
/api/forecast/{arch}	POST	analyst	meter_id, history_h, horizon_h
/api/generative/gan	POST	analyst	meter_id, n_samples, seed
/api/generative/vae	POST	analyst	meter_id, window
/api/anomalies	GET	viewer	severity, page
/api/models/metrics	GET	viewer	module
/api/export/{artifact}	GET	analyst	format
Validation: magic-byte type check rather than extension, 10 MB image cap, 4096 px cap, EXIF stripped, meter_id existence check before inference.

8. Data model
users(id, email, password_hash, role, created_at) — argon2id; roles viewer | analyst | admin.
datasets(id, name, source, license, sha256, admitted_meters, quality_report_path, ingested_at)
model_registry(id, name, module, architecture, config_hash, dataset_hash, seed, weights_path, promoted, created_at)
experiments(id, model_id, optimizer, lr, batch, epochs, seed, mae, rmse, mase, wall_clock, hardware, created_at)
predictions(id, model_id, type, input_ref, result, created_at)
inspections(id, image_path, model_id, predicted_class, detections, confidence, created_at)
anomalies(id, meter_id, score, threshold, severity, created_at)
jobs(id, kind, params, state, attempts, started_at, finished_at, error)
9. Non-functional requirements
Area	Requirement
Dashboard TTI	under 2.5 s cold, under 800 ms warm
Inference p95 (vision, CPU)	under 4 s
Forecast p95	under 1.5 s
Training job timeout	20 min, then fail
Uptime during demo window	99% (pre-warmed containers)
Uploads	magic-byte validated, 10 MB, EXIF stripped
Secrets	env only, never in repo
Logging	structured JSON, job ID on every model call
Reproducibility	seeds, hashes, requirements.lock, pinned TF version
10. Security and privacy
JWT, 24-hour expiry, refresh via httpOnly cookie. Training routes admin-only.
Rate limit 60 req/min per user, 10 uploads/min.
Meter IDs are pseudonymous upstream; do not join to identifiable survey rows without stating the join key.
No household address, name, or coordinates exposed in any response.
Object storage is private; images served through signed URLs.
11. Risks
ID	Risk	Likelihood	Impact	Mitigation
R1	Meter attrition leaves too few usable series	High	High	Quality gates; fall back to D2 grid series and aggregated files
R2	Forecasting fails to beat seasonal-naive	Medium	High	Baseline-first build order; publish the negative result
R3	Render OOM on model load	High	Medium	Separate inference service; no TF import in the API process
R4	14.5 GB dataset exhausts disk	High	Medium	allow_patterns subsets only; start with two subsets
R5	Class imbalance collapses the CNN to majority class	High	Medium	Macro-F1 as primary; class weights; per-class recall reported
R6	Segmentation has no test split	Certain	Medium	15% carve-out from train, documented
R7	Five modules overrun 12 weeks	High	High	Cut lines pre-agreed per module, ordered by value
R8	GAN mode collapse	Medium	Medium	WGAN-GP; the utility test catches it
R9	Examiner asks for an ablation you did not run	Medium	High	Experiments pre-registered in section 6; nothing claimed without a row
12. Timeline and cut lines
Week	Milestone
1	Ingest D1, quality report, admission rate, split manifest
2	M1 complete; feature builder frozen
3-4	M4 baselines (persistence, seasonal-naive, LightGBM), then LSTM
5	M4 GRU; forecast API
6-7	M3 CNN and YOLO; inspection queue
8	M2 optimiser, init, and gradient lab
9	M5 VAE with the injection protocol
10	GAN; dashboard wiring
11	Deployment, hardening, pre-warm
12	Docs, demo script, buffer
Cut lines in order: U-Net → BiLSTM and RNN → GAN → D4/D5 profiling → activation sweep. No module is cut below its MVP row.

13. Acceptance and demo script
Ship is blocked unless all pass.

Every module returns a live prediction from a registered model.
Every metric table shows a baseline row.
LSTM and GRU beat seasonal-naive MASE on the majority of admitted meters.
VAE precision and recall reported against injected anomalies.
All 15 endpoints return the documented envelope, including on malformed input.
docs/DATA_CONTRACT.md, docs/MODEL_CARDS.md, docs/CREDITS.md exist and are current.
Demo runs in 8 minutes: overview → forecast with baseline overlay → upload a leaning-pole image → anomaly monitor → optimiser comparison → gradient plot. Backup is a recorded video plus a local Docker compose, in case venue wifi dies.

14. Stated limitations
Advisory only; a qualified engineer verifies any inspection output.
An anomaly is a flag, not evidence of theft or tampering.
One utility's network; transfer needs re-validation.
No calibrated physical measurements — lean and clearance are geometric proxies.
Outage periods are modelled as missing, so true zero-demand behaviour is under-represented.
Grid-level forecasts carry no weather covariates, which caps accuracy in monsoon months.
15. Open questions
Do D4 and D5 require a data-use agreement beyond the DOI landing page?
Is CPU-only inference acceptable for the demo, or is a GPU budget available?
What is the exam deadline, to fix the week-12 buffer?
Appendix A — Glossary
Term	Meaning
MASE	Mean absolute scaled error; scale-free, defined at zeros
Outage bin	3-minute sample where voltage or current indicates no supply
Admitted meter	Meter-series passing all quality gates in 3.3
Utility test	Training on synthetic data, evaluating on real data
Config hash	Digest of architecture, hyperparameters, and seed