# SmartgridAI — 10-Member Team Presentation Script

**Project Title:** SmartgridAI: Autonomous Grid Telemetry, Machine Learning Inference & Computer Vision Inspection Platform  
**Target Duration:** ~15–20 minutes (~1.5 to 2 minutes per speaker)  
**Live Demo URL:** `http://127.0.0.1:8080`  

---

## Presentation Structure Overview

| Member | Role | Topic / Segment | Live UI / Visual Anchor |
|:---:|:---|:---|:---|
| **1** | Project Lead / Anchor | Problem Statement, Motivation & Architectural Overview | Login Screen & System Architecture Diagram |
| **2** | Data Engineer | Ingestion Pipeline, Contract Validation & Quality Gates | Overview KPI Cards (`/overview`) |
| **3** | ML Engineer (M1) | M1: Shallow Learning Baselines & Linear Benchmarks | Model Lab (`/models` - M1 Ridge) |
| **4** | ML Engineer (M2) | M2: Dynamics & Exogenous Neural Forecaster | Model Lab (`/models` - M2 MLP with Weather Sliders) |
| **5** | Computer Vision Engineer (M3) | M3: Drone Vision, Pole Detection & Lean Angle CNN | Inspection Queue (`/inspection` - Real Image Upload) |
| **6** | CV & Geospatial Analyst (M3) | M3: Vegetation Corridor Encroachment U-Net Segmentation | Inspection Queue (`/inspection` - U-Net Overlay & Work Order) |
| **7** | Deep Learning Engineer (M4) | M4: Deep Autoregressive Sequence Forecasting | Consumption Telemetry Curve (`/consumption`) |
| **8** | AI Safety & Anomaly Analyst (M5) | M5: Generative Anomaly Detection & Grid Transient Triage | Anomaly Monitor (`/anomalies`) |
| **9** | Backend & Security Architect | FastAPI Microservice, RBAC, HMAC-SHA256 JWT & Rate Limiting | API Swagger (`/docs`) & Auth Flow |
| **10** | QA Lead & DevOps Engineer | 158-Test Verification Suite, Docker Deployment & Conclusion | Terminal Pytest Suite & Production Roadmap |

---

## Detailed Speaking Scripts (Member 1 through 10)

### 👤 Member 1: Project Lead & Anchor
* **Role:** Project Lead / Anchor  
* **Slide / Screen:** Login Screen (`/login`) & Architecture Diagram  
* **Time:** ~1.5 mins  

#### 🎙️ Speaking Script:
> "Good morning, respected professors and peers. Welcome to the presentation of **SmartgridAI**.
>
> Today's electrical grids are facing unprecedented challenges: the integration of intermittent renewable energy, rapid EV adoption, aging physical infrastructure, and unpredictable weather extremes. Traditional Supervisory Control and Data Acquisition (SCADA) systems are reactive and fragmented. Utilities struggle to ingest high-frequency telemetry while simultaneously triaging thousands of physical assets across vast geographic territories.
>
> To solve this, our 10-person team engineered **SmartgridAI** — an end-to-end, production-grade intelligence platform that unifies high-frequency smart meter ingestion, five distinct machine learning model architectures (M1 through M5), and autonomous drone-based computer vision inspection into a unified, responsive dashboard.
>
> Our solution bridges the gap between raw edge sensors and real-time operator decision-making. Over the next fifteen minutes, our team will walk you through every tier of this system — from mathematical data quality gates and neural load forecasting to computer vision triage and microservice security.
>
> I’ll now hand over to **[Member 2]**, who will explain our data ingestion pipeline and quality assurance protocols."

---

### 👤 Member 2: Data Engineer
* **Role:** Data Engineering & Pipeline Lead  
* **Slide / Screen:** Grid Overview KPI Cards (`/overview`)  
* **Time:** ~1.5 mins  

#### 🎙️ Speaking Script:
> "Thank you, **[Member 1]**.
>
> In real-world grid operations, sensor data is notoriously noisy — filled with missing intervals, duplicate timestamps, negative power spikes, and packet dropouts. Feeding corrupt data into downstream deep learning models guarantees failure.
>
> In SmartgridAI, we implemented a rigorous **Multi-Stage Ingestion Quality Gate** inside `src/smartgrid/data`:
> 1. **Schema & Contract Validation:** Every incoming smart meter CSV is parsed against physical electrical contracts: active power ($P$), reactive power ($Q$), voltage ($V$), and power factor.
> 2. **Quality Gate Filtering:** We evaluate meters on continuity and variance. Out of 84 candidate physical meters in our dataset, **55 met our stringent physical criteria**, achieving a verified **65.5% admission rate**.
> 3. **15-Minute Deterministic Resampling:** Using forward-fill interpolation capped at 3 consecutive intervals, telemetry is normalized into 96 discrete daily time buckets.
>
> As shown on the **Grid Overview** screen, this engine tracks **1.42 million kilowatt-hours** of aggregate demand across admitted feeders while continuously flagging harmonic anomalies.
>
> Now, **[Member 3]** will demonstrate how this curated data powers our M1 baseline models."

---

### 👤 Member 3: ML Engineer (M1: Shallow Baseline)
* **Role:** Classical Machine Learning Engineer  
* **Slide / Screen:** Model Lab (`/models` → M1 Ridge Regularized Baseline)  
* **Time:** ~1.5 mins  

#### 🎙️ Speaking Script:
> "Thank you, **[Member 2]**.
>
> In applied AI, a complex deep neural network is unjustified unless it mathematically outperforms rigorous, computationally lightweight baselines. This is where **Module M1: Shallow Learning** comes in.
>
> Located in `src/smartgrid/models/shallow.py`, M1 establishes our benchmark foundations:
> 1. **Linear & Ridge Regularization:** We formulated $L_2$-penalized Ridge regression to model primary diurnal load trends, preventing coefficient explosion when multi-collinear features appear.
> 2. **Non-Linear Capability Verification:** We subjected our linear baselines and a shallow multilayer perceptron to the canonical XOR classification boundary. This proved mathematically that linear models fail on complex phase interactions, establishing the strict necessity for non-linear architectures.
> 3. **Seasonal-Naive Floor:** We implemented a 96-step Seasonal-Naive baseline as our primary quality hurdle. Every advanced model in our suite is required to beat a **Mean Absolute Scaled Error (MASE) of 1.0**.
>
> With these classical bounds solidified, **[Member 4]** will present our M2 Neural Forecaster incorporating dynamic weather variables."

---

### 👤 Member 4: ML Engineer (M2: Dynamics & Weather Forecaster)
* **Role:** Dynamics & Forecasting Engineer  
* **Slide / Screen:** Model Lab (`/models` → M2 MLP with Temperature & Solar Sliders)  
* **Time:** ~1.5 mins  

#### 🎙️ Speaking Script:
> "Thank you, **[Member 3]**.
>
> Power grid consumption cannot be predicted by historical load alone — human behavior is deeply tied to external atmospheric dynamics. When temperatures spike in summer, air conditioning load surges; conversely, peak solar irradiance offsets grid draw through distributed rooftop photovoltaics.
>
> In **Module M2: Dynamics**, located in `src/smartgrid/models/dynamics.py`, we developed a deep Multilayer Perceptron incorporating **exogenous atmospheric covariates**:
> - We integrate ambient temperature ($10^\circ\text{C}$ to $45^\circ\text{C}$) and solar irradiance ($0$ to $1000\text{ W/m}^2$) directly into the hidden state representations.
> - As you can see in our live **Model Lab**, moving the temperature slider upwards dynamically recalculates the predicted mean load and shifts the $95\%$ confidence interval upward, simulating grid thermal stress.
> - M2 achieves an outstanding **MASE score of 0.42 to 0.55**, outperforming the naive baseline by more than **$50\%$**.
>
> Furthermore, every trained model artifact is signed with an automated **SHA-256 manifest checksum** to prevent weight tampering in production.
>
> Next, **[Member 5]** will introduce our computer vision inspection pipeline."

---

### 👤 Member 5: Computer Vision Engineer (M3: Detection & Tilt)
* **Role:** Computer Vision Engineer  
* **Slide / Screen:** Inspection Queue (`/inspection` → Upload Real Image & Detection Overlay)  
* **Time:** ~1.5 mins  

#### 🎙️ Speaking Script:
> "Thank you, **[Member 4]**.
>
> While digital meters monitor electricity flow, physical grid integrity depends on millions of overhead wooden and concrete utility distribution poles. Extreme windstorms and soil erosion cause structural leaning, which precedes pole collapse and catastrophic wildfire ignition.
>
> In **Module M3: Vision**, located in `src/smartgrid/models/vision.py`, we engineered an automated aerial inspection pipeline:
> 1. **Security & Sanitization Guard:** Incoming drone payloads are scanned for JPEG/PNG magic bytes, checked against a 10 MB ceiling, and stripped of all EXIF/GPS metadata via fresh canvas recreation.
> 2. **Object Detection Head:** Using localized gradient variance and bounding box proposals, our detector identifies distribution poles with an **mAP@0.5 greater than 0.85**.
> 3. **PoleLeanCNN Architecture:** A 4-block Convolutional Neural Network with batch normalization and adaptive pooling classifies pole alignment into three classes: *Straight*, *Leaned*, and *Severe Tilt*.
>
> As shown on the screen, operators can drag and drop any **real drone image** directly into our **Inspection Queue**. Within 200 milliseconds, the UI overlays bounding boxes and reports the precise tilt deviation — here, $7.6^\circ$ — flagging it for crew intervention.
>
> **[Member 6]** will now explain our vegetation encroachment segmentation system."

---

### 👤 Member 6: CV & Geospatial Analyst (M3: Segmentation)
* **Role:** Semantic Segmentation & Risk Analyst  
* **Slide / Screen:** Inspection Queue (`/inspection` → U-Net Metrics & Work Order)  
* **Time:** ~1.5 mins  

#### 🎙️ Speaking Script:
> "Thank you, **[Member 5]**.
>
> Beyond pole tilt, tree branch encroachment into high-voltage conductors is the number one cause of distribution line flashovers and outages during seasonal growth.
>
> To automate vegetation risk assessment:
> 1. **VegetationConductorUNet:** We built a skip-connected U-Net architecture that executes 3-class semantic pixel segmentation: Background, High-Voltage Conductor Wire, and Vegetative Canopy.
> 2. **Mathematical Encroachment Metric:** The engine computes the **Excess Green Index** ($2G - R - B$) to isolate foliage, extracts high-frequency Sobel gradients to trace linear wire conductors, and runs a morphological dilation corridor of 12 pixels around each line.
> 3. **Automated Work Order Generation:** If the encroachment ratio exceeds safety thresholds, the platform automatically generates an engineering maintenance directive — as seen here: *'WORK ORDER #TR-8821: Dispatch crew for guy-wire tensioning and canopy clearance.'*
> 4. **Inference Latency:** The entire composite pipeline completes in **under 250 milliseconds**, easily beating our project requirement of **p95 latency under 4.0 seconds**.
>
> Now, **[Member 7]** will guide us through M4: Deep Sequence Modeling."

---

### 👤 Member 7: Deep Learning Engineer (M4: Sequence Predictor)
* **Role:** Time-Series Deep Learning Engineer  
* **Slide / Screen:** Consumption Telemetry Curve (`/consumption`)  
* **Time:** ~1.5 mins  

#### 🎙️ Speaking Script:
> "Thank you, **[Member 6]**.
>
> Single-step forecasting cannot capture multi-hour grid momentum or morning and evening ramp-up peaks. This brings us to **Module M4: Sequence Modeling**, located in `src/smartgrid/models/sequence.py` and `notebooks/M4_sequence.ipynb`.
>
> In M4, we focus on **autoregressive multi-horizon forecasting**:
> 1. **Temporal Horizon:** We model sequential dependencies across 96 time steps — representing a complete 24-hour cycle of 15-minute intervals.
> 2. **Sequential Memory:** Using recurrent cell dynamics, the model learns the diurnal cadence: the low base load at 04:00, the sharp residential morning peak at 08:00, the midday solar displacement dip, and the evening cooking and heating peak between 18:00 and 21:00.
> 3. **Live Visualization:** As demonstrated on our **Consumption Curve** page, operators can toggle between Feeder A, B, and C. The responsive SVG chart computes the peak load (31.2 kW), base load (10.5 kW), and average demand in real-time.
>
> Now, **[Member 8]** will present our M5 Generative Anomaly Detection system."

---

### 👤 Member 8: AI Safety & Anomaly Analyst (M5: Generative Anomaly)
* **Role:** Anomaly Detection & Grid Reliability Analyst  
* **Slide / Screen:** Anomaly & Transient Monitor (`/anomalies`)  
* **Time:** ~1.5 mins  

#### 🎙️ Speaking Script:
> "Thank you, **[Member 7]**.
>
> In a modern smart grid, anomalies take many forms: sudden voltage sags from lightning strikes, equipment insulation degradation, and illegal or unregistered solar export known as *Reverse Power Flow*.
>
> In **Module M5: Generative Anomaly Detection** (`src/smartgrid/models/generative_anomaly.py`), we deploy an ensemble approach:
> 1. **Isolation Forest & Statistical Z-Scores:** High-dimensional telemetry feature vectors are evaluated for path length anomalies, isolating sudden phase deviations in real time.
> 2. **Harmonic Outlier Detection:** We monitor Total Harmonic Distortion (THD). If harmonic distortion spikes above $8\%$, the event is classified as equipment degradation.
> 3. **Triage Table Interface:** On our **Anomaly Monitor** screen, operators have an interactive dispatch center. Each incident displays its Feeder ID, anomaly classification, magnitude (such as a $-18.4\%$ voltage drop), model confidence percentage, and investigation status.
>
> Next, **[Member 9]** will present the backend architecture and security protocols that hold this system together."

---

### 👤 Member 9: Backend & Security Architect
* **Role:** Backend & Cloud Security Architect  
* **Slide / Screen:** FastAPI OpenAPI `/docs` or Architecture Diagram  
* **Time:** ~1.5 mins  

#### 🎙️ Speaking Script:
> "Thank you, **[Member 8]**.
>
> A mission-critical energy platform demands enterprise-grade API performance, strict security, and modularity.
>
> Our backend is engineered using **FastAPI** in `backend/app`:
> 1. **Modular Architecture:** We decoupled logic into 10 modular routers: `/analytics`, `/ann`, `/forecast`, `/models`, `/vision`, `/infrastructure`, `/generative`, `/export`, and `/auth`.
> 2. **Enterprise RBAC & Security:** All private endpoints enforce a three-tier Role-Based Access Control hierarchy: *Viewer*, *Analyst*, and *Administrator*.
> 3. **HMAC-SHA256 Signed JWTs:** Authentication generates cryptographically secure JSON Web Tokens with strict 24-hour expirations.
> 4. **Standardized Envelopes:** All responses follow strict Pydantic `DataEnvelope` and `ErrorEnvelope` schemas with standardized error codes, preventing data leakage.
> 5. **Single-Port Unified Host:** Through Starlette static mounting, the backend serves both the JSON REST API and the production React Single Page Application on a single unified port, eliminating CORS vulnerabilities.
>
> I’ll now hand over to **[Member 10]** to cover our test verification suite and production deployment."

---

### 👤 Member 10: QA Lead & DevOps Engineer
* **Role:** Quality Assurance Lead & DevOps Engineer  
* **Slide / Screen:** Terminal showing Pytest Output & Docker Architecture  
* **Time:** ~1.5 mins  

#### 🎙️ Speaking Script:
> "Thank you, **[Member 9]**.
>
> To guarantee that SmartgridAI is production-ready and free of regressions, we built a comprehensive test harness:
> 1. **158 Passed Unit & E2E Tests:** Using Pytest and pytest-asyncio, our test suite validates four distinct testing tiers:
>    - *Tier 1:* Feature coverage of data contracts and quality gates.
>    - *Tier 2:* Boundary conditions, corrupt payloads, and rate limits.
>    - *Tier 3:* Cross-feature integration between models and API envelopes.
>    - *Tier 4:* Real-world automated retraining hash checksum verification cycles.
> 2. **Production Containerization:** We created a multi-stage `Dockerfile` and `docker-compose.yml`. Stage 1 compiles the Vite React bundle using Node 20; Stage 2 deploys a lightweight Python 3.11 container with automated health checks, ready for AWS, GCP, or on-premise deployment.
> 3. **Summary:** In SmartgridAI, we have delivered a robust, mathematically grounded, and visually polished platform that demonstrates the power of autonomous AI in critical infrastructure.
>
> On behalf of all 10 members of our team, thank you for your time. We are now open for questions!"

---

## Quick Q&A Cheat Sheet (For Any Team Member)

| Anticipated Question | Recommended Answer | Speaker to Handle |
|:---|:---|:---:|
| **"Why use Ridge and MLP when Transformers exist?"** | "For tabular 15-minute smart meter load profiles, Ridge and MLPs with exogenous weather features achieve MASE < 0.50 with sub-5ms latency, avoiding the massive parameter overhead and training data hunger of transformers." | **Member 3 / 4** |
| **"How do you handle privacy on drone photos?"** | "Our `validate_and_sanitize_image` function strips all EXIF, GPS, and camera hardware tags by re-rendering raw bytes onto a fresh RGB PIL canvas before model inference." | **Member 5** |
| **"What happens when a new meter is added?"** | "The automated retraining pipeline reads `retrain_manifest.json`, validates the SHA-256 hash of the new tranche, tests across 3 random seeds, and only deploys if the MASE beats the 1.0 floor." | **Member 2 / 10** |
| **"How is authentication secured?"** | "We use HMAC-SHA256 JWT tokens with role hierarchies: Viewers can read telemetry, Analysts can run model inference, and Administrators can dispatch work orders and access audit packs." | **Member 9** |

