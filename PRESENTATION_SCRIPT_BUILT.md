# SmartGrid AI — Production Presentation Script (Based on What We Built)
**Pacing:** Exactly 60 Seconds per Speaker (10 Minutes Total) | **Live Dashboard:** `http://127.0.0.1:8080`  
**Slide Mapping:** Slides 1 to 15  

---

## Team Roster & Slide Assignment

| # | Assigned Member | Role & Module | Corresponding Slide | Live System Reference |
|:---:|:---|:---|:---:|:---|
| **01** | **Sohan Ahmed (Leader)** | Project & Team Introduction | **Slides 1–2** | Live Dashboard Entry & Architecture Overview |
| **02** | **Angsu Dolai** | Practical Problem Statement | **Slides 3–4** | Grid Volatility & Inspection Inefficiencies |
| **03** | **Arunabha Maiti** | Engineered Solution & Objectives | **Slides 5–6** | 5 AI Modules (M1–M5) Single-Pane Console |
| **04** | **Binita Roy** | Datasets & Ingestion Architecture | **Slides 7–8** | Ingestion Engine, 65.5% Admission Gate & Contracts |
| **05** | **Dyutimoy Panda** | Household Electricity Prediction (ANN) | **Slide 9** | M1 Shallow & M2 Dynamics (MASE < 0.50 Benchmark) |
| **06** | **Enakhya Mandal** | Deep Neural Network Training (DNN) | **Slide 10** | Training Engine, Adam/SGD & SHA-256 Checksum |
| **07** | **Pritam** | AI-Based Infrastructure Inspection | **Slide 11** | Live Drone Photo Upload, PoleLeanCNN & U-Net |
| **08** | **Rittick Raj Mishra** | Electricity Demand Forecasting (Sequence) | **Slide 12** | M4 Autoregressive Sequence Model & Load Curves |
| **09** | **Sahil Ahmed** | Synthetic Data & Anomaly Detection (M5) | **Slide 13** | VAE Reconstruction & Isolation Forest Triage |
| **10** | **Supriyo Jana** | Live Verification, Deployment & Conclusion | **Slides 14–15** | 158 Passed Pytests, Docker Container & Impact |

---

## 01 Sohan Ahmed (Leader)
**Role:** Team Leader — Project Introduction  
**Slide Coverage:** Slides 1–2 (60 Seconds)  
**Live UI Anchor:** Dashboard Home / Login Screen (`http://127.0.0.1:8080/login`)  

* **0–20 sec:**  
  > *"Good morning, respected evaluators and peers. I am Sohan Ahmed, team leader for **SmartGrid AI**. Today, we are proud to present not just a conceptual proposal, but a fully functional, production-grade intelligence platform engineered by our ten-member team."*

* **20–40 sec:**  
  > *"Over the past weeks, we built an integrated system that connects raw smart meter telemetry with five distinct AI architectures — spanning neural demand forecasting, generative anomaly detection, and aerial computer vision inspection of distribution equipment."*

* **40–60 sec:**  
  > *"Everything we will show you today runs live on a single-port FastAPI and React Vite dashboard. I will now hand over to **Angsu Dolai**, who will present the critical operational challenges that drove our engineering choices."*

---

## 02 Angsu Dolai
**Role:** Grid Systems Analyst — Problem Statement  
**Slide Coverage:** Slides 3–4 (60 Seconds)  
**Live UI Anchor:** Overview KPI Cards & Telemetry Grid  

* **0–20 sec:**  
  > *"Electricity consumption is volatile and non-linear. Looking at daily averages blinds operators to dangerous intra-day surges. Without 15-minute granular forecasting, utilities cannot balance intermittent renewables or avoid brownouts during peak hours."*

* **20–40 sec:**  
  > *"Second, detecting power theft or equipment faults is difficult because what is normal for one feeder is an anomaly for another. Manually reviewing millions of smart meter data points is mathematically impossible without automated intelligence."*

* **40–60 sec:**  
  > *"Finally, utilities manage thousands of miles of physical distribution poles. Manually inspecting drone imagery creates massive backlogs. SmartGrid AI eliminates these bottlenecks. **Arunabha Maiti** will now introduce our engineered solution."*

---

## 03 Arunabha Maiti
**Role:** Systems Architect — Solution & Objectives  
**Slide Coverage:** Slides 5–6 (60 Seconds)  
**Live UI Anchor:** Live Console Navigation (Overview, Consumption, Models, Inspection, Anomalies)  

* **0–20 sec:**  
  > *"To solve this, we engineered a unified platform combining five specialized AI modules into one reactive dashboard. Module 1 establishes classical baselines, while Module 2 evaluates neural consumption forecasting with weather covariates."*

* **20–40 sec:**  
  > *"Module 3 deploys computer vision directly on aerial drone imagery for pole lean and vegetation encroachment. Module 4 runs autoregressive 24-hour sequence forecasts, and Module 5 uses generative models to isolate electrical transients."*

* **40–60 sec:**  
  > *"Our objective was zero fragmentation: giving field staff and control room operators one live dashboard with real-time graphs, bounding-box overlays, and automated work orders. **Binita Roy** will now walk through our datasets and ingestion pipeline."*

---

## 04 Binita Roy
**Role:** Data Pipeline Engineer — Ingestion & System Architecture  
**Slide Coverage:** Slides 7–8 (60 Seconds)  
**Live UI Anchor:** Overview Dashboard (`/overview`) — Total Meters: 84 | Admitted: 55 (65.5%)  

* **0–20 sec:**  
  > *"Our data pipeline in `src/smartgrid/data` processes multi-source data: high-frequency smart meter readings from Mathura and Bareilly, grid demand records, and aerial drone photos from the PD-Defect distribution dataset."*

* **20–40 sec:**  
  > *"We implemented strict **Physical Quality Gates**. Out of 84 candidate physical smart meters, our pipeline admitted **55 verified meters**, achieving a **65.5% admission rate** by enforcing continuity, non-negative power bounds, and 15-minute forward-fill resampling."*

* **40–60 sec:**  
  > *"Clean data feeds our analytics backend, which serves 1.42 million kilowatt-hours of telemetry to the frontend via secure, rate-limited REST APIs. **Dyutimoy Panda** will now explain our first machine learning module."*

---

## 05 Dyutimoy Panda
**Role:** Machine Learning Engineer — Household Electricity Prediction (ANN)  
**Slide Coverage:** Slide 9 (60 Seconds)  
**Live UI Anchor:** Model Lab (`/models` → M1 Ridge Baseline & M2 MLP)  

* **0–20 sec:**  
  > *"In Module M1 and M2, located in `src/smartgrid/models`, we predict household electricity use using Artificial Neural Networks. We began with a single-layer perceptron to prove the necessity of non-linear decision boundaries on the XOR phase test."*

* **20–40 sec:**  
  > *"We then engineered a multi-layer neural network with ReLU activations. Our models ingest historical load alongside exogenous variables like ambient temperature and solar irradiance, which directly influence cooling and heating demands."*

* **40–60 sec:**  
  > *"As shown in our live **Model Lab**, our trained network achieves a **Mean Absolute Scaled Error of 0.42**, outperforming the Seasonal-Naive benchmark floor by more than **$50\%$**. **Enakhya Mandal** will now explain our training and optimization pipeline."*

---

## 06 Enakhya Mandal
**Role:** Deep Learning Optimization Engineer — DNN Training  
**Slide Coverage:** Slide 10 (60 Seconds)  
**Live UI Anchor:** Model Lab Training Diagnostics & SHA-256 Checksum Badge  

* **0–20 sec:**  
  > *"In our deep neural network experiments in `M2_dynamics.ipynb`, we evaluated convergence dynamics across different optimizers. We benchmarked Adam against Stochastic Gradient Descent with momentum across identical train-validation splits."*

* **20–40 sec:**  
  > *"Adam converged three times faster, reaching lower validation loss without oscillating near sharp minima. We added Dropout at 0.3 and Batch Normalization to prevent the network from overfitting during sudden holiday usage shifts."*

* **40–60 sec:**  
  > *"To ensure enterprise reproducibility, our automated training cycle signs every model weight artifact with a **SHA-256 cryptographic manifest checksum**. **Pritam** will now demonstrate our computer vision infrastructure inspection system."*

---

## 07 Pritam
**Role:** Computer Vision Lead — AI-Based Infrastructure Inspection  
**Slide Coverage:** Slide 11 (60 Seconds)  
**Live UI Anchor:** Live Inspection Queue (`/inspection` → Drag & Drop Upload & Detection Canvas)  

* **0–20 sec:**  
  > *"Physical grid reliability depends on overhead distribution poles. In **Module M3**, located in `src/smartgrid/models/vision.py`, we built a complete computer vision triage system that you can test live on our dashboard right now."*

* **20–40 sec:**  
  > *"Operators can drag and drop any **real drone photo** into our **Inspection Queue**. First, our security guard strips EXIF metadata. Then, our detection head isolates the pole, while **PoleLeanCNN** calculates tilt angle deviation — flagging critical lean over $7.5^\circ$."*

* **40–60 sec:**  
  > *"Third, our **U-Net semantic segmenter** maps conductor wires and tree canopies, computing the Excess Green Index to measure vegetation encroachment risk. The entire pipeline executes in **under 250 milliseconds**, easily passing our p95 4-second requirement. **Rittick Raj Mishra** will now cover demand forecasting."*

---

## 08 Rittick Raj Mishra
**Role:** Time-Series Specialist — Electricity Demand Forecasting  
**Slide Coverage:** Slide 12 (60 Seconds)  
**Live UI Anchor:** Consumption Telemetry Curve (`/consumption`) — 24-Hour 96-Bucket Curve  

* **0–20 sec:**  
  > *"Grid dispatchers need accurate forward visibility. In **Module M4**, located in `src/smartgrid/models/sequence.py`, we implement autoregressive multi-horizon forecasting, taking seven days of 15-minute history to predict the next 24 hours."*

* **20–40 sec:**  
  > *"We compared Long Short-Term Memory networks and Gated Recurrent Units. The GRU maintained equivalent accuracy while reducing parameter count by 25%, accurately predicting both morning ramp-ups and evening cooking peaks."*

* **40–60 sec:**  
  > *"On our live **Consumption** screen, the SVG telemetry curve dynamically reports peak load at 31.2 kW and base load at 10.5 kW across feeders, enabling proactive spinning reserve dispatch. **Sahil Ahmed** will now explain synthetic data and anomaly detection."*

---

## 09 Sahil Ahmed
**Role:** AI Safety Engineer — Synthetic Data & Anomaly Detection (GAN & VAE)  
**Slide Coverage:** Slide 13 (60 Seconds)  
**Live UI Anchor:** Anomaly & Transient Monitor (`/anomalies`) — Incident Triage Table  

* **0–20 sec:**  
  > *"Rare electrical anomalies lack historical training examples. In **Module M5**, we implemented generative architectures in `src/smartgrid/models/generative_anomaly.py`. We trained a GAN where the generator creates synthetic load scenarios to augment training data."*

* **20–40 sec:**  
  > *"For online monitoring, we use a Variational Autoencoder paired with Isolation Forests. The VAE learns nominal load reconstructions; if an incoming telemetry batch exhibits high reconstruction loss, it is immediately flagged as anomalous."*

* **40–60 sec:**  
  > *"On our live **Anomaly Monitor**, operators can triage incidents in real time — such as an $18.4\%$ voltage drop or reverse power flow from unregistered solar feed-in. **Supriyo Jana** will now conclude with our results and deployment."*

---

## 10 Supriyo Jana
**Role:** QA & Deployment Lead — Results, Applications & Conclusion  
**Slide Coverage:** Slides 14–15 (60 Seconds)  
**Live UI Anchor:** Live Dashboard Terminal / Swagger Docs (`/docs`) & Docker Setup  

* **0–20 sec:**  
  > *"In summary, SmartGrid AI is fully realized. Our live dashboard connects all five AI modules into an operational console with role-based JWT authentication, interactive model sliders, real-time image triage, and automated work orders."*

* **20–40 sec:**  
  > *"To ensure rock-solid production quality, we implemented a rigorous testing suite: **158 unit and end-to-end tests** passing across data quality, model inference, and API security tiers. The platform is containerized with a multi-stage **Dockerfile** ready for cloud deployment."*

* **40–60 sec:**  
  > *"By uniting smart meter analytics with automated drone vision, SmartGrid AI empowers utilities to prevent outages, protect assets, and accelerate the clean energy transition. Thank you for your time, and we are now delighted to take your questions!"*

---

### 💡 Rehearsal Tips for the Team
1. **Timing Checkpoints:** Each speaker has three 20-second blocks. Keep your eye on the timer — hand off to the next teammate at the 50–55 second mark.
2. **Live Visual Sync:** Whenever you mention a module, point to the projector where the corresponding screen (`/overview`, `/models`, `/inspection`, `/anomalies`) is live on `http://127.0.0.1:8080`.
3. **Pritam (Speaker 07):** Show the live **drag-and-drop real image upload** on `/inspection` during your segment — it directly proves your module is working live in production.

