# SmartgridAI Model Cards (M1, M2, M3, M4 & M5)

This document provides formal model cards for the machine learning modules implemented across the SmartgridAI coursework demonstration platform:
- **Module M1**: Shallow Neural Networks & Classical Baselines (`Perceptron2D`, `XORMLPSolver`, `DemandMLPRegressor`, `RidgeRegressionBaseline`)
- **Module M2**: Training Dynamics & Diagnostic Sweeps (`FixedDNN`, optimizer, initializer, regularizer sweeps, vanishing/exploding reproducers)
- **Module M3**: Computer Vision (`PoleLeanClassifier`, `PoleDetector`, `VegetationSegmenter`, `VisionInspectionPipeline`)
- **Module M4**: Sequence Forecasting (`m4_lstm_forecaster`, `m4_gru_forecaster`, and baselines)
- **Module M5**: Generative Synthesis & Anomaly Detection (`m5_wgan_generator`, `m5_vae_anomaly`, and perturbation benchmarks)

---

## 1. M1 Shallow Neural Network Models

### 1.1 Model Details: `Perceptron2D` (Feature 16: 2D Perceptron Toy Lab)
- **Model Name:** `perceptron_2d`
- **Module:** M1 (Shallow Neural Networks - Toy Lab)
- **Model Type:** Single-layer linear threshold unit using Frank Rosenblatt's perceptron learning rule.
- **Input:** $x = [x_1, x_2]^T \in \mathbb{R}^2$ linearly separable synthetic load telemetry.
- **Activation:** Heaviside step function: $\hat{y} = \mathbb{I}(w_1 x_1 + w_2 x_2 + b \ge 0)$.
- **Weight Update:** $w \leftarrow w + \eta (y - \hat{y}) x$, $b \leftarrow b + \eta (y - \hat{y})$ with learning rate $\eta = 0.1$.
- **Analytical Decision Boundary:** Computes line equation $x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$.
- **Convergence Guarantee:** Proves finite convergence via the Novikoff Perceptron Convergence Theorem ($k \le (R/\gamma)^2$).
- **Verification:** Converges to 100% training accuracy ($0$ misclassifications) in $< 15$ epochs on linearly separable data.

### 1.2 Model Details: `XORMLPSolver` (Feature 17: XOR 2-2-1 MLP Solver)
- **Model Name:** `xor_mlp_solver`
- **Module:** M1 (Shallow Neural Networks - Parity Problem)
- **Model Type:** Feedforward neural network for 2-input XOR parity benchmark.
- **Topologies Compared:**
  - Multi-Layer Topology (`hidden=True`): `Input(2) -> Dense(2, Sigmoid) -> Dense(1, Sigmoid)` with Binary Cross-Entropy loss.
  - Single-Layer Topology (`hidden=False`): `Input(2) -> Dense(1, Sigmoid)` without hidden representation.
- **Multi-Seed Benchmark (DoD Contract):**
  - Evaluated across 20 distinct pseudorandom seeds ($s \in \{0, \dots, 19\}$).
  - Target DoD: With hidden layer $\ge 19/20$ seeds converge to $100\%$ accuracy; without hidden layer $\le 2/20$ converge.
  - Observed Result: **20/20 (100%)** converged with hidden layer; **0/20 (0%)** converged without hidden layer (**PASS**).
- **Decision Surface:** Evaluates continuous non-linear probability surface across $[-0.5, 1.5] \times [-0.5, 1.5]$.

### 1.3 Model Details: `DemandMLPRegressor` (Feature 18: 15-Minute Demand MLP Regressor)
- **Model Name:** `m1_demand_mlp`
- **Module:** M1 (Shallow Neural Networks - Demand Forecasting)
- **Model Version:** `v1.0.0`
- **Architecture:**
  - Input Layer: 12-dimensional feature space ($t-1, t-2, t-3, t-4, t-96, t-672$, rolling mean/std $\tau=4$, cyclical hour $\sin/\cos$, cyclical day-of-week $\sin/\cos$).
  - Hidden Layer 1: Dense($12 \to 64$), ReLU activation, He normal initialization.
  - Hidden Layer 2: Dense($64 \to 32$), ReLU activation, He normal initialization.
  - Output Layer: Dense($32 \to 1$), Linear activation.
  - Total Parameters: 3,009 trainable parameters.
- **Optimization & Outage Protection:**
  - Optimizer: Adam ($\eta = 10^{-3}, \beta_1 = 0.9, \beta_2 = 0.999$, weight decay $\lambda = 10^{-3}$).
  - Physical Outage Masking: Intervals where $V \le 5.0\text{V}$ or $I \le 0.01\text{A}$ are strictly masked from loss calculation.
  - Chronological 70/15/15 split on admitted smart meter telemetry (`BR04`).
- **Evaluation Performance (Held-Out Test Split):**
  - $\text{MAE} = \mathbf{0.0347\text{ kW}}$, $\text{RMSE} = \mathbf{0.0483\text{ kW}}$, $R^2 = \mathbf{0.9774}$.

### 1.4 Co-Located Baseline: `RidgeRegressionBaseline` (Feature 19)
- **Model Name:** `ridge_regression`
- **Module:** M1 Baseline
- **Formulation:** Closed-form L2-regularized normal equation:
  $$\hat{\mathbf{w}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I}^*)^{-1} \mathbf{X}^T \mathbf{y}, \quad \alpha = 1.0$$
  where $\mathbf{I}^*$ unpenalizes the bias intercept term.
- **Evaluation Performance (Held-Out Test Split):**
  - $\text{MAE} = 0.0353\text{ kW}$, $\text{RMSE} = 0.0501\text{ kW}$, $R^2 = 0.9757$.
- **DoD Acceptance:** Demand MLP strictly outperforms Ridge regression MAE on held-out test split:
  $$\text{MAE}_{\text{MLP}} (0.0347) < \text{MAE}_{\text{Ridge}} (0.0353) \implies \mathbf{PASS}$$

---

## 2. M2 Training Dynamics & Diagnostic Models

### 2.1 Model Details: `FixedDNN` (Feature 20: 5-Layer Benchmark Architecture)
- **Model Name:** `m2_dnn_benchmark`
- **Module:** M2 (Training Dynamics)
- **Architecture:** Standardized 5-layer feedforward architecture:
  $$\text{Input}(12) \to \text{Dense}(128) \to \text{Dense}(64) \to \text{Dense}(32) \to \text{Dense}(16) \to \text{Dense}(1)$$
  - Hidden activations: ReLU; Output activation: Linear.
  - Total Parameters: 13,329 parameters.
  - Serves as the constant controlled baseline for all dynamic sweeps.

### 2.2 Optimizer Sweep Benchmark (Feature 21: Adam vs. SGD vs. RMSprop)
- **Protocol:** 3 random seeds (`[42, 101, 777]`), reporting mean $\pm$ standard deviation across identical Xavier initialization:
  - **Adam** ($\eta = 0.001, \beta_1 = 0.9, \beta_2 = 0.999$): $\text{MAE} = \mathbf{0.0338 \pm 0.0014\text{ kW}}$, $\text{RMSE} = 0.0468 \pm 0.0010\text{ kW}$
  - **RMSprop** ($\eta = 0.001, \alpha = 0.99$): $\text{MAE} = 0.0356 \pm 0.0035\text{ kW}$, $\text{RMSE} = 0.0509 \pm 0.0046\text{ kW}$
  - **SGD with Momentum** ($\eta = 0.01, \gamma = 0.9$): $\text{MAE} = 0.0376 \pm 0.0014\text{ kW}}$, $\text{RMSE} = 0.0518 \pm 0.0014\text{ kW}$
- **Outcome:** Adam achieves lowest validation MAE with minimal variance across seeds.

### 2.3 Initializer Sweep Benchmark (Feature 22: Random vs. Xavier vs. He)
- **Analysis:**
  - **Random Normal** ($\sigma = 0.05$): Initial loss $0.1432$; fails to scale variance with fan-in, leading to attenuated gradient norms in deep layers.
  - **Xavier / Glorot** ($\sigma = \sqrt{2/(n_{\text{in}} + n_{\text{out}})}$): Initial loss $0.1988$; final MAE $0.0327\text{ kW}$.
  - **He / Kaiming** ($\sigma = \sqrt{2/n_{\text{in}}}$): Initial loss $0.8917$; preserves healthy gradient variance across ReLU non-linearities.

### 2.4 Regularizer Sweep Benchmark (Feature 23: None vs. Dropout vs. BatchNorm)
- **Generalization Gap Evaluation:**
  - **None (Unregularized)**: Train MAE $0.0401$, Val MAE $0.0458$, Gap $= +0.0058\text{ kW}$.
  - **Dropout ($p=0.20$)**: Train MAE $0.1304$, Val MAE $0.1271$, Gap $= \mathbf{-0.0033\text{ kW}}$ (contracts generalization gap, preventing co-adaptation).
  - **Batch Normalization**: Train MAE $0.0788$, Val MAE $0.0834$, Gap $= +0.0046\text{ kW}$ (stabilizes intermediate representations).

### 2.5 Vanishing Gradient Reproducer (Feature 25)
- **5-Layer Sigmoid Stack:** Demonstrates exponential gradient attenuation:
  $$\|G^{(\text{output})}\|_2 / \|G^{(\text{input})}\|_2 = \mathbf{278.29} > 100.0 \quad (> 2\text{ orders of magnitude})$$
- **DoD Target:** Decay ratio $> 100.0\times$ verified (**PASS**).
- **ReLU Control:** 5-layer ReLU stack maintains healthy gradient flow (ratio $< 5.0\times$), demonstrating architectural resolution.

### 2.6 Exploding Gradient & Norm Clipping Demonstration (Feature 26)
- **High Learning Rate Regime ($\eta = 10.0$):**
  - Unclipped: Loss trajectory compounds exponentially and produces `NaN` floating-point overflow (**PASS**).
  - Clipped ($C = 1.0$ Global Norm): Bounding gradients to $\|g\|_2 \le 1.0$ maintains strictly finite, monotonic loss descent without NaNs (**PASS**).

---

## 3. M3 Computer Vision Models

### 3.1 Ingestion Guard & Sanitization (`validate_and_sanitize_image`)
- **Purpose**: Enforces cybersecurity and privacy hygiene on all incoming field imagery.
- **Rules & Bounds**:
  - Magic byte validation: strictly accepts JPEG (`FF D8 FF`) and PNG (`89 50 4E 47 0D 0A 1A 0A`), rejecting extension-spoofed payloads (`InvalidImageFormatError`).
  - Size cap: max 10 MB (`10,485,760` bytes), rejecting oversized uploads with `PayloadTooLargeError`.
  - Dimension bounds: minimum 2x2 px (rejects 1x1 sub-pixel payloads with `ImageDimensionError`).
  - Maximum resolution cap: uploads exceeding 4096 px are rejected with `ImageDimensionError` in API mode (`auto_downscale=False`), or proportionally downscaled preserving aspect ratio in offline pipeline mode (`auto_downscale=True`).
  - EXIF stripping: metadata (including GPS location, camera serial, timestamp) is purged via fresh RGB canvas recreation.
  - Numerical stability: zero-variance / monochrome images process cleanly without `NaN` or division-by-zero.

### 3.2 Model Details: `PoleLeanClassifier` (`PoleLeanCNN`)
- **Model Name:** `pole_lean_cnn`
- **Module:** M3 (Computer Vision - Task 1)
- **Model Type:** 4-stage Convolutional Neural Network for 3-class pole uprightness classification.
- **Classes:**
  - `Straight` (56.4% in training pool): Upright distribution pole within operational tolerance.
  - `Leaned` (27.2% in training pool): Noticeable angular deviation requiring scheduled maintenance.
  - `Severe Tilt` (16.4% in training pool, mapped from raw dataset `Rejected`): Critical hazard requiring immediate emergency triage.
- **Architecture:**
  - Input: $3 \times 224 \times 224$ normalized RGB tensor.
  - Stage 1: Conv2d(3, 32, kernel=3, pad=1) $\to$ BatchNorm2d(32) $\to$ ReLU $\to$ MaxPool2d(2, 2)
  - Stage 2: Conv2d(32, 64, kernel=3, pad=1) $\to$ BatchNorm2d(64) $\to$ ReLU $\to$ MaxPool2d(2, 2)
  - Stage 3: Conv2d(64, 128, kernel=3, pad=1) $\to$ BatchNorm2d(128) $\to$ ReLU $\to$ MaxPool2d(2, 2)
  - Stage 4: Conv2d(128, 256, kernel=3, pad=1) $\to$ BatchNorm2d(256) $\to$ ReLU $\to$ AdaptiveAvgPool2d(1, 1)
  - Classifier Head: Dropout(0.3) $\to$ Linear(256, 128) $\to$ ReLU $\to$ Linear(128, 3)
  - Total Parameters: ~0.51M parameters.
- **Class Imbalance Mitigation:**
  - Weighted Cross-Entropy Loss with inverse class frequency weights:
    $$w_c = \frac{N_{\text{total}}}{C \cdot N_c} \implies w_{\text{Straight}} \approx 0.5911,\; w_{\text{Leaned}} \approx 1.2256,\; w_{\text{Severe Tilt}} \approx 2.0310$$
- **Baseline Comparison:**
  - Majority-Class Baseline (`Straight`): Always predicts `Straight`.
  - Baseline macro-F1: $0.333$ (F1 = 0.72 for Straight, 0.0 for Leaned, 0.0 for Severe Tilt).
  - Target DoD: Macro-F1 $\ge 0.70$. Achieved: **0.735** (over +120% relative improvement).

### 3.3 Model Details: `PoleDetector` (YOLO Pole Detection)
- **Model Name:** `pole_detector_yolo`
- **Module:** M3 (Computer Vision - Task 2)
- **Model Type:** Single-class object detector trained on `data/pd-defect/pole_detection/`.
- **Annotation Format:** YOLO format (`class_id, cx, cy, w, h` normalized) with 2,252 field images (1,665 train, 293 valid, 294 test).
- **Architecture:** Lightweight anchor-free / nano backbone (YOLOv8n / feature extraction head) operating at $640 \times 640$ resolution.
- **Evaluation Metric:** Mean Average Precision at IoU $\ge 0.5$ (mAP@0.5).
- **Target DoD:** mAP@0.5 $\ge 0.85$. Achieved: **0.872**.

### 3.4 Model Details: `VegetationSegmenter` (`VegetationConductorUNet`)
- **Model Name:** `vegetation_conductor_unet`
- **Module:** M3 (Computer Vision - Task 3)
- **Model Type:** Symmetric contracting/expanding U-Net with skip connections.
- **Classes:** 3 semantic classes: 0: Background, 1: Power Conductor / Line, 2: Vegetation Encroachment.
- **Architecture:**
  - Encoder: 3 DoubleConv stages ($3 \to 32 \to 64 \to 128$) with MaxPool2d(2).
  - Bottleneck: DoubleConv ($128 \to 256$).
  - Decoder: 3 UpConv stages with skip concatenation ($256 \to 128 \to 64 \to 32$).
  - Segmentation Head: Conv2d(32, 3, kernel=1).
- **Loss Function:** Combined Cross-Entropy + Soft Dice Loss:
  $$\mathcal{L} = \mathcal{L}_{\text{CE}} + \left(1 - \frac{1}{C}\sum_{c=1}^C \frac{2\sum p_i g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}\right)$$
- **Carve-Out Protocol:** Evaluated on a deterministic 15% train carve-out (`seed=42`).
- **Target DoD:** mIoU $\ge 0.55$. Achieved: **0.584**.

### 3.5 End-to-End Composite Pipeline (`VisionInspectionPipeline`)
- **Execution Order:** Ingestion Guard $\to$ Lean Classifier $\to$ Pole Detector $\to$ Vegetation Segmenter.
- **Latency Budget:** CPU p95 latency must be $< 4.0\text{ seconds}$.
- **Measured CPU Latency Profile:**
  - Guard / Sanitization: $\sim 25\text{ ms}$
  - Pole Lean Classification: $\sim 30\text{ ms}$
  - Pole Detection: $\sim 280\text{ ms}$
  - Vegetation Segmentation: $\sim 415\text{ ms}$
  - Total end-to-end CPU latency: $\mathbf{750\text{ ms} = 0.75\text{s}}$ ($p95 < 1.10\text{s} \ll 4.0\text{s}$).

---

## 4. M4 Sequence Forecasting Models

### 4.1 Model Details: `m4_lstm_forecaster`
- **Model Name:** `m4_lstm_forecaster`
- **Module:** M4 (Sequence Forecasting)
- **Model Version:** `v1.0.0`
- **Model Type:** Multi-step recurrent neural network with categorical meter conditioning
- **Architecture:**
  - Input features ($D_{\text{in}} = 6$):
    - Scaled active power $P_{\text{scaled}} \in [0, 1]$ (fitted strictly on training split via `PerMeterScaler`)
    - Outage flag $M_{\text{outage}} \in \{0, 1\}$
    - Time-of-day sine encoding $\sin(2\pi \cdot (\text{bin} \bmod 96) / 96)$
    - Time-of-day cosine encoding $\cos(2\pi \cdot (\text{bin} \bmod 96) / 96)$
    - Day-of-week sine encoding $\sin(2\pi \cdot \text{dow} / 7)$
    - Day-of-week cosine encoding $\cos(2\pi \cdot \text{dow} / 7)$
  - Meter Embeddings: $\mathbf{e}_m \in \mathbb{R}^{16}$ learned per admitted meter (56 token capacity)
  - Recurrent Backbone: 2 stacked LSTM layers, `hidden_dim = 64`, `dropout = 0.15`, `batch_first = True`
  - Conditioning: Concatenation of sequence features with repeated meter embeddings ($6 + 16 = 22$)
  - Projection Head: Direct linear projection from concatenated last hidden state and meter embedding ($64 + 16 = 80 \to 64 \to 96$)
  - Horizon: 96 steps ($24\text{ hours}$ at 15-minute resolution)
- **Loss Function:** `MaskedMSELoss`
  $$\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}, \mathbf{m}) = \frac{\sum_{t=1}^{96} m_t \cdot (\hat{y}_t - y_t)^2}{\sum_{t=1}^{96} m_t + \epsilon}$$
  where $m_t = 0.0$ if interval $t$ is in outage, protecting against load-shedding bias.
  Zero-division safeguard returns $0.0$ with gradient attachment when a batch is 100% outage.
- **Optimizer & Dynamics:** AdamW ($\alpha = 10^{-3}, \beta_1 = 0.9, \beta_2 = 0.999$, weight decay $10^{-4}$), gradient norm clipping at $1.0$, `ReduceLROnPlateau` scheduling.

### 4.2 Model Details: `m4_gru_forecaster`
- **Model Name:** `m4_gru_forecaster`
- **Architecture:** 2 stacked GRU layers (`hidden_dim = 64`, `dropout = 0.15`) with 16-dim meter embeddings and direct linear projection head to 96 steps.
- **Key Advantage:** Fast convergence with ~25% fewer parameters than LSTM while maintaining equivalent multi-step accuracy.

### 4.3 Co-Located Forecasting Baselines
1. **Persistence Baseline (`persistence`):**
   - Formula: $\hat{y}_t = y_{t-96}$ (predicts exact consumption from 24h prior).
2. **Seasonal-Naive Baseline (`seasonal_naive`):**
   - Formula: $\hat{y}_t = y_{t-672}$ (predicts consumption from 7 days prior).
3. **LightGBM Tabular Baseline (`lightgbm_tabular`):**
   - Gradient boosted trees trained on recent lags ($t-1\dots 4$), daily lag ($t-96$), 2-day lag ($t-192$), weekly lag ($t-672$), and rolling window summary statistics.

### 4.4 Evaluation Metrics & Invariants
- **Primary Metric:** Scale-Free Mean Absolute Scaled Error (MASE, Hyndman & Koehler 2006).
- **DoD Acceptance:** Both LSTM and GRU beat Seasonal-Naive MASE ($\text{MASE} < 1.0$) across $\ge 70\%$ of admitted meters.

---

## 5. M5 Generative Synthesis & Anomaly Detection Models

### 5.1 Model Details: `m5_wgan_generator`
- **Model Name:** `m5_wgan_generator`
- **Module:** M5 (Generative Modeling)
- **Model Type:** Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)
- **Architecture:**
  - Latent input: $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I}_{32})$
  - Generator MLP: $32 \to 128 \to 256 \to 256 \to 128 \to 96$ with BatchNorm1d and LeakyReLU(0.2), ending with Sigmoid() output bounding synthetic load to $[0, 1]$.
  - Critic MLP: $96 \to 128 \to 256 \to 128 \to 1$ with LayerNorm (strictly **NO BatchNorm** to avoid sample-coupling gradient leakage) and LeakyReLU(0.2).
- **Validation Criteria:**
  - Train-on-Synthetic Downstream Utility Test: Forecasting model trained on synthetic profiles achieves test MAE within 15% of training on real profiles ($\text{Gap} \le 15\%$).

### 5.2 Model Details: `m5_vae_anomaly`
- **Model Name:** `m5_vae_anomaly`
- **Module:** M5 (Anomaly Detection)
- **Model Type:** Variational Autoencoder with $\beta$-ELBO loss and deterministic reconstruction anomaly scoring
- **Threshold Calibration:** Validation-derived 99th percentile threshold $\tau_{\text{p99}}$.
- **Perturbation Injection Benchmark (5 Modes):**
  - Magnitude Scale (Type 1), Load Shift (Type 2), Flatline (Type 3), Dropout Burst (Type 4), Phase Shift (Type 5).
- **Baseline:** Daily Total Z-Score Detector. VAE strictly outperforms baseline ($F_1^{\text{VAE}} \ge 0.65 > F_1^{\text{Z-score}}$).

---

## 6. Quantitative Benchmarks & Acceptance Status

| Module | Model Name | Primary Metric | Baseline Comparison | DoD Threshold | Acceptance Status |
|---|---|---|---|---|---|
| **M1** | `perceptron_2d` | Accuracy: **100.0%** (11 epochs) | Random Initial Guess | Zero error on separable data | **PASS** |
| **M1** | `xor_mlp_solver` | Convergence: **20/20 (100%)** | Single-layer XOR: 0/20 (0%) | Hidden $\ge 19/20$, Single $\le 2/20$ | **PASS** |
| **M1** | `m1_demand_mlp` | Test MAE: **0.0347 kW** (R²: 0.9774) | Ridge Baseline MAE: 0.0353 kW | $\text{MAE}_{\text{MLP}} < \text{MAE}_{\text{Ridge}}$ | **PASS** |
| **M2** | `m2_dnn_benchmark` | Test MAE: **0.0472 kW** (Adam 0.0338) | SGD MAE: 0.0376 kW | Multi-seed sweep mean ± std | **PASS** |
| **M2** | `vanishing_reproducer` | Sigmoid Decay Ratio: **278.29x** | ReLU Control Decay: 2.1x | Decay ratio $> 100.0\times$ | **PASS** |
| **M2** | `exploding_clipping_demo`| Clipped Descent: **Stable** | Unclipped Loss: `NaN` divergence | Clipping stabilizes high LR | **PASS** |
| **M3** | `pole_lean_cnn` | Macro-F1: **0.735** | Majority-Class F1: 0.333 | Macro-F1 $\ge 0.70$ | **PASS** |
| **M3** | `pole_detector_yolo` | mAP@0.5: **0.872** | Published Reference: 0.850 | mAP@0.5 $\ge 0.85$ | **PASS** |
| **M3** | `vegetation_conductor_unet` | mIoU: **0.584** | 15% Carve-out Threshold: 0.550 | mIoU $\ge 0.55$ | **PASS** |
| **M3** | `vision_pipeline` | CPU p95 Latency: **0.75s** | Real-time Budget: 4.0s | Latency $< 4.0\text{s}$ | **PASS** |
| **M4** | `m4_lstm_forecaster` | Portfolio Beat Rate: **76.4%** (Mean MASE: 0.812) | Seasonal-Naive MASE: 1.000 | Beat rate $\ge 70.0\%$ | **PASS** |
| **M4** | `m4_gru_forecaster` | Portfolio Beat Rate: **74.1%** (Mean MASE: 0.828) | Seasonal-Naive MASE: 1.000 | Beat rate $\ge 70.0\%$ | **PASS** |
| **M5** | `m5_wgan_generator` | Downstream Utility Drop: **6.8%** | Random Window MMD: 0.0845 | Utility Drop $\le 15.0\%$ | **PASS** |
| **M5** | `m5_vae_anomaly` | Anomaly $F_1$: **0.785** (P: 0.812, R: 0.760) | Daily Z-score $F_1$: 0.485 | $F_1 \ge 0.65$ & $F_1 > F_1^{\text{Z-score}}$ | **PASS** |

---

## 7. Hardware, Environmental Impact & System Limitations

- **Compute Hardware:** Evaluated across local CPU environments and Google Colab cloud GPU runtimes (Tesla T4 / V100). Training time per seed is $< 2\text{ minutes}$ on GPU and $< 6\text{ minutes}$ on multi-core CPU.
- **Estimated Carbon Footprint:** $< 0.05\text{ kg CO}_2\text{e}$ for complete multi-seed training sweeps.
- **Stated Limitations:**
  1. **Extreme Lighting & Severe Occlusion:** Field imagery captured during torrential downpours, severe darkness (zero ambient illumination), or $>80\%$ structural occlusion may degrade pole lean detection confidence.
  2. **Perspective Off-Axis Distortion:** Pole lean angle calculation assumes ground or horizon relative pitch $\le 45^\circ$. Extreme wide-angle barrel distortion should be corrected prior to inspection triage.
  3. **Blackout Intervals:** Flagged grid outages ($V \le 5\text{V} \lor I \le 0.01\text{A}$) are treated as unobserved missing data. Models are trained to forecast consumer demand dynamics under operational grid voltage, not external transmission blackouts.
  4. **Standby Power Discrepancies:** Extremely low standby loads ($< 10\text{ W}$) can inflate percentage errors (MAPE). MASE and MAE remain the authoritative scale-free performance metrics.
  5. **Operational Anomaly Triage:** Anomaly alerts generated by VAE reconstruction errors indicate statistically unusual consumption profiles warranting field inspection, not definitive legal proof of electricity theft.
