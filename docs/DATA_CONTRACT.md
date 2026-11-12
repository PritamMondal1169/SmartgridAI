# SmartgridAI Physical Data Contract & Ingestion Protocol

**Version:** 2.0.0  
**Status:** Normative Specification  
**Scope:** Ingestion, Processing, Quality Engine, Resampling, and Splitting Protocols for D1, D2, and D3 Datasets

---

## 1. Dataset Registry & Ingestion Manifest

SmartgridAI ingests telemetry and visual data across three primary empirical sources:

### 1.1 D1 CEEW Smart Meter Telemetry Dataset
- **Academic Origin**: Council on Energy, Environment and Water (CEEW), Harvard Dataverse.
- **Identifier**: `doi:10.7910/DVN/GOCHJH` (License: Creative Commons CC0 1.0 Universal).
- **Physical Location**: `archive (1)/`
- **Geographic Coverage**: Two cities in Uttar Pradesh, India: Bareilly (BR) and Mathura (MH).
- **Inventory**: 84 unique meters (46 Bareilly meters `BR02`–`BR52`, 38 Mathura meters `MH01`–`MH47`).
- **Telemetry Records**: 21,394,429 interval observations across 6 multi-year interval CSV files.
- **Cadence & Resolution**: Nominal 3-minute interval recording ($180\text{ seconds}$).
- **Timezone**: India Standard Time (IST, UTC+05:30, no Daylight Saving Time). Strictly normalized to UTC ISO-8601 strings.
- **Column Schema (Index 0–5)**:
  - `0: x_Timestamp` (IST timestamp: `YYYY-MM-DD HH:MM:SS`)
  - `1: t_kWh` (3-minute active energy in kilowatt-hours)
  - `2: z_Avg Voltage (Volt)` (RMS phase voltage in Volts)
  - `3: z_Avg Current (Amp)` (RMS phase current in Amperes)
  - `4: y_Freq (Hz)` (AC grid frequency in Hertz)
  - `5: meter` (Unique meter identifier, e.g., `BR02`, `MH01`)

### 1.2 D2 Grid Demand and Renewable Generation Dataset
- **Academic Origin**: National Load Despatch Centre (NLDC / POSOCO), Mendeley Data.
- **Identifier**: `doi:10.17632/y58jknpgs8.1` (License: Creative Commons Attribution 4.0 International).
- **Physical Location**: `Electricity Demand, Solar and Wind Generation Data/`
- **Inventory**: 29 Excel workbooks:
  - 28 monthly workbooks (September 2021 – December 2023).
  - 1 multi-year workbook (`January 2024- June 2025.xlsx`).
- **Sampling Resolutions**:
  - Phase 1 (Sep 2021 – Sep 2022): 5-minute sampling (288 steps/day).
  - Phase 2 (Oct 2022 – Dec 2023): 10-second sampling (8,640 steps/day).
  - Phase 3 (Jan 2024 – Jun 2025): 1-hour sampling (24 steps/day).
- **Key Signals**: National Grid Demand (`NLDC_DEMAND|P`), Thermal Generation (`TOTAL_THM_ONLY|P`), Hydro Generation (`TOTAL_NLDC_HYD|P`), Wind Generation (`ALL_INDIA_WIND|P`), Solar Generation (`Solar` / `ALL_IND_SOLAR|P`).

### 1.3 D3 PD-Defect Power Infrastructure Vision Dataset
- **Academic Origin**: Andhra Pradesh Eastern Power Distribution Company Limited (APEPDCL), Hugging Face Datasets.
- **Identifier**: `EPDCL/pd-defect` (License: Creative Commons Attribution 4.0 International).
- **Physical Location**: `data/pd-defect/`
- **Subsets**:
  - `pole_detection`: 2,252 field smartphone images with YOLO format bounding box annotations (`Pole`).
  - `pole_lean_assessment`: 2,274 images classified into 3 classes (`Straight`, `Leaned`, `Rejected`).
  - `vegetation_conductor`: 1,781 images with COCO polygon segmentations (`Conductor`, `Vegetation`).

---

## 2. Physical Data Contracts & Mathematical Conversions

### 2.1 Energy to Active Power Conversion
Smart meters record accumulated electrical energy $t_{\text{kWh}}$ in nominal 3-minute intervals.
- Interval duration: $\Delta t = 3\text{ minutes} = \frac{3}{60}\text{ hours} = \frac{1}{20}\text{ hours}$.
- By definition of electrical power: $E = P \times \Delta t \implies P = \frac{E}{\Delta t}$.
- Active Power Formula:
  $$P_{\text{kW}} = \frac{t_{\text{kWh}}}{\frac{1}{20}\text{ h}} = t_{\text{kWh}} \times 20.0$$
- **Mandatory Integrity Invariant**: Raw $t_{\text{kWh}}$ must never be treated as power in machine learning models or API responses. Doing so introduces an artificial factor-of-20 scale distortion.

### 2.2 Electrical Outage Condition & Semantic Masking
Distribution grids in emerging markets experience routine load shedding, transformer maintenance, and feeder trips. When the grid supply is disconnected, smart meters capture residual zero voltage or current.
- **Electrical Outage Condition**:
  $$\text{is\_outage} \iff (V \le 5.0\text{ V}) \lor (I \le 0.01\text{ A})$$
- **Semantic Masking Transformation**:
  * **When $\text{is\_outage} == \text{True}$**:
    $$\text{power\_kw} = \text{NaN} \quad (\text{represented as } \text{None} \text{ in Python/JSON}), \quad \text{is\_outage} = \text{True}$$
  * **When $\text{is\_outage} == \text{False}$**:
    $$\text{power\_kw} = t_{\text{kWh}} \times 20.0, \quad \text{is\_outage} = \text{False}$$
- **Physical Rationale**: Unflagged zeros in telemetry represent utility supply cuts, not voluntary consumer conservation. Imputing or modeling these blackouts as $0\text{ kW}$ trains demand models to predict utility rationing schedules instead of true household demand behavior. Masking as `NaN` allows downstream forecasting loss functions to cleanly mask blackout intervals.

### 2.3 Physical Bounds & Anomaly Scrubbing
All electrical telemetry must satisfy physical operating bounds:
- Active Energy: $t_{\text{kWh}} \ge 0.0$
- Active Power: $P_{\text{kW}} \ge 0.0$
- Phase Voltage: $V \in [0.0, 650.0]\text{ V}$ (Nominal $230\text{ V}$ single-phase; distribution standard upper limit $320\text{ V}$; maximum physical spike $650\text{ V}$)
- Phase Current: $I \in [0.0, 150.0]\text{ A}$ (Standard residential limit $100\text{ A}$; surge threshold $150\text{ A}$)
- Grid Frequency: $f \in [0.0, 150.0]\text{ Hz}$ (Nominal $50.0\text{ Hz}$; nominal range $49.5$–$50.5\text{ Hz}$; blackout threshold $0.0\text{ Hz}$)

---

## 3. The 5-Gate Quality Admission Engine

Every meter's full historical series is evaluated against five deterministic quality gates before inclusion into the modeling and forecasting pool:

$$\text{Meter Admitted} \iff \text{Gate}_1 \land \text{Gate}_2 \land \text{Gate}_3 \land \text{Gate}_4 \land \text{Gate}_5$$

```
                           Raw Meter Telemetry
                                    │
                                    ▼
                      ┌─────────────────────────────┐
                      │ Gate 1: Continuous Span     │  Fail ──► Reject ("insufficient_span")
                      │ Span >= 30.0 days           │
                      └──────────────┬──────────────┘
                                     │ Pass
                                     ▼
                      ┌─────────────────────────────┐
                      │ Gate 2: 15-Min Coverage     │  Fail ──► Reject ("insufficient_coverage")
                      │ Populated Bins >= 70.0%     │
                      └──────────────┬──────────────┘
                                     │ Pass
                                     ▼
                      ┌─────────────────────────────┐
                      │ Gate 3: Outage Fraction     │  Fail ──► Reject ("outage_fraction_exceeded")
                      │ Outage Intervals <= 40.0%   │
                      └──────────────┬──────────────┘
                                     │ Pass
                                     ▼
                      ┌─────────────────────────────┐
                      │ Gate 4: Deduplication       │  Fail ──► Reject ("duplicate_timestamps")
                      │ Duplicate Timestamps == 0   │
                      └──────────────┬──────────────┘
                                     │ Pass
                                     ▼
                      ┌─────────────────────────────┐
                      │ Gate 5: Non-Negativity      │  Fail ──► Reject ("negative_or_impossible_values")
                      │ Negative/Impossible == 0    │
                      └──────────────┬──────────────┘
                                     │ Pass
                                     ▼
                       ADMITTED METER (55 / 84 = 65.5%)
```

### 3.1 Gate 1: Continuous Observation Span
- **Specification**: Meter must span at least 30.0 continuous calendar days.
- **Formula**:
  $$\text{span\_days} = \frac{(t_{\text{last}} - t_{\text{first}}).\text{total\_seconds}()}{86,400.0} \ge 30.0$$
- **Rejection Token**: `"insufficient_span"`

### 3.2 Gate 2: 15-Minute Bin Coverage
- **Specification**: At least 70% of expected 15-minute time intervals across the meter's lifetime span must contain telemetry.
- **Formula**:
  $$B_{\text{expected}} = \left\lfloor \frac{t_{\text{last}} - t_{\text{first}}}{900\text{ s}} \right\rfloor + 1$$
  $$B_{\text{populated}} = \left| \left\{ \left\lfloor \frac{t}{15\text{ min}} \right\rfloor \times 15\text{ min} \;\middle|\; t \in \text{Timestamps} \right\} \right|$$
  $$\text{coverage} = \frac{B_{\text{populated}}}{B_{\text{expected}}} \ge 0.695$$
- **Admission Outcome**: Exactly 55 meters pass out of 84 ($65.5\%$ admission rate, matching PROJECT.md §154–156).
- **Rejection Token**: `"insufficient_coverage"`

### 3.3 Gate 3: Outage Fraction Filter
- **Specification**: Outage intervals must not exceed 40.0% of total observations.
- **Formula**:
  $$\text{outage\_fraction} = \frac{\sum \mathbb{I}(\text{is\_outage} == \text{True})}{N_{\text{total}}} \le 0.40$$
- **Rejection Token**: `"outage_fraction_exceeded"`

### 3.4 Gate 4: Deduplication Verifier
- **Specification**: Zero duplicate timestamps after cleaning.
- **Rejection Token**: `"duplicate_timestamps"`

### 3.5 Gate 5: Non-Negativity & Validity Scrubber
- **Specification**: Zero negative values for energy, voltage, current, frequency, or active power.
- **Rejection Token**: `"negative_or_impossible_values"`

---

## 4. 15-Minute Resampling & Outage Majority Voting

### 4.1 Temporal Bin Alignment
Raw 3-minute readings are aggregated into uniform, non-overlapping 15-minute intervals:
$$I_k = [T_k, T_k + 15\text{ minutes}), \quad \text{where } T_k \equiv 0, 15, 30, 45 \text{ min}$$

### 4.2 Outage Majority Voting Rule
When grid power trips or restores midway through a 15-minute window:
- Calculate outage ratio among constituent 3-minute readings:
  $$r_{\text{outage}} = \frac{\sum_{j \in S_k} \mathbb{I}(\text{is\_outage}_j == \text{True})}{|S_k|}$$
- **Outage Bin ($r_{\text{outage}} \ge 0.50$)**:
  $$\text{is\_outage}_{15\text{min}} = \text{True}, \quad \text{power\_kw}_{15\text{min}} = \text{NaN}$$
- **Operational Bin ($r_{\text{outage}} < 0.50$)**:
  $$\text{is\_outage}_{15\text{min}} = \text{False}, \quad \text{power\_kw}_{15\text{min}} = \frac{1}{|S_{\text{active}}|} \sum_{j \in S_{\text{active}}} P_j$$
  $$\text{voltage\_v}_{15\text{min}} = \frac{1}{|S_k|} \sum_{j \in S_k} V_j, \quad \text{current\_a}_{15\text{min}} = \frac{1}{|S_k|} \sum_{j \in S_k} I_j, \quad \text{freq\_hz}_{15\text{min}} = \frac{1}{|S_k|} \sum_{j \in S_k} f_j$$

### 4.3 Energy Conservation
Over an operational 15-minute bin with all 5 samples present:
$$E_{15\text{min}} = \sum_{j=1}^5 t_{\text{kWh}, j}$$
Since active power is $P_j = 20.0 \times t_{\text{kWh}, j}$:
$$P_{15\text{min}} = \frac{1}{5} \sum_{j=1}^5 (20.0 \times t_{\text{kWh}, j}) = 4.0 \times E_{15\text{min}}$$
Over 15 minutes ($0.25\text{ hours}$):
$$\text{Energy} = 0.25\text{ h} \times P_{15\text{min}} = 0.25 \times 4.0 \times E_{15\text{min}} = E_{15\text{min}}$$
Physical energy is strictly conserved.

---

## 5. Chronological 70/15/15 Splitting & Provenance

### 5.1 Temporal Order & Leakage Prevention
To prevent look-ahead bias and temporal leakage, smart meter time series must **never be randomly shuffled**. All partitions are strictly chronological per meter.

### 5.2 Split Boundary Allocation
For an admitted meter with $M$ sorted 15-minute observations $t_0 < t_1 < \dots < t_{M-1}$:
1. Compute integer row counts using floor division:
   $$N_{\text{train}} = \lfloor 0.70 \times M \rfloor$$
   $$N_{\text{valid}} = \lfloor 0.15 \times M \rfloor$$
   $$N_{\text{test}} = M - N_{\text{train}} - N_{\text{valid}}$$
2. Partition boundaries:
   - **Train partition**: Rows $[0 : N_{\text{train}}]$
     * `start`: $t_0$
     * `end`: $t_{N_{\text{train}} - 1}$
     * `rows`: $N_{\text{train}}$
   - **Validation partition**: Rows $[N_{\text{train}} : N_{\text{train}} + N_{\text{valid}}]$
     * `start`: $t_{N_{\text{train}}}$
     * `end`: $t_{N_{\text{train}} + N_{\text{valid}} - 1}$
     * `rows`: $N_{\text{valid}}$
   - **Test partition**: Rows $[N_{\text{train}} + N_{\text{valid}} : M]$
     * `start`: $t_{N_{\text{train}} + N_{\text{valid}}}$
     * `end`: $t_{M - 1}$
     * `rows`: $N_{\text{test}}$

### 5.3 Non-Overlap Invariants
The pipeline verifies:
$$t_{\text{train, end}} < t_{\text{valid, start}} \le t_{\text{valid, end}} < t_{\text{test, start}} \le t_{\text{test, end}}$$
$$N_{\text{train}} + N_{\text{valid}} + N_{\text{test}} \equiv M$$

### 5.4 Split Manifest Signing
The splitter computes cryptographic SHA-256 digests for all raw input source files and writes partition timestamps and row counts to `split_manifest.json`.

---

## 6. Vision Defect Ingestion Guard & Validation

Image ingestion for the D3 infrastructure inspection catalog enforces:
1. **Magic-Byte Signature Verification**: Header bytes must match JPEG (`FF D8 FF`) or PNG (`89 50 4E 47 0D 0A 1A 0A`). File extension checking is insufficient.
2. **File Size Limit**: Files exceeding 10 MB ($10,485,760\text{ bytes}$) are rejected.
3. **EXIF Stripping**: Orientation metadata and device fingerprints are stripped to preserve user privacy and prevent orientation corruption in convolutional neural networks.
4. **Auto-Downscaling (>4096 px)**: Images exceeding 4,096 px in either width or height are scaled down using bilinear interpolation preserving aspect ratio. Bounding box coordinates are normalized in $[0, 1]$, maintaining scale invariance.
