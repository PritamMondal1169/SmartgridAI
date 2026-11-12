# SmartgridAI — Dataset Credits & Academic Citations

This project builds upon open empirical datasets generously provided by the academic, power systems, and computer vision research communities. All licenses and attribution notices are documented below.

---

## 1. CEEW Mathura and Bareilly Smart Meter Dataset (D1)

- **Dataset Title**: Smart meter data for Mathura and Bareilly, Uttar Pradesh, India
- **Authors**: Council on Energy, Environment and Water (CEEW)
- **Repository**: Harvard Dataverse
- **Identifier**: `doi:10.7910/DVN/GOCHJH`
- **License**: Creative Commons CC0 1.0 Universal (Public Domain Dedication)
- **Role in SmartgridAI**: Primary empirical telemetry source for household active power conversion ($P_{\text{kW}} = t_{\text{kWh}} \times 20$), grid blackout analysis ($V \le 5\text{V} \lor I \le 0.01\text{A}$), 5-gate quality admission, 15-minute load aggregation, and 24-hour sequence forecasting (LSTM / GRU).
- **Academic Citation**:
  ```bibtex
  @data{CEEW_SmartMeter_2021,
    author = {Council on Energy, Environment and Water (CEEW)},
    publisher = {Harvard Dataverse},
    title = {{Smart meter data for Mathura and Bareilly, Uttar Pradesh, India}},
    year = {2021},
    version = {V1},
    doi = {10.7910/DVN/GOCHJH},
    url = {https://doi.org/10.7910/DVN/GOCHJH}
  }
  ```

---

## 2. National Electricity Demand, Solar and Wind Generation Data (D2)

- **Dataset Title**: Electricity Demand, Solar and Wind Generation Data for India (2021–2025)
- **Authors**: Power System Operation Corporation (POSOCO) / National Load Despatch Centre (NLDC) researchers
- **Repository**: Mendeley Data
- **Identifier**: `doi:10.17632/y58jknpgs8.1`
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Attribution Notice**: "This coursework platform uses national grid demand and renewable generation telemetry from Mendeley Data under CC BY 4.0."
- **Role in SmartgridAI**: Regional grid-level demand forecasting, multi-energy generation analysis (thermal, hydro, solar, wind), and multi-cadence time-series resampling.
- **Academic Citation**:
  ```bibtex
  @data{POSOCO_GridDemand_2024,
    author = {NLDC and POSOCO},
    title = {{Electricity Demand, Solar and Wind Generation Data for India}},
    publisher = {Mendeley Data},
    year = {2024},
    version = {V1},
    doi = {10.17632/y58jknpgs8.1},
    url = {https://doi.org/10.17632/y58jknpgs8.1}
  }
  ```

---

## 3. PD-Defect: Distribution Infrastructure Inspection Dataset (D3)

- **Dataset Title**: PD-Defect: A Field-Captured Smartphone Image Dataset for AI-Based Inspection of Power Distribution Infrastructure
- **Authors**: Andhra Pradesh Eastern Power Distribution Company Limited (APEPDCL) & Research Team
- **Repository**: Hugging Face Datasets (`EPDCL/pd-defect`)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Attribution Notice**: "Field inspection photographs and annotations courtesy of APEPDCL and the PD-Defect research team under CC BY 4.0."
- **Role in SmartgridAI**: Computer vision infrastructure triage:
  - Pole Detection: YOLO format bounding box detection.
  - Pole Lean Assessment: 3-class classification (`Straight`, `Leaned`, `Rejected`).
  - Vegetation-Conductor: COCO format polygon segmentation.
- **Academic Citation**:
  ```bibtex
  @misc{APEPDCL_PDDefect_2023,
    title = {{PD-Defect: A Field-Captured Smartphone Image Dataset for AI-Based Inspection of Power Distribution Infrastructure}},
    author = {APEPDCL Research Group},
    year = {2023},
    howpublished = {\url{https://huggingface.co/datasets/EPDCL/pd-defect}},
    note = {CC BY 4.0}
  }
  ```

---

## 4. Socioeconomic and Household Survey Baselines (D4 & D5)

- **India Residential Energy Survey (IRES 2020)**:
  - Council on Energy, Environment and Water (CEEW)
  - Harvard Dataverse: `doi:10.7910/DVN/U8NYUP`
  - License: CC0 1.0 Universal
  - Role: Socioeconomic appliance ownership and demographic energy profiling.
- **Rural Energy Delivery in India (REDI)**:
  - Harvard Dataverse: `doi:10.7910/DVN/1ZNLUY`
  - License: CC0 1.0 Universal
  - Role: Reference rural electrification survey.

---

## 5. Software & Tooling Attributions

SmartgridAI is built using the following open-source software libraries:
- **FastAPI** (`MIT License`) — High-performance asynchronous web framework
- **Pydantic v2** (`MIT License`) — Data validation using Python type annotations
- **PyTorch** (`Modified BSD License`) — Deep learning and neural network training
- **LightGBM** (`MIT License`) — Gradient boosted decision tree framework
- **scikit-learn** (`BSD 3-Clause License`) — Machine learning utilities and Ridge baselines
- **Pillow** (`HPND License`) — Python Imaging Library for visual preprocessing
- **openpyxl** (`MIT License`) — Excel workbook parsing
- **React 18 & Vite** (`MIT License`) — Frontend component framework and build tool
- **Tailwind CSS** (`MIT License`) — Utility-first CSS framework
- **Recharts** (`MIT License`) — Composable charting library
