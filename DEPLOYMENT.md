# SmartgridAI — Full-Stack Deployment Guide

This repository contains the complete production-ready **SmartgridAI** platform, unifying the Python FastAPI backend, machine learning inference models (M1–M5), computer vision inspection triage, and modern React + Vite dashboard.

---

## 🚀 Option 1: Instant Local Deployment (Active Now)

The project is currently running live on your local machine:

👉 **URL:** **`http://127.0.0.1:8080`**  
👉 **API Docs:** **`http://127.0.0.1:8080/docs`**  

### 🔑 Demo Credentials
- **Username:** `grid_operator` *(or click "Quick Launch Demo Session")*
- **Password:** `password123`
- **Role:** `Administrator (Full Access)`

### 🔁 Relaunching Anytime Locally
Simply double-click or run from PowerShell:
```powershell
.\run_all.bat
```
The launcher automatically:
1. Adds Node.js to the session environment.
2. Activates the Python virtual environment (`.venv`).
3. Compiles the frontend bundle (`frontend/dist`) if needed.
4. Auto-detects free ports (`8000` or `8080`).
5. Launches your default browser to the live dashboard.

---

## 🐳 Option 2: Docker & Docker Compose Deployment

The application features a production-grade multi-stage container build:
- **Stage 1 (Node.js 20 Alpine):** Compiles the Vite React SPA bundle with Tailwind CSS.
- **Stage 2 (Python 3.11 Slim):** Deploys the unified FastAPI server with pre-loaded models and telemetry assets.

### Build and run with Docker Compose:
```bash
docker compose up --build
```
Access the containerized platform at **`http://localhost:8000`** (or configured port).

### Run standalone Docker container:
```bash
docker build -t smartgridai .
docker run -p 8000:8000 -e ENVIRONMENT=production smartgridai
```

---

## ☁️ Option 3: Cloud Deployment (Render / Railway / Fly.io)

### Deploying to Render:
1. Push this repository to GitHub or GitLab.
2. In Render, select **New +** → **Blueprint**.
3. Point to your repository — Render will automatically read [`render.yaml`](file:///C:/Users/Pritam/OneDrive/Desktop/SmartgridAI/render.yaml) and deploy the Docker web service with zero manual configuration.

### Deploying to Railway:
1. Install the Railway CLI or connect via GitHub:
   ```bash
   railway up
   ```
2. Railway detects the multi-stage `Dockerfile`, assigns a public domain and `$PORT`, and serves the full application immediately.

---

## 📦 What Is Deployed in This Stack

| Route / Asset | Description | Source Implementation |
|---|---|---|
| **`/` (SPA)** | React Dashboard Single Page App | [`frontend/dist/index.html`](file:///C:/Users/Pritam/OneDrive/Desktop/SmartgridAI/frontend/dist/index.html) |
| **`/overview`** | Grid KPI telemetry, active meters, and harmonic stability | [`frontend/src/pages/Overview.jsx`](file:///C:/Users/Pritam/OneDrive/Desktop/SmartgridAI/frontend/src/pages/Overview.jsx) |
| **`/analytics`** | Feeder meter explorer, quality gates funnel, diurnal duck curve | [`frontend/src/pages/Analytics.jsx`](file:///C:/Users/Pritam/OneDrive/Desktop/SmartgridAI/frontend/src/pages/Analytics.jsx) |
| **`/consumption`**| 15-minute resampled active power load curve | [`frontend/src/pages/ConsumptionChart.jsx`](file:///C:/Users/Pritam/OneDrive/Desktop/SmartgridAI/frontend/src/pages/ConsumptionChart.jsx) |
| **`/models`** | Interactive M1–M5 AI inference with temperature & solar sliders | [`frontend/src/pages/ModelLab.jsx`](file:///C:/Users/Pritam/OneDrive/Desktop/SmartgridAI/frontend/src/pages/ModelLab.jsx) |
| **`/inspection`** | Real drone photo upload & M3 computer vision defect detection | [`frontend/src/pages/InspectionQueue.jsx`](file:///C:/Users/Pritam/OneDrive/Desktop/SmartgridAI/frontend/src/pages/InspectionQueue.jsx) |
| **`/anomalies`** | Isolation Forest & reverse power flow incident triage table | [`frontend/src/pages/AnomalyMonitor.jsx`](file:///C:/Users/Pritam/OneDrive/Desktop/SmartgridAI/frontend/src/pages/AnomalyMonitor.jsx) |
| **`/api/*`** | Modular REST endpoints (Analytics, Vision, Models, Auth, etc.) | [`backend/app/routes/`](file:///C:/Users/Pritam/OneDrive/Desktop/SmartgridAI/backend/app/routes/) |

