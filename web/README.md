# DermaSense AI — How to Run

## Quick Start

You need **two terminal windows**.

---

### Terminal 1 — Backend (FastAPI)

```bash
cd "web/backend"

# Create a virtual environment (first time only)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Start the server
python main.py
```

The backend will start at `http://localhost:8000`.  
On first launch it will load all model artifacts (the GP model is ~229MB, may take 30–60s).

---

### Terminal 2 — Frontend (Vite + React)

```bash
cd "web/frontend"
npm run dev
```

The frontend will start at `http://localhost:5173`.

---

## Pages

| Page | URL | Description |
|---|---|---|
| Home | `/` | Landing page with metrics and 3-phase overview |
| Analyze | `/analyze` | Upload image + run inference |
| Research | `/research` | Interactive charts — ablation, per-class F1, safety |
| Pipeline | `/pipeline` | HAM10000 dataset, architecture timeline, key decisions |

---

## What the Backend Loads

| File | Purpose |
|---|---|
| `Final/outputs/checkpoints/best_model_b3.pth` | EfficientNet-B3 (Phase 2 & 3) |
| `Final/outputs/temperature_scaler.pth` | Temperature T=1.5 |
| `Final/outputs/mahalanobis_ood.pkl` | OOD detector (threshold=596.14) |
| `Final/outputs/conformal_predictor.pkl` | Conformal predictor (q̂=0.984) |
| `Final/outputs/model_a_gp.pkl` | GP + IsoForest (Phase 1, ~229MB) |

---

## Notes

- If the backend is not running, the Analyze page will show a red error with instructions.
- The Research page works **without** the backend (all charts use pre-computed static data).
- The Pipeline page is fully static — no backend needed.
