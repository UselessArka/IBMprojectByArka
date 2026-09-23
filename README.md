# ☀ Solar Power Grid Reliability & Yield Analytics

**IBM SkillsBuild Academic Internship Project**  
Plant 1 · Gradient Boosting ML · Plotly Dash · Docker

---

## 📋 Project Overview

This project delivers a complete, production-ready analytics system for a solar power plant. It ingests raw generation and weather sensor CSV data, trains a machine-learning regression model to predict DC power output from atmospheric conditions, and exposes an interactive executive web dashboard — containerised in Docker for one-command deployment.

---

## 🗂️ Project Structure

```
project/
├── dashboard.py                    # Main application (pipeline + ML + dashboard)
├── requirements.txt                # Pinned Python dependencies
├── Dockerfile                      # Production container (python:3.11-slim)
├── README.md                       # This file
├── Project_Report.docx             # Full project report with screenshots
├── Plant_1_Generation_Data.csv     # Input: inverter generation data
└── Plant_1_Weather_Sensor_Data.csv # Input: weather sensor data
```

---

## 🚀 Quick Start

### Option A — Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
python dashboard.py

# 3. Open in your browser
#    http://127.0.0.1:8050
```

### Option B — Run with Docker

```bash
# 1. Build the image
docker build -t solar-analytics .

# 2. Run the container
docker run -p 8050:8050 solar-analytics

# 3. Open in your browser
#    http://localhost:8050
```

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.2.2 | CSV loading, datetime parsing, merge/aggregation |
| `numpy` | 1.26.4 | Numerical ops, RMSE computation |
| `scikit-learn` | 1.5.1 | GradientBoostingRegressor, StandardScaler, R²/MAE |
| `plotly` | 5.22.0 | Interactive charts (time-series, scatter, bar) |
| `dash` | 2.17.1 | Reactive web application framework |

---

## 📊 Dataset Description

### `Plant_1_Generation_Data.csv`

| Column | Description |
|---|---|
| `DATE_TIME` | Timestamp — format `DD-MM-YYYY HH:MM` (15-min intervals) |
| `PLANT_ID` | Plant identifier (4135001) |
| `SOURCE_KEY` | Inverter unit identifier (22 unique inverters) |
| `DC_POWER` | DC power output of the inverter (Watts) |
| `AC_POWER` | AC power output of the inverter (Watts) |
| `DAILY_YIELD` | Energy yielded since midnight on this day (Wh) |
| `TOTAL_YIELD` | Lifetime cumulative energy yield (Wh) |

### `Plant_1_Weather_Sensor_Data.csv`

| Column | Description |
|---|---|
| `DATE_TIME` | Timestamp — format `YYYY-MM-DD HH:MM:SS` (15-min intervals) |
| `PLANT_ID` | Plant identifier (4135001) |
| `SOURCE_KEY` | Sensor identifier |
| `AMBIENT_TEMPERATURE` | Ambient air temperature (°C) |
| `MODULE_TEMPERATURE` | Solar panel surface temperature (°C) |
| `IRRADIATION` | Solar irradiation intensity (kW/m²) |

> **Note:** The two files use different `DATE_TIME` formats. The pipeline normalises both before merging on the timestamp key.

---

## 🔧 Data Pipeline

```
Generation CSV (22 inverters/slot)
    → parse DD-MM-YYYY HH:MM
    → groupby DATE_TIME: sum DC_POWER, AC_POWER; mean DAILY_YIELD
                ↓
Weather CSV (1 sensor/slot)
    → parse YYYY-MM-DD HH:MM:SS
                ↓
         INNER JOIN on DATE_TIME
                ↓
    Feature Engineering:
      HOUR, DAY_OF_WEEK,
      TEMP_DIFF = MODULE_TEMP − AMBIENT_TEMP,
      IRRADIATION_HOUR = IRRADIATION × HOUR
                ↓
    Night-time rows removed (IRRADIATION = 0)
    Forward-fill for isolated NaN dropouts
```

---

## 🤖 Machine Learning Model

**Algorithm:** `GradientBoostingRegressor` (scikit-learn)  
**Target:** `DC_POWER` — total fleet output per 15-min slot (Watts)

**Hyperparameters:**
```
n_estimators  = 300
learning_rate = 0.05
max_depth     = 5
subsample     = 0.8
random_state  = 42
```

**Actual Results (from live run):**

| Metric | Value |
|---|---|
| R² Score | **0.9918** |
| MAE | **4,734.46 W** |
| RMSE | **7,565.37 W** |

**Feature Importance Ranking (from live model):**

| Rank | Feature | Score |
|---|---|---|
| 1 | IRRADIATION | ~1.00 (dominant) |
| 2 | TEMP_DIFF | moderate |
| 3 | IRRADIATION_HOUR | moderate |
| 4 | AMBIENT_TEMPERATURE | low |
| 5 | MODULE_TEMPERATURE | low |
| 6 | HOUR | very low |
| 7 | DAY_OF_WEEK | negligible |

---

## 📈 Dashboard Sections

### Level 1 — Key Performance Indicators
Seven headline metric cards computed from the dataset:
- Total Daily Yield: **8.8 kWh**
- Peak DC Power: **298.9 kW**
- Avg Daytime DC: **123.4 kW**
- Avg Irradiation: **0.4142 kW/m²**
- Avg Ambient Temp: **25.6 °C**
- Dataset Days: **34 days**
- Active Inverters: **22 units**

### Level 2 — Generation Trend Analysis
- **Dual-axis time-series chart:** Daily total DC power (kW) vs average irradiation over 34 days (May 15 – Jun 17, 2020)
- **Scatter chart:** Per-reading irradiation vs DC power, coloured by module temperature (plasma colour scale)

### Level 3 — ML Driver Analysis
- **Horizontal feature importance bar chart** showing which weather variables most drive DC power output
- **Real-Time Yield Forecast Tool** — four interactive sliders (Irradiation, Ambient Temp, Module Temp, Hour of Day) that feed the live trained model and display predicted fleet DC power with ±10% confidence band

### Inverter Health — Maintenance Flagging
Bar chart showing each inverter's mean deviation from model-predicted output. Inverters below −1.5 SD of the fleet mean are flagged **⚠ REVIEW** in orange.

**Flagged units (from live run):**
- `bvBOhCH3iADSZry` — ⚠ REVIEW (~−630 W mean deviation)
- `1BY6WEcLGh8j5v7` — ⚠ REVIEW (~−430 W mean deviation)

### Strategic Insight Panel
Three executive-level insight cards:
- **⚠ Risk** — Weather-driven output variance: CV > 85% on cloudy transition days
- **◆ Opportunity** — Smart grid BESS storage during 10:00–14:00 peak windows; fleet exceeds 224.2 kW; potential 12–18% revenue uplift
- **✦ Action** — 2 inverter units flagged for maintenance; likely soiling/MPPT drift; 3–7% yield recovery potential

---

## 🐳 Docker Details

```dockerfile
FROM python:3.11-slim
# Requirements copied first → pip layer cached on code-only rebuilds
# PYTHONUNBUFFERED=1 → logs stream live via `docker logs`
# host=0.0.0.0 → Dash reachable outside container (not just 127.0.0.1)
# HEALTHCHECK every 30s → container marked unhealthy if Dash goes down
EXPOSE 8050
```

**Useful Docker commands:**

```bash
# Build
docker build -t solar-analytics .

# Run (detached)
docker run -d -p 8050:8050 --name solar solar-analytics

# View logs
docker logs -f solar

# Stop
docker stop solar

# Remove
docker rm solar
```

---

## 🔍 How the Forecast Tool Works

The sliders send four inputs to the live `GradientBoostingRegressor`:

1. Irradiation (kW/m²)
2. Ambient Temperature (°C)
3. Module Temperature (°C)
4. Hour of Day

The model internally computes `TEMP_DIFF`, `IRRADIATION_HOUR`, and uses a fixed `DAY_OF_WEEK = 2` (Wednesday). Output is fleet-level predicted DC power in kW, with a ±10% confidence band displayed below the value.

---

## 📋 Requirements

```
Python >= 3.9
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
plotly==5.22.0
dash==2.17.1
```

Install: `pip install -r requirements.txt`

---

## 🏷️ License

IBM SkillsBuild Academic Internship — educational use.

---

*Built with Python · scikit-learn · Plotly Dash · Docker*
