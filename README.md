# NFL Cloud Data Platform — Code Repository

IE University · Data Analytics in the Cloud · MBD-EN2025 APRIL  
Group Work Option A: Cloud Data Platform for Sport

---

## Overview

This repository contains the analytics and ML code that supports our RFP response for migrating and modernising the NFL's on-premises data infrastructure to AWS.

The code demonstrates three of the six RFP proposal sections:
- **Section 3 (Low-Level Architecture)** — `src/aws/pipeline.py` maps each data flow to a specific AWS service
- **Section 5 (Examples of Data Processing, Visualisation, and Predictions)** — injury prediction model and player tracking analytics
- **Section 5 (NLP/AI)** — Bedrock chatbot with tool-use definitions

> Delivering a functioning end-to-end AWS deployment is not required by the assignment. This code runs fully locally and is structured so each module could be deployed to its corresponding AWS service with minimal changes.

---

## Proposed AWS Architecture

```
Live NGS Sensors ──► Kinesis Data Streams ──► Kinesis Data Analytics (Flink)
                                                         │
Stadium FTP / Weather API ──► S3 (raw zone)              │
                                   │                     ▼
                              AWS Glue ETL ──► S3 (processed zone) ──► Redshift
                                                         │
                                                    SageMaker
                                                 (injury prediction)
                                                         │
                                               API Gateway + Lambda
                                                         │
                                           Amazon Bedrock (Claude)
                                              NLP Chatbot / Agents
```

---

## Repository Structure

```
nfl-analytics/
├── src/
│   ├── ingestion/
│   │   └── data_loader.py          # Load CSVs locally; maps to S3 in production
│   ├── processing/
│   │   └── feature_engineering.py  # Feature joins & encoding; maps to Glue job
│   ├── ml/
│   │   └── injury_prediction.py    # Random Forest multi-output classifier; maps to SageMaker
│   ├── analytics/
│   │   └── player_tracking.py      # NGS heat maps, speed profiles, movement paths
│   ├── aws/
│   │   └── pipeline.py             # Full pipeline orchestration with AWS service stubs
│   └── chatbot/
│       └── bedrock_chatbot.py      # NLP chatbot; maps to Amazon Bedrock + API Gateway
├── models/                         # Auto-created: model metadata JSON after training
├── outputs/                        # Auto-created: visualisation PNG files
├── InjuryRecord.csv                # Appendix 3C — 105 injury records
├── PlayList.csv                    # Appendix 3B — 267k player-play records
├── PlayerTrackData_43540.csv       # Appendix 3A — 45k NGS tracking frames
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended.

---

## Running the Modules

All modules can be run as scripts from the repo root.

### Data ingestion
```bash
python -m src.ingestion.data_loader
# Output: row/column counts for all three datasets
```

### Feature engineering
```bash
python -m src.processing.feature_engineering
# Output: merged feature set shape and feature column list
```

### Injury prediction model
```bash
python -m src.ml.injury_prediction
# Output: 5-fold CV AUC-ROC scores for each severity threshold,
#         feature importances, model metadata saved to models/
```

### Player tracking analytics
```bash
python -m src.analytics.player_tracking
# Output: movement summary table + 3 PNG visualisations saved to outputs/
#   - heatmap_positions.png
#   - speed_profile_<play>.png
#   - path_<play>.png
```

### AWS pipeline (local simulation)
```bash
python -m src.aws.pipeline
# Output: simulated S3 uploads, Glue job run, SageMaker batch job,
#         and a sample real-time inference result
```

### Bedrock chatbot (mock mode)
```bash
python -m src.chatbot.bedrock_chatbot
# Output: demo Q&A session using pre-configured mock responses
# To connect to real AWS: set use_mock=False and configure AWS credentials
```

---

## Datasets (Appendix 3 of the RFP)

| File | Records | Description |
|---|---|---|
| `InjuryRecord.csv` | 105 | Injuries by player/game/play; severity encoded as days-missed flags |
| `PlayList.csv` | 267,004 | Player-play records with weather, stadium, field type, play type |
| `PlayerTrackData_43540.csv` | 45,661 | NGS tracking frames for player 43540 (position, speed, orientation) |

The datasets are from the [NFL 1st and Future — Analytics Competition on Kaggle](https://www.kaggle.com/c/nfl-playing-surface-analytics).

---

## Injury Prediction Model

Multi-output Random Forest predicting four severity thresholds simultaneously:

| Target | Meaning |
|---|---|
| `DM_M1` | 1+ days missed |
| `DM_M7` | 7+ days missed |
| `DM_M28` | 28+ days missed |
| `DM_M42` | 42+ days missed |

Key features: surface type, field type, stadium type, temperature, play type, position group, player workload (days into season, snaps in current game).

On AWS: trained as a SageMaker training job with input from S3, deployed as a real-time inference endpoint consumed via API Gateway.

---

## Team

| Member | Role |
|---|---|
| Nicolas Beard (Nico) | Code & ML |
| Sebastian Otegui | Code support |
| Juan (Juanca) | Cloud Architecture |
| Álvaro Calonge | Cloud Architecture |
| Tom Biefel | Report / RFP document |
| Sofía Mollón Laorca | Report support |

---

## Generative AI Acknowledgement

Generative AI tools (Claude) were used to assist with code structure, documentation, and architecture suggestions during development, in accordance with course guidelines.
