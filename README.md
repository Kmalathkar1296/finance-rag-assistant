---
title: Finance RAG Assistant
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Finance RAG Assistant

> AI-powered payment reconciliation and financial analysis system — query your financial data in plain English.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.1.0-green)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.23-orange)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.2-red?logo=streamlit)](https://streamlit.io/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.1-yellow?logo=gradio)](https://www.gradio.app/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow?logo=huggingface)](https://huggingface.co/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
  - [Streamlit Cloud](#streamlit-cloud)
  - [Gradio / HuggingFace Spaces](#gradio--huggingface-spaces)
  - [Docker](#docker)
- [Usage](#usage)
- [Testing](#testing)
- [Future Scope](#future-scope)
- [Contributing](#contributing)

---

## Overview

**Finance RAG Assistant** combines Retrieval-Augmented Generation (RAG) with rule-based financial analysis to let finance teams query their data conversationally — no SQL or spreadsheet expertise required.

It generates (or ingests) synthetic financial datasets covering accounts receivable, payments, general ledger entries, budget forecasts, and expense claims. A ChromaDB vector store indexes these records so that natural-language questions retrieve the most relevant context before producing structured insights and recommendations.

**Both Streamlit and Gradio UIs have been tested end-to-end**, making the app deployable to Streamlit Cloud, HuggingFace Spaces, or any server with Docker.

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Natural Language Queries** | Ask questions like *"Which customers have overdue invoices?"* |
| 💸 **Discrepancy Detection** | Automatically flags amount mismatches, missing payments, and overdue records |
| 📊 **Budget Variance Analysis** | Identifies departments over/under budget with configurable thresholds |
| 💳 **Expense Claims Management** | Tracks pending, approved, rejected, and over-limit claims |
| 📈 **Interactive Analytics** | Plotly charts for invoice status, top customers, budget vs. actual, and expense categories |
| 📄 **Comprehensive Reports** | One-click text report covering all modules |
| 🎲 **Synthetic Data Generator** | Realistic AR, payment, GL, budget, and claims data for demos and testing |
| 🐳 **Docker Ready** | Containerised for consistent local and cloud deployments |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM / RAG Orchestration** | LangChain 1.1.0 |
| **Vector Database** | ChromaDB 0.5.23 |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace |
| **UI (Option A)** | Streamlit 1.40.2 |
| **UI (Option B)** | Gradio 4.44.1 |
| **Data Processing** | Pandas 2.2.3, NumPy |
| **Visualisation** | Plotly 5.24.1 |
| **File I/O** | openpyxl 3.1.5 |
| **Containerisation** | Docker |

---

## Project Structure

```
finance-rag-assistant/
├── src/
│   ├── __init__.py
│   ├── config.py            # Environment variables and constants
│   ├── data_generator.py    # Synthetic financial data generation
│   ├── rag_system.py        # ChromaDB vector store + RAG query engine
│   ├── utils.py             # Shared helpers (save/load, formatting)
│   └── app.py               # Gradio application entry point
├── streamlit_app.py         # Streamlit application entry point
├── scripts/
│   ├── generate_data.py     # CLI: generate and persist sample data
│   ├── run_demo.py          # CLI: end-to-end demo run
│   └── interactive_query.py # CLI: REPL-style query session
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_rag_demo.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data_generator.py
│   └── test_rag_system.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── outputs/
│   ├── reports/
│   └── exports/
├── chroma_db/               # Persisted vector store (git-ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
└── .env.example
```

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- `pip` / `venv`

### 1 Clone and install

```bash
git clone https://github.com/<your-username>/finance-rag-assistant.git
cd finance-rag-assistant

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2 Configure environment

```bash
cp .env.example .env
# Optional: add ANTHROPIC_API_KEY or OPENAI_API_KEY for LLM-powered responses
```

### 3 Run the app

**Streamlit (recommended for local use)**

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

**Gradio (recommended for HuggingFace Spaces)**

```bash
python src/app.py
# Opens at http://localhost:7860
```

**Command-line demo**

```bash
python scripts/run_demo.py
```

**Interactive query REPL**

```bash
python scripts/interactive_query.py
```

---

## Deployment

### Streamlit Cloud

1. Push your repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set **Main file path** to `streamlit_app.py`.
4. Add secrets (e.g. `ANTHROPIC_API_KEY`) in **Settings → Secrets**.
5. Click **Deploy** — the app is live within minutes.

### Gradio / HuggingFace Spaces

Using the included deployment script:

```bash
chmod +x deploy_gradio.sh
./deploy_gradio.sh
```

Or manually:

```bash
# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/<HF_USERNAME>/<SPACE_NAME>
git push hf main
```

The Space auto-builds from `src/app.py`. Environment secrets can be set in **Settings → Repository secrets** on HuggingFace.

> **Tested deployments:** Both Streamlit Cloud and HuggingFace Spaces (Gradio SDK) have been verified end-to-end, including data generation, vector store initialisation, query answering, and report export.

### Docker

```bash
# Build image
docker build -t finance-rag-assistant .

# Run (Gradio on port 7860)
docker run -p 7860:7860 finance-rag-assistant

# With environment variables
docker run -p 7860:7860 \
  -e ANTHROPIC_API_KEY=your_key \
  finance-rag-assistant
```

---

## Usage

### In-app workflow

1. **Setup tab** — choose the number of invoices and expense claims, then click *Generate Data*. The system builds the vector store in the background.
2. **Query Assistant tab** — type any financial question or use a quick-query button.
3. **Discrepancies tab** — run automated discrepancy detection across all invoice/payment pairs.
4. **Data Overview tab** — browse raw records and key metrics.
5. **Analytics tab** — view interactive Plotly dashboards.
6. **Reports tab** — generate and download a full-text reconciliation report.

### Example queries

```
Show me all payment discrepancies
Which customers have invoices overdue by more than 30 days?
Which departments are over budget this quarter?
Show me all expense claims over the policy limit
What is the total outstanding amount for Acme Corp?
Show me pending expense claims submitted this month
```

### Using your own data

Replace the synthetic data generation step with your own DataFrames. The only requirement is that they match the expected column schemas (see `src/data_generator.py` for reference) before calling `rag_system.load_data(...)`.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_data_generator.py -v
pytest tests/test_rag_system.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

The test suite covers:

- Data generator initialisation and column schemas
- AR, payment, GL, budget, and claims record counts and data types
- RAG system load, vector store build, and query routing
- Discrepancy detection logic
- Report generation and file output
- Edge cases (empty datasets, missing optional modules)

---

## Future Scope

The following enhancements are planned or under consideration:

### 🤖 AI & LLM Upgrades
- **Multi-provider LLM support** — swap between Anthropic Claude, OpenAI GPT-4o, and open-source models (Mistral, LLaMA) via a single config flag
- **Agentic workflows** — LangGraph-based agents that can autonomously fetch additional context, run follow-up queries, and draft reconciliation emails
- **LLM-as-Judge evaluation** — integrate RAGAS or DeepEval to continuously measure retrieval quality and answer faithfulness
- **Fine-tuned finance embeddings** — replace `all-MiniLM-L6-v2` with a domain-specific model trained on accounting and ERP vocabulary

### 📊 Data & Integrations
- **Live ERP connectors** — direct integrations with QuickBooks, SAP, and NetSuite via their REST APIs
- **Database backends** — swap flat-file storage for PostgreSQL or Snowflake for enterprise-scale data volumes
- **Real-time streaming** — ingest payment events from Kafka or AWS Kinesis and update the vector store incrementally
- **Multi-currency support** — FX conversion layer with live exchange rates for cross-border reconciliation
- **OCR pipeline** — extract invoice and receipt data from uploaded PDFs and images before indexing

### 🛡️ Compliance & Security
- **Role-based access control (RBAC)** — restrict query scope by user role (AP clerk, controller, CFO)
- **Audit trail** — immutable log of every query, result, and data modification for SOX/GAAP compliance
- **PII redaction** — automatically mask sensitive fields before embedding or LLM processing
- **SOC 2 / ISO 27001 alignment** — data residency controls and encryption at rest/in-transit

### 📈 Analytics & Reporting
- **Scheduled reports** — cron-driven email or Slack delivery of daily/weekly reconciliation summaries
- **Anomaly detection** — ML-based outlier detection for unusual payment patterns or expense spikes
- **Forecasting module** — time-series models (Prophet, ARIMA) for cash-flow and budget projections
- **Custom dashboard builder** — drag-and-drop chart configuration without code
- **Power BI / Tableau connectors** — publish processed datasets directly to BI tools

### 🚀 Infrastructure & MLOps
- **MLflow integration** — track embedding model versions, query latency, and retrieval metrics
- **Vector store scaling** — migrate from ChromaDB to Pinecone or Weaviate for production workloads
- **CI/CD pipeline** — GitHub Actions workflow for automated testing, Docker builds, and HuggingFace Space deployments
- **Kubernetes deployment** — Helm chart for scalable, load-balanced production deployment
- **Observability** — OpenTelemetry traces and Grafana dashboards for latency, error rates, and usage metrics

### 🌐 UX & Collaboration
- **Multi-tenant workspace** — isolated environments per organisation or business unit
- **Collaborative annotations** — allow finance teams to tag, comment on, and resolve flagged discrepancies within the app
- **Mobile-responsive UI** — optimised Gradio/Streamlit layouts for tablet and mobile use
- **Multilingual support** — query and report in languages beyond English using multilingual embedding models

---

## Contributing

Contributions are welcome! Please open an issue to discuss proposed changes before submitting a pull request.

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
# Make changes
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
# Open a PR against main
```

---
