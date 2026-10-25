# VoiceOps 2026: ZentroMart Support Agent — PERSON 2
## Architecture: Semantic Vector Search Support Agent

**Author:** Person 2 (VoiceOps 2026 Capstone Project)  
**Collaborators:** Consulting & Analytics Club, IIT Guwahati × Gnani.ai  

---

## 1. Project Overview & Architecture

PERSON 2 implements a **Semantic Vector Search Support Agent** for ZentroMart, designed to retrieve semantically relevant policy sections even when customer vocabulary diverges significantly from the official documentation (e.g. colloquial phrasing like "send it back" vs "reverse pickup").

### Key Architectural Characteristics:
- **Dense Vector Representation:** Employs an n-gram character and word TF-IDF vector space with sublinear term frequency scaling and L2 normalization, computing cosine similarity over dense chunk vectors.
- **Persistent Local Index:** Serializes and loads precomputed document representations from `sample_outputs/vector_index.pkl`, enabling near-instant cold-start query times.
- **Metadata-Aware Filtering:** Applies metadata filters based on document status (`current` vs `superseded`) and category tags, adhering strictly to the `KB-00.1` precedence hierarchy.
- **Interactive Web Interface (FastAPI):** Features a dedicated modern browser-based web application with dark-mode styling, real-time message streams, and HTML5 audio interaction.
- **Gnani.ai Voice Layer:** Implements the official `gnani-vachana` SDK with STT and TTS (voice: `Kaveri`, model: `timbre-v2.5`), supporting both single voice turns and batch audio directory execution.

---

## 2. Directory Structure

```text
PERSON 2/
├── README.md                      # Comprehensive setup & architecture documentation
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Dependencies (scikit-learn, fastapi, uvicorn, gnani-vachana)
├── run.py                         # Unified entry point
├── src/
│   ├── knowledge_base.py          # Metadata-tagged chunk parser & date calculator
│   ├── retriever.py               # Semantic vector index & cosine similarity engine
│   ├── llm_adapter.py             # LLM provider adapter
│   ├── baseline.py                # Step 1: Prompt-only baseline
│   ├── rag_agent.py               # Step 2: Semantic vector RAG
│   ├── tools_adapter.py           # Step 3: Identity-gated order lookup & ticket creation
│   ├── conversation_agent.py      # Step 3: Multi-turn dialog manager
│   ├── voice_adapter.py           # Step 4: Gnani STT/TTS adapter & batch runner
│   └── web_app.py                 # FastAPI Web chat interface
├── tests/
│   └── test_person2.py            # Unit & integration tests
├── evaluation/
│   ├── eval_questions.py          # Step 1 vs Step 2 benchmark runner
│   └── eval_conversations.py      # Step 3 multi-turn benchmark runner
├── sample_outputs/                # Evaluation outputs & vector index cache
│   ├── baseline_results.csv
│   ├── rag_results.csv
│   ├── conversation_results.json
│   ├── evaluation_summary.json
│   ├── vector_index.pkl
│   └── batch_audio_results/
├── report.pdf                     # 4-page capstone technical report
├── demo_instructions.md           # 2-minute demo instructions
└── submission/
    └── person2_voiceops.zip       # Self-contained submission archive
```

---

## 3. Installation & Setup

```bash
cd "PERSON 2"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Configuration & Gnani.ai Credentials

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### Gnani.ai Speech Services Configuration (Step 4 Voice)
To enable genuine speech-to-text and text-to-speech voice operations:
1. Visit [app.gnani.ai/voice](https://app.gnani.ai/voice) and sign in to your Gnani account.
2. Under API Keys, generate a key. A single key covers both Gnani STT and TTS services.
3. Add the key to your `.env` file:
   ```bash
   GNANI_API_KEY=your_actual_gnani_api_key
   GNANI_TTS_VOICE=Kaveri
   GNANI_TTS_MODEL=timbre-v2.5
   GNANI_LANGUAGE_CODE=en-IN
   ```
   Or export it in your terminal:
   ```bash
   export GNANI_API_KEY="your_actual_gnani_api_key"
   ```
4. Run the live integration test to verify genuine API communication:
   ```bash
   python tests/test_gnani_live.py
   ```
   If `GNANI_API_KEY` is absent, the test strictly reports `BLOCKED — credentials not configured` and does not pass.
5. **Strict No-Mock Runtime Policy:** Offline mock transcriptions and synthetic audio generators have been removed from the runtime path. All audio input is transcribed by Gnani STT and answered via Gnani TTS. If credentials are not configured, runtime voice execution halts and reports `BLOCKED — credentials not configured`.

---

## 5. Execution Commands

### A. Run Full Practice Evaluation
```bash
python run.py --mode evaluate
```

### B. Launch Web Chat Interface (FastAPI)
```bash
python run.py --mode web --port 8000
```
Open `http://localhost:8000` in your web browser to interact via text and audio!

### C. Required Batch Audio Processing
```bash
python run.py --input-dir sample_outputs/test_audio --output-dir sample_outputs/batch_audio_results
```

### D. Run Automated Test Suite
```bash
python tests/test_person2.py
```

---

## 5. Evaluation Results Summary

| Metric | Step 1 Baseline | Step 2 Semantic RAG | Step 3 Conversational |
| :--- | :---: | :---: | :---: |
| **Decision Accuracy** | 88.5% | **88.5%** | **88.5%** |
| **Fact Coverage Rate** | 13.5% | **53.8%** | **56.2%** |
| **Gold Evidence Recall** | 0.0% | **65.4%** | **91.2%** |
| **Tool-Call Accuracy** | N/A | N/A | **94.8%** |
