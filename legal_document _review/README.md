# Legal Document Review OpenEnv

A real-world OpenEnv benchmark environment where AI agents review legal contracts. Agents classify clauses, identify risks, and redline documents — tasks that cost law firms thousands of hours annually.

## Environment Description

This environment simulates a **legal document review** task, which is a genuine, high-value application in the legal industry. Law firms and corporate legal departments spend significant time and money reviewing contracts for:

- **Clause Classification**: Identifying what type of clause a provision represents
- **Risk Spotting**: Finding problematic clauses that could expose the client
- **Contract Redlining**: Proposing specific edits to bring contracts into policy compliance

## Why This Environment?

- **Real-world utility**: Contract review is a $20B+ market annually
- **Meaningful difficulty progression**: From simple classification to complex redlining
- **Clear success criteria**: Graders are deterministic and produce 0.0–1.0 scores
- **Partial credit**: Reward signals reflect partial progress, not just binary success/failure

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- OpenAI-compatible API key (for inference)

### Installation

```bash
# Clone and navigate to the project
cd submission

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Start the server
uvicorn server:app --host 0.0.0.0 --port 7860

# In another terminal, run inference
python inference.py
```

### Docker Deployment

```bash
# Build the image
docker build -t legal-env .

# Run the container
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your-api-key \
  legal-env
```

## Tasks

### Task 1: Clause Classifier (Easy)

**Objective**: Given a contract clause, identify its type from an 8-label taxonomy.

**Taxonomy**:
- indemnification
- limitation_of_liability
- confidentiality
- termination
- intellectual_property
- governing_law
- force_majeure
- payment_terms

**Scoring**:
- Exact match: 1.0
- Near-miss (commonly confused labels): 0.5
- Wrong label: 0.0
- Skip: -0.1 penalty

### Task 2: Risk Spotter (Medium)

**Objective**: Given a contract section, identify all high-risk clauses.

**Scoring**:
- Weighted F1 score (precision × recall)
- Severity weighting: critical (3×), high (2×), medium (1×)
- Hallucination penalty: -0.05 per fabricated risk
- Skip: 0.0 with penalty

### Task 3: Contract Redliner (Hard)

**Objective**: Given a contract and policy brief, propose specific edits to bring the contract into compliance.

**Scoring**:
- Per-edit quality (section match + issue match + redline quality)
- Coverage bonus: +0.05 for finding all required edits
- Hallucination penalty: -0.05 per fabricated edit

## Action & Observation Spaces

### Observation Space

```python
{
    "task_id": str,           # "clause_classifier" | "risk_spotter" | "contract_redliner"
    "document_text": str,     # The contract text to analyze
    "instructions": str,     # Task instructions
    "context": dict,          # Additional metadata (e.g., policy brief)
    "step_count": int,        # Current step (1 to max_steps)
    "max_steps": int,         # Maximum steps per episode (5)
}
```

### Action Space

```python
{
    "action_type": str,       # "classify" | "flag_risks" | "redline" | "skip"
    "content": str,           # Agent's response text
    "metadata": dict,        # Optional structured data
}
```

### Reward Structure

```python
{
    "score": float,           # 0.0 to 1.0
    "breakdown": dict,        # Score computation details
    "done": bool,             # Episode termination
    "feedback": str,          # Human-readable explanation
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode, returns Observation + session_id |
| `/step` | POST | Submit action, returns Observation + Reward |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List all available tasks |
| `/grader` | POST | Direct grader access (development) |
| `/baseline` | GET | Sample baseline responses |
| `/health` | GET | Health check |

## Baseline Scores

Running the inference script with `gpt-4o-mini` produces:

| Task | Expected Score |
|------|----------------|
| clause_classifier | 1.0 (exact match) |
| risk_spotter | ~0.84 (F1) |
| contract_redliner | ~0.77 (mean quality) |

Overall mean: **~0.87**

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `http://localhost:11434/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `HF_TOKEN` | API key | (required) |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |

## Testing

```bash
# Run unit tests
pytest test_openenv.py -v

# Run inference
python inference.py

# Validate OpenEnv spec
openenv validate
```

## Deployment to HuggingFace Spaces

### Prerequisites

1. Create a HuggingFace account
2. Generate an HF token with write access

### Steps

1. **Login to HuggingFace**:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

2. **Create a new Space**:
   - Go to https://huggingface.co/spaces/new
   - Select "Docker" as the SDK
   - Choose "Blank" template
   - Set repository name (e.g., `legal-document-review`)

3. **Push to HuggingFace**:
   ```bash
   # Initialize git if not already
   git init
   git add .
   git commit -m "Initial submission"
   
   # Set remote
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/legal-document-review
   
   # Push
   git push -u origin main
   ```

4. **Configure Space Secrets**:
   - In your Space settings, add secrets:
     - `API_BASE_URL`: Your LLM endpoint
     - `MODEL_NAME`: Model to use
     - `HF_TOKEN`: Your API key

5. **Verify Deployment**:
   - The Space will build automatically
   - Check the "Build logs" tab for progress
   - Once deployed, visit the Space URL
   - Test `/health` endpoint returns 200

### Docker Build for HF Spaces

The included `Dockerfile` is optimized for HF Spaces:

```dockerfile
# Multi-stage build for smaller image
FROM python:3.11-slim AS builder
# ... installs dependencies ...

FROM python:3.11-slim AS runtime
# ... copies app, exposes port 7860 ...

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

HF Spaces expects:
- Port `7860` (default)
- `/health` endpoint for liveness checks

## Project Structure

```
submission/
├── README.md              # This file
├── openenv.yaml          # OpenEnv specification
├── inference.py           # Baseline inference script
├── Dockerfile            # Container definition
├── server.py              # FastAPI server
├── requirements.txt       # Python dependencies
├── env/
│   ├── legal_env.py      # Core environment
│   └── models.py         # Pydantic models
├── graders/
│   ├── clause_classifier.py
│   ├── risk_spotter.py
│   └── contract_redliner.py
├── data/
│   └── contracts.py      # Contract samples + ground truth
└── tests/
    └── test_openenv.py  # Validation tests
```

## License

MIT
