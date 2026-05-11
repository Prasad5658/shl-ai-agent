# SHL AI Assessment Recommender

FastAPI service for conversational SHL assessment recommendations.

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python catalog/download_catalog.py
python embeddings/build_index.py
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Environment

Copy `.env.example` to `.env` and set:

```env
GEMINI_API_KEY=your_key_here
```

The API still returns deterministic grounded replies when the key is missing.

## Test

```bash
python evaluation/evaluate.py
```

## API

`GET /health`

Returns service status.

`POST /chat`

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a Java developer with stakeholder management skills"
    }
  ]
}
```

Returns:

```json
{
  "reply": "...",
  "recommendations": [
    {
      "name": "...",
      "url": "...",
      "test_type": "...",
      "duration": "...",
      "remote": "yes",
      "adaptive": "no",
      "score": 1.23
    }
  ],
  "end_of_conversation": true
}
```
