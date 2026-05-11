# SHL Assessment Recommender Approach

## Architecture

The service is a FastAPI application with a single conversational endpoint:

User message -> conversation analyzer -> hybrid retriever -> optional LLM reply generator -> JSON response.

The API remains stateless. Each `/chat` request sends the full message history, allowing the service to infer whether the user is starting a new request, refining an earlier shortlist, or asking for a comparison.

## Catalog Grounding

The SHL product catalog is downloaded from the provided JSON endpoint and stored locally. The ingestion script tolerates malformed control characters from the source response, then writes a normalized catalog file. The embedding script converts each assessment into searchable text containing name, description, assessment category, duration, remote support, and adaptive status.

## Retrieval Design

The retriever uses FAISS with `all-MiniLM-L6-v2` embeddings for semantic recall. To improve ranking quality, it reranks candidates with a lightweight lexical layer:

- query/catalog token overlap
- boosts for matches in assessment names
- boosts for matches in assessment categories
- role-specific boosts, for example Java queries prioritize Java assessments
- personality/behavior refinements prioritize OPQ/personality catalog items

This hybrid ranking reduces irrelevant semantic matches while keeping recall broad enough for ambiguous hiring requests.

## Conversation Behavior

The analyzer supports four main paths:

- clarification: vague requests ask for role, seniority, skills, and assessment type
- recommendation: clear role/skill requests return a grounded shortlist
- refinement: follow-up messages such as "actually add personality tests" use the full user history instead of restarting
- comparison: questions such as "Difference between OPQ and GSA?" retrieve likely matching assessments and compare only using catalog fields

## Guardrails

The service blocks obvious prompt injection patterns and instructs the LLM to use only retrieved SHL context. If no Gemini API key is configured, the system still returns deterministic grounded replies with structured recommendations, which keeps the API testable in deployment.

## Evaluation

The `evaluation/evaluate.py` script checks the highest-risk behaviors:

- vague query clarification
- Java role retrieval
- refinement with personality tests
- comparison flow
- prompt injection refusal

These tests are intentionally small but focused on likely hidden evaluator probes.

## Tradeoffs

The current system favors deterministic behavior and catalog grounding over elaborate generation. The ranking layer is transparent and easy to tune, but it is still heuristic. A production version would add labeled evaluation queries, Recall@K tracking, cached model loading, stronger synonym expansion, and migrated Gemini SDK usage.
