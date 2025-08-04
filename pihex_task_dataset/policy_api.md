# PiHex API Policy (v1.2, 2025-07-15)

## Authentication
- All requests require `Authorization: Bearer <API_KEY>`.
- Per-project API keys; rotate at least once every 90 days.
- Keys can be created/revoked in *Settings → API Keys*.

## Rate Limits
- Free: 20 req/min, 30k tokens/day.
- Pro: 60 req/min, 200k tokens/day.
- Enterprise: negotiated.
- 429s include a `Retry-After` header (seconds).

## Endpoints
- `POST /v1/ingest` – Accepts NDJSON of records.
- `POST /v1/query` – Ask questions over your workspace.
- `POST /v1/extract` – Extract structured fields from unstructured text.
- `GET /v1/usage` – Token and request usage for the current billing period.

## Webhooks
- Optional webhooks at `/v1/hooks/events` for job completion.
