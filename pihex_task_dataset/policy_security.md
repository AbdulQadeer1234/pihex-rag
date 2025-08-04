# Security & Compliance

- Data encrypted in transit (TLS 1.3) and at rest (AES-256).
- Regional data residency: `in` (India) and `eu` (Frankfurt) regions.
- PII redaction is enabled by default on `/v1/ingest`.
- Extract/Query requests are **not** used for model training.
- Access controls via workspace roles: `owner`, `admin`, `member`, `viewer`.
