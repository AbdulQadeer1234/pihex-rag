# Quickstart: Extract

1. Create an API key.
2. `POST /v1/extract` with a JSON body containing:
   - `text`: the raw input string.
   - `schema`: a JSON Schema describing required fields.
3. The response contains `data` (JSON) and `sources` (citations).

### Example
```
POST /v1/extract
{ "text": "Book a meeting with Raj on 12 Aug at 3pm IST", "schema": {...} }
```
Response:
```
{
  "data": { "person": "Raj", "date": "2025-08-12", "time": "15:00", "tz": "Asia/Kolkata" },
  "sources": [{ "doc": "product_quickstart.md", "snippet": "Book a meeting..." }]
}
```
