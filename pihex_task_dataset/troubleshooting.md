# Troubleshooting

- **401 Unauthorized**: Check API key, workspace, and region header.
- **429 Too Many Requests**: Back off per `Retry-After`. Consider upgrading plan.
- **Hallucinations**: Use `return_sources=true` and tighten schema; add negative examples.
- **Slow queries**: Enable `hybrid=true` to combine keyword and vector search.
