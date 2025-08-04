# Architecture & Trade-offs

## Overview
- The API exposes a single POST /ask endpoint for Q&A.
- Uses FastAPI for minimal, high-performance web service.
- Pydantic models ensure strict schema validation for all responses.
- Dockerfile and requirements.txt provide easy deployment and reproducibility.

## Trade-offs
- The answer logic is currently a stub; real retrieval or model integration is needed for production.
- No persistent storage or session management for simplicity.
- Only essential dependencies are included for minimal footprint.
- Strict schema validation may require updates if requirements change.
