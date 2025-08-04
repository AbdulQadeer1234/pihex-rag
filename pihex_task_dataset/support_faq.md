# FAQ

**Does PiHex store prompts and outputs?**  
- Defaults to 30 days retention for auditability; configurable to 0 in *Settings*.

**Can I get references in answers?**  
- Yes. Set `return_sources=true` and we include snippets plus doc ids.

**Do you support on-prem?**  
- Enterprise only; requires Kubernetes 1.27+ and object storage (S3 API).

**What happens on hallucination risk?**  
- You can enable `strict_schema=true` to enforce JSON schema with function calling.
