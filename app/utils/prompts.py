PROMPT_TEMPLATE = """
You are an expert assistant. Given a user question and context, answer it along with citations for each source. 
Answer strictly in this JSON format: 
{{"answer": "<string>", "category": "<api|security|pricing|support|other>", "confidence": <float 0-1>, "sources": [{{"doc": "<document_name>", "snippet": "<source_snippet>"}}]}}
"Context: {context}\nUser Question: {question}\nJSON Output:"
"""