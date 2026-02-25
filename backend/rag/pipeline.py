"""Default pipeline alias — uses AgenticRAGPipeline."""
from backend.rag.langchain_pipeline import AgenticRAGPipeline, get_langchain_pipeline
__all__ = ["AgenticRAGPipeline", "get_langchain_pipeline"]
