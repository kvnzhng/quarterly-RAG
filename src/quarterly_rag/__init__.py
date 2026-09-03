"""quarterly-RAG: local RAG over SEC 10-Q/10-K filings.

Pipeline layers (left may not import right):
ingestion -> chunking -> indexing -> retrieval -> generation -> evaluation
"""

__version__ = "0.1.0"
