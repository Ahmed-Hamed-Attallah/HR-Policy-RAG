"""High-level domain retriever converting search hits to structured RetrievedChunks."""

import logging
from typing import List, Optional

from src.config.model_config import retrieval_config
from src.domain.responses import RetrievedChunk
from src.retrieval.filters import MetadataFilter
from src.retrieval.similarity_search import SimilaritySearchEngine
from src.vector_store.index_manager import IndexManager

logger = logging.getLogger(__name__)


class PolicyRetriever:
    """Facade for policy-aware chunk retrieval and domain object packaging."""

    def __init__(
        self,
        index_manager: Optional[IndexManager] = None,
        search_engine: Optional[SimilaritySearchEngine] = None,
    ):
        if search_engine:
            self.search_engine = search_engine
        else:
            mgr = index_manager or IndexManager()
            mgr.load_or_create()
            self.search_engine = SimilaritySearchEngine(index_manager=mgr)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_criteria: Optional[MetadataFilter] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve most relevant policy chunks packaged as typed RetrievedChunk objects."""
        raw_matches = self.search_engine.search(
            query=query,
            top_k=top_k or retrieval_config.top_k,
            filter_criteria=filter_criteria,
            score_threshold=score_threshold,
        )

        retrieved_chunks: List[RetrievedChunk] = []
        for chunk, score in raw_matches:
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity_score=float(score),
                    rerank_score=None,
                )
            )

        return retrieved_chunks

    def retrieve_by_document(
        self,
        query: str,
        document_name: str,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """Targeted retrieval restricted to a specific policy document."""
        criteria = MetadataFilter(document_names=[document_name])
        return self.retrieve(query=query, top_k=top_k, filter_criteria=criteria)