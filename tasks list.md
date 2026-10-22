### **Part 1: Environment & Foundational Configuration**

Set up configurations, constants, environment settings, and basic utility helpers.

1. `.gitignore`
2. `.env.example`
3. `pyproject.toml` / `requirements.txt`
4. `src/utils/file_utils.py`
5. `src/utils/text_utils.py`
6. `src/config/settings.py`
7. `src/config/model_config.py`

---

### **Part 2: Domain Data Contracts & Ingestion Layer**

Define the standard data models (Pydantic / Dataclasses) used across the pipeline, followed by document parsing.

1. `src/domain/documents.py`
2. `src/domain/chunks.py`
3. `src/domain/responses.py`
4. `src/ingestion/loader_factory.py`
5. `src/ingestion/pdf_loader.py`
6. `src/ingestion/docx_loader.py`
7. `src/ingestion/document_registry.py`
8. `src/preprocessing/cleaner.py`
9. `src/preprocessing/metadata_extractor.py`

---

### **Part 3: Chunking, Embeddings & Vector Storage**

Split documents into contextual chunks, generate vector embeddings, and build the FAISS index manager.

1. `src/chunking/chunk_metadata.py`
2. `src/chunking/text_splitter.py`
3. `src/embeddings/embedding_model.py`
4. `src/vector_store/base.py`
5. `src/vector_store/faiss_store.py`
6. `src/vector_store/index_manager.py`

---

### **Part 4: Retrieval, Re-ranking & Context Assembly**

Implement similarity search, metadata filtering, cross-encoder re-ranking, and context formatting.

1. `src/retrieval/filters.py`
2. `src/retrieval/similarity_search.py`
3. `src/retrieval/retriever.py`
4. `src/reranking/reranker.py`
5. `src/rag/context_builder.py`
6. `src/rag/citations.py`

---

### **Part 5: Generation, Guardrails & Core Pipeline Assembly**

Set up the LLM integration, prompt templates, output validation, and the unified RAG orchestrator.

1. `src/generation/prompt_templates.py`
2. `src/generation/llm_client.py`
3. `src/generation/answer_generator.py`
4. `src/rag/guardrails.py`
5. `src/rag/pipeline.py`

---

### **Part 6: Standalone Scripts (Indexing & Exploration)**

Create the CLI workflows to ingest sample HR documents and build/inspect FAISS artifacts before wiring up the web framework.

1. `scripts/ingest_documents.py`
2. `scripts/build_index.py`
3. `scripts/inspect_index.py`

---

### **Part 7: Evaluation Framework**

Implement metrics to evaluate retrieval performance, faithfulness, and answer relevance.

1. `src/evaluation/retrieval_metrics.py`
2. `src/evaluation/answer_metrics.py`
3. `src/evaluation/faithfulness.py`
4. `src/evaluation/evaluation_runner.py`
5. `scripts/evaluate_rag.py`

---

### **Part 8: Backend Service & API Layer (FastAPI)**

Expose the RAG engine over REST API endpoints with proper error handling and schemas.

1. `app/backend/schemas/common.py`
2. `app/backend/schemas/documents.py`
3. `app/backend/schemas/chat.py`
4. `app/backend/exceptions.py`
5. `app/backend/middleware.py`
6. `app/backend/services/document_service.py`
7. `app/backend/services/chat_service.py`
8. `app/backend/dependencies.py`
9. `app/backend/lifespan.py`
10. `app/backend/api/v1/routers/health.py`
11. `app/backend/api/v1/routers/documents.py`
12. `app/backend/api/v1/routers/chat.py`
13. `app/backend/api/v1/routers/admin.py`
14. `app/backend/main.py`

---

### **Part 9: Frontend User Interface (Streamlit)**

Build the interactive client to chat with policies, upload documents, and view metrics.

1. `app/frontend/config.py`
2. `app/frontend/api_client.py`
3. `app/frontend/components/chat_message.py`
4. `app/frontend/components/source_card.py`
5. `app/frontend/Home.py`
6. `app/frontend/pages/1_Chat.py`
7. `app/frontend/pages/2_Policy_Library.py`
8. `app/frontend/pages/3_Evaluation.py`
9. `app/frontend/pages/4_Settings.py`

---

### **Part 10: Testing, Containerization & Documentation**

Verify functionality with automated tests, dockerize both services, and finalize project instructions.

1. `tests/unit/test_chunking.py`
2. `tests/unit/test_retrieval.py`
3. `tests/unit/test_citations.py`
4. `tests/integration/test_rag_pipeline.py`
5. `tests/api/test_health.py`
6. `tests/api/test_chat.py`
7. `deployment/Dockerfile.backend`
8. `deployment/Dockerfile.frontend`
9. `deployment/docker-compose.yml`
10. `Makefile`
11. `README.md`

back to school