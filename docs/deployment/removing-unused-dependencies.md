# Removing Unused Dependencies

Before deploying your app, you might want to remove unused dependencies from your [pyproject.toml](/pyproject.toml) file to reduce the size of your app and improve its performance. Depending on the vector database provider you choose, you can remove the packages that are not needed for your specific provider.

Here are the packages you can remove for each vector database provider:

- **Pinecone:** Remove `weaviate-client`, `pymilvus`, `qdrant-client`, `redis`, `chromadb`, `marqo`, `llama-index`, `azure-identity`, `azure-search-documents`, `supabase`, `psycopg2`+`pgvector`, and `psycopg2cffi`.
- **Weaviate:** Remove `pinecone-client`, `pymilvus`, `qdrant-client`, `redis`, `chromadb`, `marqo`, `llama-index`, `azure-identity` and `azure-search-documents`, `supabase`, `psycopg2`+`pgvector`, `psycopg2cffi`.
- **Marqo:** Remove `pinecone-client`, `weaviate-client`, `pymilvus`, `qdrant-client`, `redis`, `chromadb`, `llama-index`, `azure-identity` and `azure-search-documents`, `supabase`, `psycopg2`+`pgvector`, and `psycopg2cffi`..
- **Zilliz:** Remove `pinecone-client`, `weaviate-client`, `qdrant-client`, `redis`, `chromadb`, `marqo`, `llama-index`, `azure-identity` and `azure-search-documents`, `supabase`, `psycopg2`+`pgvector`, and `psycopg2cffi`.
- **Milvus:** Remove `pinecone-client`, `weaviate-client`, `qdrant-client`, `redis`, `chromadb`, `marqo`, `llama-index`, `azure-identity` and `azure-search-documents`, `supabase`, `psycopg2`+`pgvector`, and `psycopg2cffi`.
- **Qdrant:** Remove `pinecone-client`, `weaviate-client`, `pymilvus`, `redis`, `chromadb`, `marqo`, `llama-index`, `azure-identity` and `azure-search-documents`, `supabase`, `psycopg2`+`pgvector`, and `psycopg2cffi`.
- **Redis:** Remove `pinecone-client`, `weaviate-client`, `pymilvus`, `qdrant-client`, `chromadb`, `marqo`, `llama-index`, `azure-identity` and `azure-search-documents`, `supabase`, `psycopg2`+`pgvector`, and `psycopg2cffi`.
- **LlamaIndex:** Remove `pinecone-client`, `weaviate-client`, `pymilvus`, `qdrant-client`, `chromadb`, `marqo`, `redis`, `azure-identity` and `azure-search-documents`, `supabase`, `psycopg2`+`pgvector`, and `psycopg2cffi`.
- **Chroma:**: Remove `pinecone-client`, `weaviate-client`, `pymilvus`, `qdrant-client`, `marqo`, `llama-index`, `redis`, `azure-identity` and `azure-search-documents`, `supabase`, `psycopg2`+`pgvector`, and `psycopg2cffi`.
- **Azure Cognitive Search**: Remove `pinecone-client`, `weaviate-client`, `pymilvus`, `qdrant-client`, `marqo`, `llama-index`, `redis` and `chromadb`, `supabase`, `psycopg2`+`pgvector`, and `psycopg2cffi`.
- **Supabase:** Remove `pinecone-client`, `weaviate-client`, `pymilvus`, `qdrant-client`, `marqo`, `redis`, `llama-index`, `azure-identity` and `azure-search-documents`, `psycopg2`+`pgvector`, and `psycopg2cffi`.
- **Postgres:** Remove `pinecone-client`, `weaviate-client`, `pymilvus`, `qdrant-client`, `marqo`, `redis`, `llama-index`, `azure-identity` and `azure-search-documents`, `supabase`, and `psycopg2cffi`.
- **AnalyticDB:** Remove `pinecone-client`, `weaviate-client`, `pymilvus`, `qdrant-client`, `marqo`, `redis`, `llama-index`, `azure-identity` and `azure-search-documents`, `supabase`, and `psycopg2`+`pgvector`.

After removing the unnecessary packages from the `pyproject.toml` file, you don't need to run `poetry lock` and `poetry install` manually. The provided Dockerfile takes care of installing the required dependencies using the `requirements.txt` file generated by the `poetry export` command.
