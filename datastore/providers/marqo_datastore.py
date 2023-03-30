import os
import json
from typing import Dict, List, Optional
import marqo
import warnings
from tenacity import retry, wait_random_exponential, stop_after_attempt

from datastore.datastore import DataStore
from models.models import (
    DocumentChunk,
    DocumentChunkMetadata,
    DocumentChunkWithScore,
    DocumentMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
)
from services.date import to_unix_timestamp

# Read environment variables for Marqo configuration
MARQO_API_URL = os.environ.get("MARQO_API_URL")
MARQO_API_KEY = os.environ.get("MARQO_API_KEY")
MARQO_INDEX = os.environ.get("MARQO_INDEX")
assert MARQO_API_URL is not None
assert MARQO_API_KEY is not None
assert MARQO_INDEX is not None

# optional configuration environmental variables
MARQO_INFERENCE_MODEL = os.environ.get('MARQO_INFERENCE_MODEL')
MARQO_UPSERT_BATCH_SIZE = os.environ.get('MARQO_UPSERT_BATCH_SIZE')

# batchsize for upsert operations
if not MARQO_UPSERT_BATCH_SIZE:
    MARQO_UPSERT_BATCH_SIZE = 64

# if model not provided then use default
if not MARQO_INFERENCE_MODEL:
    MARQO_INFERENCE_MODEL = "sentence-transformers/all-mpnet-base-v2"

class MarqoDataStore(DataStore):
    def __init__(self):

        self.client = marqo.Client(url=MARQO_API_URL, api_key=MARQO_API_KEY)

        try:
            self.client.create_index(MARQO_INDEX)
            print(f"Created index {MARQO_INDEX}")
        except marqo.errors.MarqoWebError:
            print(f"Using existing index {MARQO_INDEX}")

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a dict from document id to list of document chunks and inserts them into the index.
        Return a list of document ids.
        """
        # Initialize a list of ids to return
        doc_ids: List[str] = []

        # Initialize a list of documents
        documents: List[str] = []
        fields = set()
        for doc_id, chunk_list in chunks.items():
            # Append the id to the ids list
            doc_ids.append(doc_id)
            print(f"Upserting document_id: {doc_id}")
            for chunk in chunk_list:

                # Get the wrangled metadata
                metadata = self._wrangle_metadata(chunk.metadata)

                # Add the text and document id to the document dict
                document = {}
                document["text"] = chunk.text
                document["document_id"] = doc_id
                document["_id"] = doc_id
                document["metadata"] = metadata

                documents.append(document)

                fields |= document.keys()

        non_tensor_fields = list(fields-{"text",})

        # insert all documents
        request_batch_size = 1024
        for i in range(0, len(documents), request_batch_size):
            response = self.client.index(MARQO_INDEX).add_documents(
                documents=documents[i:i+request_batch_size],
                non_tensor_fields=non_tensor_fields,
                client_batch_size=MARQO_UPSERT_BATCH_SIZE,
                server_batch_size=MARQO_UPSERT_BATCH_SIZE,
            )
            
            if response['errors']:
                print(f"ERROR: Errors occured while adding documents, error occured in index range {i}-{i+request_batch_size}")
                print(response)

        return doc_ids

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    async def _query(
        self,
        queries: List[QueryWithEmbedding],
    ) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """

        results = self.client.bulk_search(
            [{"index": MARQO_INDEX, "q": q.query, "limit": q.top_k, 'filter': q.filter} for q in queries]
        )

        query_results: List[QueryResult] = []

        for result in results['result']:
            doc_chunks: List[DocumentChunkWithScore] = []
            for hit in result['hits']:
                metadata: Dict[str, str] = json.loads(hit['metadata'])
                doc_metadata = DocumentChunkMetadata(
                    source=metadata.get('source'),
                    source_id=metadata.get('source_id'),
                    url=metadata.get('url'),
                    created_at=metadata.get('created_at'),
                    author=metadata.get('author')
                )
                doc_chunk = DocumentChunkWithScore(
                    id=hit['_id'], 
                    text=hit['text'], 
                    metadata=doc_metadata, 
                    embedding=hit['embedding'], 
                    score=hit['_score']
                )
                doc_chunks.append(doc_chunk)
            query_results.append(QueryResult(query=result['query'], results=doc_chunks))

        return query_results

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """
        Removes vectors by ids, filter, or everything from the index.
        """
        if delete_all:
            print("Deleting all documents")
            response = self.client.delete_index(MARQO_INDEX)
            print(response)
            response = self.client.create_index(MARQO_INDEX)
            print(response)

        if ids:
            print(f"Deleting {len(ids)} documents by id")
            response = self.client.index(MARQO_INDEX).delete_documents(ids)
            print(response)
    
        if filter:
            warnings.warn("Delete with filter is not implemented for Marqo")

    def _get_marqo_filter(
        self, filter: Optional[DocumentMetadataFilter] = None
    ) -> str:
        if filter is None:
            return ""

        marqo_filter = {}
        # For each field in the MetadataFilter, check if it has a value and add the corresponding Marqo filter expression
        for field, value in filter.dict().items():
            if value is None: continue 

            if field == "start_date":
                f = marqo_filter.get("date", ["*", "*"])
                f[0] = str(to_unix_timestamp(value))
                marqo_filter["date"] = f
            elif field == "end_date":
                f = marqo_filter.get("date", ["*", "*"])
                f[1] = str(to_unix_timestamp(value))
                marqo_filter["date"] = f
            else:
                marqo_filter[field] = value

        filter_terms: List[str] = []
        for f in marqo_filter:
            if isinstance(marqo_filter[f], list):
                marqo_filter[f] = f"[{marqo_filter[f][0]} TO {marqo_filter[f][0]}]"
            filter_terms.append(f"{f}: {marqo_filter[f]}")

        return " AND ".join(filter_terms)

    def _wrangle_metadata(
        self, metadata: Optional[DocumentChunkMetadata] = None
    ) -> str:
        if metadata is None:
            return {}

        processed_metadata = {**metadata}

        # For fields that are dates, convert them to unix timestamps
        for field, value in processed_metadata.dict().items():
            if field == "created_at":
                processed_metadata[field] = to_unix_timestamp(value)

        return json.dumps(processed_metadata)
