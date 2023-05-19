import os
import json
from typing import Dict, List, Optional
import marqo
import warnings
import re
from tenacity import retry, wait_random_exponential, stop_after_attempt

from datastore.datastore import DataStore
from models.models import (
    Document,
    DocumentChunk,
    DocumentChunkMetadata,
    DocumentChunkWithScore,
    DocumentMetadataFilter,
    QueryResult,
    Query
)
from services.date import to_unix_timestamp
from services.chunks import create_document_chunks

# Read environment variables for Marqo configuration
MARQO_API_URL = os.environ.get("MARQO_API_URL")
MARQO_API_KEY = os.environ.get("MARQO_API_KEY", "")
MARQO_INDEX = os.environ.get("MARQO_INDEX", "chatgpt-retrieval")
assert MARQO_API_URL is not None
assert MARQO_API_KEY is not None
assert MARQO_INDEX is not None

# optional configuration environmental variables
MARQO_INFERENCE_MODEL = os.environ.get('MARQO_INFERENCE_MODEL')
MARQO_UPSERT_BATCH_SIZE = os.environ.get('MARQO_UPSERT_BATCH_SIZE')
TREAT_URLS_AND_POINTERS_AS_IMAGES = os.environ.get('TREAT_URLS_AND_POINTERS_AS_IMAGES')

# batchsize for upsert operations
if not MARQO_UPSERT_BATCH_SIZE:
    MARQO_UPSERT_BATCH_SIZE = 64

# if model not provided then use default
if not MARQO_INFERENCE_MODEL:
    MARQO_INFERENCE_MODEL = None

if not TREAT_URLS_AND_POINTERS_AS_IMAGES or TREAT_URLS_AND_POINTERS_AS_IMAGES.lower() != "true":
    TREAT_URLS_AND_POINTERS_AS_IMAGES = False
else:
    TREAT_URLS_AND_POINTERS_AS_IMAGES = True

class MarqoDataStore(DataStore):
    def __init__(self):

        self.client = marqo.Client(url=MARQO_API_URL, api_key=MARQO_API_KEY)
        self.index_name = MARQO_INDEX
        try:
            self.client.create_index(self.index_name, model=MARQO_INFERENCE_MODEL, treat_urls_and_pointers_as_images=TREAT_URLS_AND_POINTERS_AS_IMAGES)
            print(f"Created index {self.index_name}")
        except marqo.errors.MarqoWebError:
            print(f"Using existing index {self.index_name}")

    async def upsert(
        self, documents: List[Document], chunk_token_size: Optional[int] = None
    ) -> List[str]:
        # Initialize an empty dictionary of lists of chunks
        chunks: Dict[str, List[DocumentChunk]] = {}

        # Initialize an empty list of all chunks
        all_chunks: List[DocumentChunk] = []

        # Loop over each document and create chunks
        for doc in documents:
            doc_chunks, doc_id = create_document_chunks(doc, chunk_token_size)

            # Append the chunks for this document to the list of all chunks
            all_chunks.extend(doc_chunks)

            # Add the list of chunks for this document to the dictionary with the document id as the key
            chunks[doc_id] = doc_chunks
        print(all_chunks)
        return await self._upsert(all_chunks)
    
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
        images = set()
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

                if TREAT_URLS_AND_POINTERS_AS_IMAGES:
                    urls = self._extract_urls(chunk.text)
                    for idx, url in urls:
                        document[f"img{idx}"] = url
                        images.add(f"img{idx}")

                documents.append(document)

                fields |= document.keys()

        non_tensor_fields = list(fields-{*images, "text"})

        # insert all documents
        request_batch_size = 1024
        for i in range(0, len(documents), request_batch_size):
            response = self.client.index(self.index_name).add_documents(
                documents=documents[i:i+request_batch_size],
                non_tensor_fields=non_tensor_fields,
                client_batch_size=MARQO_UPSERT_BATCH_SIZE,
                server_batch_size=MARQO_UPSERT_BATCH_SIZE,
            )
            
            if response['errors']:
                print(f"ERROR: Errors occured while adding documents, error occured in index range {i}-{i+request_batch_size}")
                print(response)

        return doc_ids

    async def query(self, queries: List[Query]) -> List[QueryResult]:
        """
        Takes in a list of queries and filters and returns a list of query results with matching document chunks and scores.
        """
        return await self._query(queries)
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    async def _query(
        self,
        queries: List[Query],
    ) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """

        # if working with images as well then extract image urls from the query and create a new query object that composes the text and images together
        if TREAT_URLS_AND_POINTERS_AS_IMAGES:
            for i in range(len(queries)):
                urls = self._extract_urls(queries[i].query)
                print(urls)
                new_query = {}
                new_query[queries[i].query] = 1
                print(new_query)
                for url in urls:
                    new_query[url] = 1
                print(new_query)
                queries[i] = new_query

        results = self.client.bulk_search(
            [{"index": self.index_name, "q": q.query, "limit": q.top_k, 'filter': q.filter} for q in queries]
        )

        import json 

        print(json.dumps(results, indent=4))

        query_results: List[QueryResult] = []
   
        for result in results['result']:
            doc_chunks: List[DocumentChunkWithScore] = []
            for hit in result['hits']:
                metadata: Dict[str, str] = json.loads(hit.get('metadata', "{}"))
                doc_metadata = DocumentChunkMetadata(
                    source=metadata.get('source'),
                    source_id=metadata.get('source_id'),
                    url=metadata.get('url'),
                    created_at=metadata.get('created_at'),
                    author=metadata.get('author')
                )
                doc_chunk = DocumentChunkWithScore(
                    id=hit.get('_id'), 
                    text=hit.get('text'), 
                    metadata=doc_metadata, 
                    embedding=None, 
                    score=hit.get('_score')
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
            try:
                print("Deleting all documents")
                response = self.client.delete_index(self.index_name)
                print(response)
            except Exception as e:
                print("Error deleting index")
                raise e
                
            try:
                response = self.client.create_index(self.index_name, model=MARQO_INFERENCE_MODEL, treat_urls_and_pointers_as_images=TREAT_URLS_AND_POINTERS_AS_IMAGES)
                print(response)
            except Exception as e:
                print("Error creating new index after deletion")
                raise e
            
        if ids:
            try:
                print(f"Deleting {len(ids)} documents by id")
                response = self.client.index(self.index_name).delete_documents(ids)
                print(response)
            except:
                print("Error deleting documents.")
                raise e
    
        if filter:
            warnings.warn("Delete with filter is not implemented for Marqo")

        return True

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
    
    def _extract_urls(self, text: str) -> List[str]:
        return re.findall('https:.*?\.(?:png|jpg)', text)
