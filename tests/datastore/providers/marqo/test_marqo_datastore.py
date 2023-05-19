import pytest
import contextlib
import os
from typing import List
from models.models import (
    DocumentChunkMetadata,
    QueryResult,
    Document,
    Query,
    Source,
)

from datastore.providers.marqo_datastore import (
    MarqoDataStore,
)


@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
        


def marqo_datastore_with_settings(**kwargs) -> MarqoDataStore:
    with set_env(**kwargs):
        datastore = MarqoDataStore()
        datastore.delete(ids=[], delete_all=True)
        return datastore


@pytest.mark.asyncio
async def test_marqo():
    marqo_datastore = marqo_datastore_with_settings()
    assert marqo_datastore.client.get_indexes()['results']

@pytest.fixture
def text_query():
    return Query(query="This is a query")


@pytest.fixture
def multimodal_text_query():
    return Query(query="This is a photo of a plane https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg")

@pytest.fixture
def documents():
    doc1 = Document(
        id="one",
        text="This is a document about cats")
    doc2 = Document(
        id="two",
        text="This is a document about dogs")
    doc3 = Document(
        id="three",
        text="This is a document about hippos")
    doc4 = Document(id="four", text="This document has a plane https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg")
    return [doc1, doc2, doc3]

@pytest.mark.asyncio
async def test_datastore_upsert(documents: List[Document]):
    marqo_datastore = marqo_datastore_with_settings(MARQO_INDEX_NAME="marqo-chatgpt-retrieval-test")
    doc_ids = marqo_datastore.upsert(documents, chunk_token_size=512)
    assert doc_ids == [doc.id for doc in documents]
    assert marqo_datastore.client.index(marqo_datastore.index_name).get_stats()["numberOfDocuments"] == len(documents)

@pytest.mark.asyncio
async def test_datastore_upsert_multimodal(documents: List[Document]):
    marqo_datastore = marqo_datastore_with_settings(
        MARQO_INDEX_NAME="marqo-chatgpt-multimodal-retrieval-test",
        TREAT_URLS_AND_POINTERS_AS_IMAGES="True",
        MARQO_INFERENCE_MODEL="ViT-L/14"
    )
    doc_ids = marqo_datastore.upsert(documents, chunk_token_size=512)
    assert doc_ids == [doc.id for doc in documents]
    assert marqo_datastore.client.index(marqo_datastore.index_name).get_stats()["numberOfDocuments"] == len(documents)

    print(marqo_datastore.client.index(marqo_datastore.index_name).get_documents([doc_ids[-1]]))

@pytest.mark.asyncio
async def test_datastore_delete(documents: List[Document]):
    marqo_datastore = marqo_datastore_with_settings(MARQO_INDEX_NAME="marqo-chatgpt-retrieval-test")
    doc_ids = marqo_datastore.upsert(documents, chunk_token_size=512)
    marqo_datastore.delete([doc.id for doc in documents][:1])
    assert marqo_datastore.client.index(marqo_datastore.index_name).get_stats()["numberOfDocuments"] < len(doc_ids)


@pytest.mark.asyncio
async def test_datastore_query(documents: List[Document], text_query: Query):
    marqo_datastore = marqo_datastore_with_settings(MARQO_INDEX_NAME="marqo-chatgpt-retrieval-test")
    doc_ids = marqo_datastore.upsert(documents, chunk_token_size=512)
    
    match: QueryResult = await marqo_datastore.query([text_query])[0]
    assert len(match.results) == 3
    assert len({doc.id for doc in documents}-{r.id for r in match.results}) == 1
    assert len(set(doc_ids)-{r.id for r in match.results}) == 1

@pytest.mark.asyncio
async def test_datastore_query(documents: List[Document], multimodal_text_query: Query):
    marqo_datastore = marqo_datastore_with_settings(
        MARQO_INDEX_NAME="marqo-chatgpt-multimodal-retrieval-test",
        TREAT_URLS_AND_POINTERS_AS_IMAGES="True",
        MARQO_INFERENCE_MODEL="ViT-L/14"
    )
    doc_ids = marqo_datastore.upsert(documents, chunk_token_size=512)
    
    match: QueryResult = await marqo_datastore.query([multimodal_text_query])[0]
    assert len(match.results) == 3
    assert len({doc.id for doc in documents}-{r.id for r in match.results}) == 1
    assert len(set(doc_ids)-{r.id for r in match.results}) == 1
