import pytest
from typing import List
import os
from models.models import (
    QueryResult,
    Document,
    Query,
)

from datastore.providers.marqo_datastore import (
    MarqoDataStore,
)

MARQO_INDEX = os.environ.get("MARQO_INDEX", "chatgpt-retrieval")

async def marqo_datastore_with_settings() -> MarqoDataStore:
    datastore = MarqoDataStore()
    await datastore.delete(ids=[], delete_all=True)
    return datastore

@pytest.mark.asyncio
async def test_marqo():
    marqo_datastore = await marqo_datastore_with_settings()
    assert marqo_datastore.client.get_indexes()['results']

@pytest.fixture
def text_query():
    return Query(query="This is a query", top_k=3)


@pytest.fixture
def multimodal_query():
    return Query(query="What is this https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg", top_k=3)

@pytest.fixture
def image_query():
    return Query(query="https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg", top_k=3)

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
    return [doc1, doc2, doc3, doc4]

@pytest.mark.asyncio
async def test_datastore_upsert(documents: List[Document]):
    marqo_datastore = await marqo_datastore_with_settings()
    doc_ids = await marqo_datastore.upsert(documents, chunk_token_size=512)
    assert doc_ids == [doc.id for doc in documents]
    assert marqo_datastore.client.index(MARQO_INDEX).get_stats()["numberOfDocuments"] == len(documents)

@pytest.mark.asyncio
async def test_datastore_upsert_multimodal(documents: List[Document]):
    marqo_datastore = await marqo_datastore_with_settings()
    doc_ids = await marqo_datastore.upsert(documents, chunk_token_size=512)
    assert doc_ids == [doc.id for doc in documents]
    assert marqo_datastore.client.index(MARQO_INDEX).get_stats()["numberOfDocuments"] == len(documents)

    print(marqo_datastore.client.index(MARQO_INDEX).get_documents([doc_ids[-1]]))

@pytest.mark.asyncio
async def test_datastore_delete(documents: List[Document]):
    marqo_datastore = await marqo_datastore_with_settings()
    doc_ids = await marqo_datastore.upsert(documents, chunk_token_size=512)
    await marqo_datastore.delete([doc.id for doc in documents][:1])
    assert marqo_datastore.client.index(MARQO_INDEX).get_stats()["numberOfDocuments"] < len(doc_ids)


@pytest.mark.asyncio
async def test_datastore_query(documents: List[Document], text_query: Query):
    marqo_datastore = await marqo_datastore_with_settings()
    doc_ids = await marqo_datastore.upsert(documents, chunk_token_size=512)
    
    matches: List[QueryResult] = await marqo_datastore.query([text_query])
    match = matches[0]
    assert len(match.results) == 3
    assert len({doc.id for doc in documents}-{r.id for r in match.results}) == 1
    assert len(set(doc_ids)-{r.id for r in match.results}) == 1

@pytest.mark.asyncio
async def test_datastore_multimodal_query(documents: List[Document], multimodal_query: Query):
    marqo_datastore = await marqo_datastore_with_settings()
    doc_ids = await marqo_datastore.upsert(documents, chunk_token_size=512)
    
    matches: List[QueryResult] = await marqo_datastore.query([multimodal_query])
    match = matches[0]
    assert len(match.results) == 3
    assert len({doc.id for doc in documents}-{r.id for r in match.results}) == 1
    assert len(set(doc_ids)-{r.id for r in match.results}) == 1


@pytest.mark.asyncio
async def test_datastore_image_query(documents: List[Document], image_query: Query):
    marqo_datastore = await marqo_datastore_with_settings()
    doc_ids = await marqo_datastore.upsert(documents, chunk_token_size=512)
    
    matches: List[QueryResult] = await marqo_datastore.query([image_query])
    match = matches[0]
    assert len(match.results) == 3
    assert len({doc.id for doc in documents}-{r.id for r in match.results}) == 1
    assert len(set(doc_ids)-{r.id for r in match.results}) == 1
