import contextlib
import csv
from datetime import datetime
import json
import time
import os
import unittest
import uuid

from dotenv import load_dotenv

from cosmosdb import CosmosDB
from embedding import BAAIEmbeddingModel

load_dotenv()

# JSON_DOC = "small-sample-text.json"
JSON_DOC = "sample-text.json"
QUERY = "Azure service that enables you to run code on-demand"


async def write_to_csv(data: list):
    csv_path = "report.csv"
    header = ["Timestamp", "Test", "Duration (s)"]
    data.insert(0, str(datetime.now()))

    if not os.path.exists(csv_path):
        with open(csv_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)
            csv_writer.writerow(data)
    else:
        with open(csv_path, "a") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(data)


@contextlib.asynccontextmanager
async def timed(operation: str):
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    await write_to_csv([operation, round(duration, 6)])


class TestCosmosDB(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._db = CosmosDB(
            endpoint=os.environ.get("ENDPOINT"),
            key=os.environ.get("KEY"),
            database="test-json-db",
            container="test-json-container"
        )
        self._embedding_model = BAAIEmbeddingModel(
            dimension_count=1024,
            embedding_type="dense",
            metric="cosine",
            supported_languages=["en"]
        )

    async def test_indexing(self):
        data = None
        with open(JSON_DOC, "r") as file:
            data = json.load(file)

        text = []
        for item in data:
            text.append(item["title"])
            text.append(item["content"])

        embeddings = None
        async with timed("embed-docs"):
            embeddings = await self._embedding_model.embed_documents(text)

        for i, item in enumerate(data):
            item["id"] = str(uuid.uuid4())
            item["titleVector"] = embeddings[i]
            item["contentVector"] = embeddings[i + 1]
            item["@search.action"] = "upload"
        
        async with timed("index"):
            await self._db.index_vectors(data)

    async def test_search(self):
        embedding = None
        async with timed("embed-query"):
            embedding = await self._embedding_model.embed_query(QUERY)

        results = None
        async with timed("search"):
            results = await self._db.vector_search(embedding)

        for result in results:
            print(f"Similarity Score: {result['SimilarityScore']}")
            print(f"Title: {result['title']}")
            print(f"Content: {result['content']}")
            print(f"Category: {result['category']}\n")
