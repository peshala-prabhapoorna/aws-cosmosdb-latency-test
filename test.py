import contextlib
import csv
from datetime import datetime
from pathlib import Path
import time
import os
import unittest
import uuid

from dotenv import load_dotenv

from cosmosdb import CosmosDB
from embedding import BAAIEmbeddingModel
from pdf import DocumentLoaderPDF 

load_dotenv()

PDF_DOC = "arimac-rag-doc.pdf"
QUERY = "Who is the founder of Arimac Digital?"


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
    await write_to_csv([operation, round(duration, 4)])


class TestCosmosDB(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._db = CosmosDB(
            endpoint=os.environ.get("ENDPOINT"),
            key=os.environ.get("KEY"),
            database="test-db",
            container="test-container"
        )
        self._embedding_model = BAAIEmbeddingModel(
            dimension_count=1024,
            embedding_type="dense",
            metric="cosine",
            supported_languages=["en"]
        )

    async def test_indexing(self):
        pdf_loader = DocumentLoaderPDF(Path(PDF_DOC))
        await pdf_loader.initialize()
        pages = list(pdf_loader.get_pages_itr())
        self.assertGreater(len(pages), 0, "pdf document should have at least one page")
        await pdf_loader.deinitialize()

        texts = [page.content for page in pages if page.content.strip()]
        self.assertGreater(len(texts), 0, "Should have text content to embed from pdf file")
        
        embeddings = None
        async with timed("embed-doc"):
            embeddings = await self._embedding_model.embed_documents(texts)

        uuid_str = str(uuid.uuid4())
        data = [{"id": uuid_str, "content": f"content-{uuid_str[:4]}", "contentVector": embedding} for embedding in embeddings]

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
            print("result --- \n", result)
