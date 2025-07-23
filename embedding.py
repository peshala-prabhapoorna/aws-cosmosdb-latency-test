from typing import List
from langchain_community.embeddings import FastEmbedEmbeddings

class BAAIEmbeddingModel:
    """
    Instantiates the model once and reuses it. Handles errors and avoids blocking the event loop.
    Uses a dedicated ThreadPoolExecutor for embedding operations.
    """

    def __init__(self, dimension_count: int, embedding_type, metric, supported_languages: List[str] | None):
        self._dimension_count = dimension_count
        self._embedding_type = embedding_type
        self._metric = metric
        self._supported_languages = supported_languages

        try:
            self._model = FastEmbedEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        except Exception as e:
            raise Exception(f"Failed to load BAAI embedding model: {e}")

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single query string asynchronously using encode_query.
        This is optimized for query encoding for retrieval tasks.
        """
        try:
            embedding = await self._model.aembed_query(text)
            return embedding
        except Exception as e:
            raise Exception(f"Error during query embedding: {e}")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of input strings asynchronously using encode.
        This is suitable for encoding documents/passages or a batch of queries.
        """
        try:
            embeddings = await self._model.aembed_documents(texts)
            return embeddings
        except Exception as e:
            raise Exception(f"Error during batch embedding: {e}")
