from azure.cosmos import CosmosClient, PartitionKey, exceptions

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path":"/contentVector",
            "dataType":"float32",
            "distanceFunction":"cosine",
            "dimensions":1024
        }
    ]
}

indexing_policy = {
    "includedPaths": [
        {
            "path": "/*"
        }
    ],
    "excludedPaths": [
        {
            "path": "/\"_etag\"/?"
        },
        {
            "path": "/contentVector/*"
        }
    ],
    "vectorIndexes": [
        {"path": "/contentVector",
         "type": "quantizedFlat"
        }
    ]
}

class CosmosDB:
    def __init__(self, endpoint: str, key: str, database: str, container: str) -> None:
        self._client = CosmosClient(url=endpoint, credential=key)
        self._database = self._client.create_database_if_not_exists(id=database)
        self._key_path = PartitionKey(path="/categoryId")
        try:    
            self._container = self._database.create_container_if_not_exists(
                            id=container,
                            partition_key=PartitionKey(path='/id', kind='Hash'),
                            indexing_policy=indexing_policy,
                            vector_embedding_policy=vector_embedding_policy)

            print('Container with id \'{0}\' created'.format(id))

        except exceptions.CosmosHttpResponseError:
            raise

    async def index_vectors(self, data: list) -> None:
        """Add vectors to the vector store.

        :param vectors: List of vectors to add.
        """
        for item in data:
            self._container.upsert_item(item)

    async def vector_search(self, embedding, num_results=2):
        results = self._container.query_items(
                query='SELECT TOP @num_results c.content, VectorDistance(c.contentVector,@embedding) AS SimilarityScore  FROM c ORDER BY VectorDistance(c.contentVector,@embedding)',
                parameters=[
                    {"name": "@embedding", "value": embedding}, 
                    {"name": "@num_results", "value": num_results} 
                ],
                enable_cross_partition_query=True)
        return results
