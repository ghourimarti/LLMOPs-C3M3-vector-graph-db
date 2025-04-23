# explain the code bit by bit each and every thing
# and then run the code

# The code initializes a Qdrant client, creates a collection, uploads documents to it, and performs a search query.
# It uses the SentenceTransformer model to encode text data into vectors for storage and retrieval.
# The Qdrant client is initialized to work with an in-memory database for demonstration purposes.
# The code uses the QdrantClient and SentenceTransformer libraries to manage a vector database and encode text data.
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Initialize encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents
documents = [
    {"title": "Deep Learning with Python", "description": "A hands-on guide to deep learning using Python and Keras."},
    {"title": "Artificial Intelligence: A Modern Approach", "description": "Comprehensive book on AI covering search algorithms, logic, and machine learning."},
    {"title": "Python Crash Course", "description": "A beginner-friendly book for learning Python programming."},
    {"title": "Hands-On Machine Learning", "description": "Practical book with Scikit-Learn, Keras, and TensorFlow for building ML systems."},
    {"title": "The Hundred-Page Machine Learning Book", "description": "Concise and clear guide to the fundamentals of machine learning."}
]

# Initialize Qdrant client
# Using in-memory Qdrant for demonstration purposes
# In a real-world scenario, you would connect to a Qdrant server instance.
# For example, use QdrantClient("http://localhost:6333") for a local server.
# or QdrantClient("https://your-qdrant-instance.com") for a remote server.
# The ":memory:" argument creates an in-memory Qdrant instance for testing purposes.
# In production, you would typically connect to a persistent Qdrant instance.
qdrant = QdrantClient(":memory:")

# Check and create collection
# The collection is named "my_books" and is used to store the encoded vectors and their associated metadata.
# The collection is created with a vector size matching the encoder's output dimension and using cosine distance for similarity search.
# The collection is deleted if it already exists to ensure a clean state.
# This is useful for testing or when you want to start fresh without any previous data.
if qdrant.collection_exists("my_books"):
    qdrant.delete_collection("my_books")

# Create collection
qdrant.create_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE
    )
)

# Upload records using upload_points
qdrant.upload_points(
    collection_name="my_books",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["description"]).tolist(),
            payload=doc
        ) for idx, doc in enumerate(documents)
    ]
)

# Search using query_points
hits = qdrant.query_points(
    collection_name="my_books",
    query=encoder.encode("A book about AI").tolist(),
    limit=3,
    with_payload=True
)

# Print results
print("<------------------------------------->")
for hit in hits:
    print(hit)
    print("<----------------->")
    
hits = qdrant.search(  
    collection_name="my_books",  
    query_vector=encoder.encode("alien invasion").tolist(),  
    query_filter=models.Filter(  
       must=[  
           models.FieldCondition(  
               key="year",  
               range=models.Range(  
                  gte=2000  
               )  
           )  
       ]  
    ),  
    limit=1  
    )


# Print results
print("<------------------------------------->")
for hit in hits:
    print(hit)
    print("<----------------->")

    
# ('points', [ScoredPoint(id=1, version=0, score=0.6175682962297847, payload={'title': 'Artificial Intelligence: A Modern Approach', 'description': 'Comprehensive book on AI covering search algorithms, logic, and machine learning.'}, vector=None, shard_key=None, order_value=None), 
#             ScoredPoint(id=3, version=0, score=0.5334116741802924, payload={'title': 'Hands-On Machine Learning', 'description': 'Practical book with Scikit-Learn, Keras, and TensorFlow for building ML systems.'}, vector=None, shard_key=None, order_value=None), 
#             ScoredPoint(id=4, version=0, score=0.40924436814145126, payload={'title': 'The Hundred-Page Machine Learning Book', 'description': 'Concise and clear guide to the fundamentals of machine learning.'}, vector=None, shard_key=None, order_value=None)])