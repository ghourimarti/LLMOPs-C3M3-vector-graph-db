from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct ,Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# Initialize the client
print("<=================================================>")
print("Initializing Qdrant client...")
print("<=================================================>")

client = QdrantClient(url="http://localhost:6333")
qdrant = QdrantClient(url="http://localhost:6333", prefer_grpc=False)


i0 = 1
if i0 == 1:
    # Create a collection
    print("<=================================================>")
    print("Creating collection...")
    print("<=================================================>")
    # create collection if it does not exist
    client.create_collection(
        collection_name="test_collection",
        vectors_config=VectorParams(size=4, distance=Distance.DOT),
    )


i1 = 1
if i1 == 1:
# Add vectors 
    print("<=================================================>")
    print("Adding vectors...")
    print("<=================================================>")
    operation_info = client.upsert(
        collection_name="test_collection",
        wait=True,
        points=[
            PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
            PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
            PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
            PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
            PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
            PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
        ],
    )

    print("operation information: " , operation_info)

i2 = 1
if i2 == 1:
    # Run a search query
    print("<=================================================>")
    print("Running search query...")
    print("<=================================================>")
    search_result = client.query_points(
        collection_name="test_collection",
        query=[0.2, 0.1, 0.9, 0.7],
        with_payload=False,
        limit=3
    ).points

    print(search_result)


i3 = 0
if i3 == 1:
    # Add a filter to the search query
    print("<=================================================>")
    print("Running search query with filter...")
    print("<=================================================>")
    search_result = client.query_points(
        collection_name="test_collection",
        query=[0.2, 0.1, 0.9, 0.7],
        query_filter=Filter(
            must=[FieldCondition(key="city", match=MatchValue(value="London"))]
        ),
        with_payload=True,
        limit=3,
    ).points

    print(search_result)


print("<=================================================>")
print("<=================================================>")
print("<=================================================>")
print("<=================================================>")
print("<=================================================>")
# Create a collection and upload records
# Initialize Qdrant client
print("<=================================================>")
print("Initializing Qdrant client...")
print("<=================================================>")


documents = [
    {"title": "Deep Learning with Python", "description": "A hands-on guide to deep learning using Python and Keras."},
    {"title": "Artificial Intelligence: A Modern Approach", "description": "Comprehensive book on AI covering search algorithms, logic, and machine learning."},
    {"title": "Python Crash Course", "description": "A beginner-friendly book for learning Python programming."},
    {"title": "Hands-On Machine Learning", "description": "Practical book with Scikit-Learn, Keras, and TensorFlow for building ML systems."},
    {"title": "The Hundred-Page Machine Learning Book", "description": "Concise and clear guide to the fundamentals of machine learning."}
]

# Initialize Qdrant client
print("<=================================================>")
print("Initializing Qdrant client...")
print("<=================================================>")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
# Create collection
qdrant.create_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE
    )
)

# Upload records using upload_points
print("<=================================================>")
print("Uploading records...")
print("<=================================================>")
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
print("<=================================================>")
print("Searching for similar books...")
print("<=================================================>")
hits = qdrant.query_points(
    collection_name="my_books",
    query=encoder.encode("A book about AI").tolist(),
    limit=3,
    with_payload=True
)

# Print results
print("<=================================================>")
print("Search results:")
print("<=================================================>")
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
# py -m venv .qdrant
# source .qdrant/Scripts/activate

# docker pull qdrant/qdrant
# docker run -p 6333:6333 -p 6334:6334 \
#     -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
#     qdrant/qdrant:latest 

# https://qdrant.tech/documentation/quickstart/