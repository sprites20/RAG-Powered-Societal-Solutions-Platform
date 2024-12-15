import fitz  # PyMuPDF
import numpy as np
import os
import torch
import pickle
import json
import time
from sklearn.cluster import KMeans

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

"""
model = AutoModelForSequenceClassification.from_pretrained(
  "togethercomputer/m2-bert-80M-8k-retrieval",
  trust_remote_code=True
)
max_seq_length = 8192
tokenizer = AutoTokenizer.from_pretrained(
  "bert-base-uncased",
  model_max_length=max_seq_length
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
model = model.to(device)
"""

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
model = AutoModel.from_pretrained('BAAI/bge-m3')
model.eval()
max_seq_length = 8192

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
model = model.to(device)


print(device)

# Initialize document ID counter
doc_id_counter = 1

# File path for persistent storage
doc_id_file_path = 'last_doc_id.pkl'

def extract_text_and_images_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ''
    images = []
    
    for page in doc:
        text += page.get_text()
        """
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append((f"image{page.number + 1}_{img_index + 1}.png", image_bytes))
        """
    return text

def chunk_tokens(tokens, max_length, overlap_size):
    chunks = []
    chunk_indices = []
    i = 0
    while i < len(tokens):
        end_idx = min(i + max_length, len(tokens))
        chunk = tokens[i:end_idx]
        chunks.append(chunk)
        chunk_indices.append((i, end_idx))
        i += max_length - overlap_size
    return chunks, chunk_indices

def save_document_to_binary(doc_id, document_data):
    filename = f"{doc_id}.npy"
    file_path = os.path.join('documents4', filename)
    
    # Prepare data to save (start_id, end_id, embedding)
    chunk_data = []
    embedding_size = None  # To determine the embedding size

    for chunk in document_data["chunks"]:
        start_id = chunk["start_id"]
        end_id = chunk["end_id"]
        embedding = chunk["embedding"]

        # Check embedding size
        if embedding_size is None:
            embedding_size = len(embedding)  # Set the embedding size on the first chunk
        elif len(embedding) != embedding_size:
            raise ValueError("All embeddings must have the same size.")

        chunk_data.append((start_id, end_id, embedding))
    
    # Create a structured NumPy array in a memmap format
    dtype = [('start_id', np.int32), ('end_id', np.int32), ('embedding', np.float32, (embedding_size,))]
    # Create a memmap array with 'w+' mode to create a new file
    structured_array = np.memmap(file_path, dtype=dtype, mode='w+', shape=(len(chunk_data),))

    # Populate the memmap array
    for i, (start_id, end_id, embedding) in enumerate(chunk_data):
        structured_array[i]['start_id'] = start_id
        structured_array[i]['end_id'] = end_id
        structured_array[i]['embedding'] = embedding

    # Flush changes to disk
    structured_array.flush()

    print(f"Document {doc_id} saved as {file_path}")

def load_document_chunks(doc_id):
    filename = f"documents4/{doc_id}.npy"
    file_path = filename
    print(file_path)
    
    try:
        # Use mmap_mode='r' for read-only access
        structured_array = np.memmap(file_path, dtype=[('start_id', np.int32), ('end_id', np.int32), ('embedding', np.float32, (1024,))], mode='r')
        
        # Convert structured array to a list of dictionaries if needed
        document_chunks = [{'start_id': entry['start_id'], 
                            'end_id': entry['end_id'], 
                            'embedding': entry['embedding']} for entry in structured_array]

        return document_chunks  # Or return structured_array directly if you prefer
    except FileNotFoundError:
        print(f"Document {filename} not found.")
        return None
    except ValueError as e:
        print(f"Error loading {filename}: {e}")
        return None
        
def create_clusters_from_embeddings(document_data, current_doc_id, max_chunks_per_cluster=1000):
    """Create clusters of chunk indices from the document data loaded from .npy."""
    #clusters = []
    current_cluster = []
    cluster_id = 0  # Initialize cluster ID

    # Generate clusters based on chunk indices
    for index in range(len(document_data)):
        chunk_index = index  # Use the index in the loaded data as chunk_index
        chunk_tuple = (current_doc_id, chunk_index)  # Use the current document ID

        # Append the chunk index to the current cluster
        current_cluster.append(chunk_tuple)
        #Now what we need is to append the current cluster to the file
        # If the current cluster reaches the max size, finalize it
        if len(current_cluster) >= max_chunks_per_cluster:
            cluster_entry = {
                "id": cluster_id,
                #"doc_id": current_doc_id,  # Include document ID in the cluster entry
                "chunks": current_cluster,  # Store only the chunk indices
                "centroid": None,  # Placeholder for centroid if needed later
                "children": []
            }
            #clusters.append(cluster_entry)
            cluster_id += 1  # Increment cluster ID
            current_cluster = []  # Reset for a new cluster

    # Add the last cluster if it has remaining chunks
    if current_cluster:
        cluster_entry = {
            "id": cluster_id,
            #"doc_id": current_doc_id,  # Include document ID in the cluster entry
            "chunks": current_cluster,
            "centroid": None,
            "children": []
        }
        #clusters.append(cluster_entry)

    return current_cluster
class RetrievalManager:
    def __init__(self):
        self.cluster_embeddings = {}
        
class MemmapManager:
    def __init__(self):
        self.memmap_cache = {}

    def get_memmap(self, doc_id):
        """Retrieve the memmap for a given document ID, loading it if necessary."""
        file_path = f'documents4/{doc_id}.npy'
        if doc_id not in self.memmap_cache:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
            # Load the memmap for the document and cache it
            self.memmap_cache[doc_id] = np.memmap(file_path, dtype=[('start_id', 'i4'), ('end_id', 'i4'), ('embedding', 'f4', (1024,))], mode='r')
        return self.memmap_cache[doc_id]

    def process_chunks(self, chunk_indices):
        """Process the chunks specified by a list of (doc_id, chunk_index) tuples."""
        results = []
        start_time = time.time()
        for doc_id, chunk_index in chunk_indices:
            try:
                memmap = self.get_memmap(doc_id)
                chunk_data = memmap[chunk_index]
                results.append((doc_id, chunk_index, chunk_data))
            except:
                pass
        end_time = time.time()
        print(f"Time taken to process chunks: {end_time - start_time:.10f} seconds")
        return results

    def compute_centroid(self, embeddings):
        """Compute the centroid of embeddings."""
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)
def create_cluster(memmap, max_chunks_per_cluster, cluster_index):
    # Initialize the first cluster's fields (e.g., id, chunks, centroid)
    memmap['id'] = cluster_index  # Example: set ID to cluster_index + 1
    memmap['centroid'] = np.zeros(1024)  # Initialize centroid as a zero vector
    memmap['chunks'] = np.zeros((max_chunks_per_cluster,), dtype=[('doc_id', 'i4'), ('chunk_index', 'i4')])
    memmap['children'] = np.zeros(10)
    
def create_single_cluster_entry(document_data, current_doc_id, max_chunks_per_cluster=1000, n_clusters=10):
    """Assign chunks to nearest existing cluster centroids."""
    
    # Define the structure for the cluster data to be saved in the memory-mapped file
    dtype = [
        ('id', 'i4'),
        ('chunks', [('doc_id', 'i4'), ('chunk_index', 'i4')], max_chunks_per_cluster),
        ('centroid', 'f4', (1024,)),
        ('children', 'f4', (10,)),
    ]
    
    #First we search through nearest cluster
    #If no cluster exists we create 1 cluster
    # Function to create a new cluster

    # Load existing clusters (if any)
    existing_clusters = None
    cluster_file = f'./clusters/cluster_1.memmap'
    # Ensure the parent directory exists
    directory = os.path.dirname(cluster_file)  # Extract directory path
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the parent directory if it doesn't exist
    if not os.path.exists(cluster_file):    
        memmap = np.memmap(cluster_file, dtype=dtype, mode='w+', shape=(1,))
        
    if os.path.exists(cluster_file):
        existing_clusters = np.memmap(cluster_file, dtype=dtype, mode='r+')
    
    # If no clusters exist, create the first cluster
    if not existing_clusters:
        # Create a new memory-mapped file with space for 1 cluster
        memmap = np.memmap(cluster_file, dtype=dtype, mode='r+')
        
        # Create the first cluster
        create_cluster(memmap, max_chunks_per_cluster, 1)

        # Flush changes to ensure they're saved
        existing_clusters.flush()
        print("First cluster created and initialized.")
    else:
        print("Clusters already exist.")
        
    
    # Add chunk indices to the current list
    chunk_indices = [(current_doc_id, index) for index in range(len(document_data))]

    # Process chunks to retrieve embeddings
    manager = MemmapManager()
    results = manager.process_chunks(chunk_indices)
    embeddings = [data['embedding'] for _, _, data in results]
    
    #Now we have embeddings
    cluster_memmap = None
    #print(np.memmap(cluster_file, dtype=dtype, mode='r+'))
    # Assign each chunk to the nearest existing cluster centroid
    updated_cluster_embeddings = {}
    chunk_embeddings = {}
    for index, embedding in enumerate(embeddings):
        distances = []
        for cluster in existing_clusters:
            centroid = cluster['centroid'][0]
            distance = np.linalg.norm(embedding - centroid)
            distances.append(distance)
        closest_cluster_index = np.argmin(distances)
        
        # Get the closest cluster file and add chunk there
        cluster_file = f'clusters/cluster_{closest_cluster_index + 1}.memmap'
        cluster_memmap = np.memmap(cluster_file, dtype=dtype, mode='r+')
        if not closest_cluster_index in updated_cluster_embeddings:
            updated_cluster_embeddings[closest_cluster_index] = []
        #We just store the embeddings of those chunks in that cluster
        #We store doc ids existing in that chunk
        if not closest_cluster_index in chunk_embeddings:
            chunk_embeddings[closest_cluster_index] = {}
        
        # Find the first available slot in 'chunks' field
        for i in range(max_chunks_per_cluster):
            chunk_embeddings[closest_cluster_index][cluster_memmap['chunks'][0][i]['doc_id']] = None
        
        for i in range(max_chunks_per_cluster):
            if cluster_memmap['chunks'][0][i]['doc_id'] == 0:  # Assuming 0 means uninitialized
                cluster_memmap['chunks'][0][i] = chunk_indices[index]
                break
        chunk_embeddings[closest_cluster_index][current_doc_id] = None
    
    for closest_cluster_index in chunk_embeddings:   
        doc_dtype = [('start_id', np.int32), ('end_id', np.int32), ('embedding', np.float32, (1024,))]
        document_memmaps = {}
        cluster_file = f'clusters/cluster_{closest_cluster_index + 1}.memmap'
        cluster_memmap = np.memmap(cluster_file, dtype=dtype, mode='r+')
        del chunk_embeddings[closest_cluster_index][0]
        #Iterate through the chunk embeddings, get memmap of the cluster
        for doc_id in chunk_embeddings[closest_cluster_index]:
            mem = f'./documents4/{doc_id}.npy'
            #print(doc_id)
            document_memmaps[doc_id] = np.memmap(mem, dtype=doc_dtype, mode='r+')
            
        for i in cluster_memmap['chunks'][0]:
            if i['doc_id'] != 0:
                #print(i['doc_id'], i['chunk_index'])
                q = document_memmaps[i['doc_id']][i['chunk_index']]["embedding"]
                updated_cluster_embeddings[closest_cluster_index].append(q)
                #print(q)
                #print(q)
    for i in updated_cluster_embeddings:
        #print(updated_cluster_embeddings[i])
        #print(i)
        centroid = np.mean(np.array(updated_cluster_embeddings[i]), axis=0)
        #print(centroid)
        #We need a folder for each client data which includes transaction history.
        existing_clusters["centroid"][0] = centroid
        #We store clusters that need to be indexed in a pickle
        """
        with open(doc_id_file_path, 'wb') as f:
            pickle.dump(doc_id_counter, f)
        
        with open(doc_id_file_path, 'rb') as f:
            doc_id_counter = pickle.load(f)
        """
        
        #Update the centroid of that cluster
        #First we take all embeddings in that cluster and then we compute
        
        #By that we need to get all documents in that cluster
        #We load embeddings in chunk_embeddings only specific ones
        
        #And compute for centroids
        
    #Now we need to update the centroids of the clusters modified
    #We log the ids of the cluster for recomputation
    #Then we do the computations of the things
    #We need to make sure that the centroid updates else it will have issues.
    #To do this we have to make it fault tolerant, we simply store which transaction is incomplete in some file. We only need to store the latest one per client.
    
    #If suddenly interrupted, like we didnt updated centroids damn. We need to like
    #print(cluster_memmap)
    return "Chunks assigned to the nearest clusters successfully."


def load_and_cluster_embeddings_by_id(doc_id):
    """Load document chunks by document ID and create clusters from their indices."""
    document_data = load_document_chunks(doc_id)
    if document_data is not None:
        #clusters = create_clusters_from_embeddings(document_data, doc_id)  # Pass the doc_id here
        # Create a single cluster entry and save to file
        single_cluster_entry = create_single_cluster_entry(document_data, doc_id, max_chunks_per_cluster=1000)
        print(single_cluster_entry)
        #print_clusters(clusters)  # Print clusters
        return single_cluster_entry
    else:
        print(f"Document ID '{doc_id}' not found.")
        return None

def print_clusters(clusters):
    """Print the clusters and their contents."""
    for cluster in clusters:
        print(f"Cluster ID: {cluster['id']}, Number of Chunks: {len(cluster['chunks'])}")
        for chunk in cluster['chunks']:
            print(f"\tDoc ID: {chunk[0]}, Chunk Index: {chunk[1]}")

def process_pdf(pdf_file):
    # Ensure the directory for storing documents exists
    os.makedirs('documents4', exist_ok=True)

    # Extract text and images from the PDF
    text, images = extract_text_and_images_from_pdf(pdf_file)

    # Tokenize the entire document
    tokens = tokenizer.tokenize(text)
    token_offsets = tokenizer(text, return_offsets_mapping=True)['offset_mapping']

    # Define maximum token length and overlap size
    max_length = 4000  # Adjust as needed
    overlap_size = 20

    # Chunk the tokens
    chunks, chunk_indices = chunk_tokens(tokens, max_length, overlap_size)

    # Get character indices for chunks
    char_indices = [(token_offsets[start_idx][0], token_offsets[end_idx - 1][1]) for start_idx, end_idx in chunk_indices]

    # Convert token chunks back to strings
    chunk_strings = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
    
    # Prepare for storing embeddings
    embedding_dim = 1024  # Update this according to model's output dimensions
    document_data = {"chunks": []}

    # Process each chunk to generate embeddings
    for i, (start_idx, end_idx) in enumerate(chunk_indices):
        start_time = time.time()
        print(f"Chunk {i + 1}: Start char index: {start_idx}, End char index: {end_idx}")
        #print(chunk_strings[i])
        
        # Tokenize sentences and move to model's device
        encoded_input = tokenizer(chunk_strings[i], padding=True, truncation=True, return_tensors='pt').to(model.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)#['sentence_embedding']
            #print(model_output['sentence_embedding'].shape)
            # Use CLS pooling (assuming CLS token is the first token)
            sentence_embeddings = model_output[0][:,0]  # Shape should be (1, embedding_dim)
            
            # Check the shape and move to CPU for numpy conversion
            if sentence_embeddings.shape[0] == 1:
                sentence_embeddings = sentence_embeddings.squeeze(0)  # Remove batch dimension if needed

            print(f"Embeddings shape: {sentence_embeddings.shape}")
            
            # Move to CPU and scale the embedding
            sentence_embeddings_scaled = sentence_embeddings.cpu().numpy() / 20.0
            
            # Ensure embedding is of size `embedding_dim` (pad or truncate if necessary)
            if sentence_embeddings_scaled.shape[0] != embedding_dim:
                # Handle size mismatch: pad if smaller
                if sentence_embeddings_scaled.shape[0] < embedding_dim:
                    padded_embedding = np.pad(sentence_embeddings_scaled, (0, embedding_dim - sentence_embeddings_scaled.shape[0]), 'constant')
                else:
                    # Truncate if larger
                    padded_embedding = sentence_embeddings_scaled[:embedding_dim]
            else:
                padded_embedding = sentence_embeddings_scaled

            # Store chunk data
            document_data["chunks"].append({
                "start_id": start_idx,
                "end_id": end_idx,
                "embedding": padded_embedding,
            })
            
        print(f"Time taken to embed last chunk: {time.time() - start_time:.8f} seconds")
    # Save the document embeddings to binary
    global doc_id_counter
    save_document_to_binary(doc_id=doc_id_counter, document_data=document_data)
    doc_id_counter += 1  # Increment the document ID counter
    
def save_last_document_id():
    with open(doc_id_file_path, 'wb') as f:
        pickle.dump(doc_id_counter, f)

def load_last_document_id():
    global doc_id_counter
    if os.path.exists(doc_id_file_path):
        with open(doc_id_file_path, 'rb') as f:
            doc_id_counter = pickle.load(f)

def save_clusters_to_file(clusters, filename):
    """Save clusters to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(clusters, f, indent=4)
        
def query_cluster(cluster_path=None, query = None):
    start_time = time.time()
    max_chunks_per_cluster=1000
    embedding_dim = 1024
    # Define the structure for the cluster data to be saved in the memory-mapped file
    dtype = [
        ('id', 'i4'),
        ('chunks', [('doc_id', 'i4'), ('chunk_index', 'i4')], max_chunks_per_cluster),
        ('centroid', 'f4', (1024,)),
        ('children', 'f4', (10,)),
    ]
    
    #First we search through nearest cluster
    #If no cluster exists we create 1 cluster
    # Function to create a new cluster

    # Load existing clusters (if any)
    existing_clusters = None
    cluster_file = f'./clusters/cluster_1.memmap'
    if os.path.exists(cluster_file):
        existing_clusters = np.memmap(cluster_file, dtype=dtype, mode='r+')
    #Now we have embeddings
    cluster_memmap = None
    #print(np.memmap(cluster_file, dtype=dtype, mode='r+'))
    # Assign each chunk to the nearest existing cluster centroid
    updated_cluster_embeddings = {}
    chunk_embeddings = {}
    closest_cluster_index = 0
    # Get the closest cluster file and add chunk there
    cluster_memmap = np.memmap(cluster_file, dtype=dtype, mode='r+')
    if not closest_cluster_index in updated_cluster_embeddings:
        updated_cluster_embeddings[closest_cluster_index] = []
    #We just store the embeddings of those chunks in that cluster
    #We store doc ids existing in that chunk
    if not closest_cluster_index in chunk_embeddings:
        chunk_embeddings[closest_cluster_index] = {}
    
    # Find the first available slot in 'chunks' field
    for i in range(max_chunks_per_cluster):
        chunk_embeddings[closest_cluster_index][cluster_memmap['chunks'][0][i]['doc_id']] = None
    print(f"Time taken to chunk things: {time.time() - start_time:.8f} seconds")
    
    
    start_time = time.time()
    #chunk_embeddings[closest_cluster_index][current_doc_id] = None
    # Tokenize sentences and move to model's device
    encoded_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt').to(model.device)
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)#['sentence_embedding']
        #print(model_output['sentence_embedding'].shape)
        # Use CLS pooling (assuming CLS token is the first token)
        sentence_embeddings = model_output[0][:,0]  # Shape should be (1, embedding_dim)
        
        # Check the shape and move to CPU for numpy conversion
        if sentence_embeddings.shape[0] == 1:
            sentence_embeddings = sentence_embeddings.squeeze(0)  # Remove batch dimension if needed

        print(f"Embeddings shape: {sentence_embeddings.shape}")
        
        # Move to CPU and scale the embedding
        sentence_embeddings_scaled = sentence_embeddings.cpu().numpy() / 20.0
        
        # Ensure embedding is of size `embedding_dim` (pad or truncate if necessary)
        if sentence_embeddings_scaled.shape[0] != embedding_dim:
            # Handle size mismatch: pad if smaller
            if sentence_embeddings_scaled.shape[0] < embedding_dim:
                padded_embedding = np.pad(sentence_embeddings_scaled, (0, embedding_dim - sentence_embeddings_scaled.shape[0]), 'constant')
            else:
                # Truncate if larger
                padded_embedding = sentence_embeddings_scaled[:embedding_dim]
        else:
            padded_embedding = sentence_embeddings_scaled
    
    query_embedding = padded_embedding
    print(f"Time taken to embed query: {time.time() - start_time:.8f} seconds")
    for closest_cluster_index in chunk_embeddings:
        start_time = time.time()
        doc_dtype = [('start_id', np.int32), ('end_id', np.int32), ('embedding', np.float32, (1024,))]
        document_memmaps = {}
        cluster_memmap = np.memmap(cluster_file, dtype=dtype, mode='r+')
        del chunk_embeddings[closest_cluster_index][0]
        #Iterate through the chunk embeddings, get memmap of the cluster
        for doc_id in chunk_embeddings[closest_cluster_index]:
            mem = f'./documents4/{doc_id}.npy'
            #print(doc_id)
            document_memmaps[doc_id] = np.memmap(mem, dtype=doc_dtype, mode='r+')
            
        for i in cluster_memmap['chunks'][0]:
            if i['doc_id'] != 0:
                #print(i['doc_id'], i['chunk_index'])
                chunk_data = document_memmaps[i['doc_id']][i['chunk_index']]
                q = {
                    "embeddings" : chunk_data["embedding"],
                    "start_id" : chunk_data["start_id"],
                    "end_id" : chunk_data["end_id"],
                    "doc_id" : i['doc_id'],
                    "chunk_index" : i['chunk_index'],
                }
                updated_cluster_embeddings[closest_cluster_index].append(q)
                #print(q)
                #print(q)
        
        
        # List to store distances along with doc_id and chunk_index
        distances = []
        sometime = time.time() - start_time
        print(f"Time taken to load embeddings: {sometime:.8f} seconds")
        # Iterate over the updated_cluster_embeddings
        somelen = len(updated_cluster_embeddings[0])
        print(f"Number of embeddings loaded: {somelen} at rate of {somelen / sometime} embeddings or {somelen / sometime / 6} documents per second")
        start_time = time.time()
        
        for embedding in updated_cluster_embeddings[0]:
            # Extract the embedding array
            embedding_array = np.array(embedding["embeddings"])
            
            # Normalize both embeddings to calculate cosine similarity
            query_norm = query_embedding #np.linalg.norm(query_embedding)
            embedding_norm = embedding_array #np.linalg.norm(embedding_array)
            
            similarity = np.dot(query_embedding, embedding_array)
            
            # Convert similarity to a distance (1 - similarity)
            distance = 1 - similarity
            #distance *= -1
            # Extract additional metadata
            doc_id = embedding['doc_id']
            chunk_index = embedding['chunk_index']
            start_index = embedding['start_id']
            end_index = embedding['end_id']
            
            if distance < 0.35:
                # Append distance and identifiers to the list
                distances.append((distance, doc_id, chunk_index, start_index, end_index))
        print(f"Time taken to find similar ones: {time.time() - start_time:.8f} seconds")
        # Sort by distance (smaller values indicate higher similarity)
        distances.sort(key=lambda x: x[0], reverse = True)
        distances = distances[-3:]
        
        # Output sorted distances and retrieve text
        path = "D:/pyfiles/Spirit-AGI-main/temp5/AI_Papers_2/"
        
        # Initialize a list to accumulate text chunks with metadata
        accumulated_texts = []
        
        doc_texts = {}
        solutions = []
        # Iterate over the distances, extracting the relevant portions of text and storing metadata
        for distance, doc_id, chunk_index, start_index, end_index in distances:
            #print(f"doc_id: {doc_id}, chunk_index: {chunk_index}, distance (1-similarity): {distance}")
            if distance > -0.35:
                # Load the document (assuming you have a function for this)
                doc = load_doc_mapping_by_value(doc_id)
                print(doc)
                text = ""
                if doc not in doc_texts:
                    # Extract the text and images from the PDF
                    text = extract_text_and_images_from_pdf(path + doc)
                    doc_texts[doc] = text
                # Extract the relevant portion of the text
                relevant_text = doc_texts[doc][start_index:end_index]
                #print(text)
                result = f"doc: {doc}\ndoc_id: {doc_id}, chunk_index: {chunk_index}, distance (1-similarity): {distance}\n{relevant_text}"
                print(result)
                solutions.append(result)
        return solutions
        """
            if doc not in doc_texts:
                # Extract the text and images from the PDF
                text = extract_text_and_images_from_pdf(path + doc)
                doc_texts[doc] = text
            # Extract the relevant portion of the text
            relevant_text = doc_texts[doc][start_index:end_index]
            
            #print(relevant_text)
            # Accumulate the relevant text with doc_id and chunk_index
            accumulated_texts.append({
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "text": relevant_text
            })

        # After accumulating, tokenize the texts
        tokenized_texts = [chunk['text'].split() for chunk in accumulated_texts]

        # Initialize BM25 with all the tokenized chunks as documents
        bm25 = BM25Okapi(tokenized_texts)

        # Define your search query
        tokenized_query = query.split()

        # Get BM25 scores for the query against all the accumulated texts
        scores = bm25.get_scores(tokenized_query)

        # Combine BM25 scores with metadata (doc_id, chunk_index)
        scored_chunks = []
        for idx, score in enumerate(scores):
            scored_chunks.append({
                "doc_id": accumulated_texts[idx]["doc_id"],
                "chunk_index": accumulated_texts[idx]["chunk_index"],
                "score": score
            })

        # Sort the results by BM25 score (higher score means more relevant)
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)

        # Print the sorted results
        for chunk in scored_chunks:
            if chunk['score'] > 1:
                print(f"doc_id: {chunk['doc_id']}, chunk_index: {chunk['chunk_index']}, BM25 score: {chunk['score']}")
                print(load_doc_mapping_by_value(chunk['doc_id']))
                doc = load_doc_mapping_by_value(chunk['doc_id'])
                print(doc.replace(".txt", ""))
                print(get_disease_number(doc.replace(".txt", "")))
        
        # Load the document (assuming you have a function for this)
        """
        """
        for i in accumulated_texts:
            if i['doc_id'] == scored_chunks[0]['doc_id'] and i['chunk_index'] == scored_chunks[0]['chunk_index']:
                print(i['text'])
        """
import shelve

# File names for bidirectional mapping
key_to_value_file = 'key_to_value.db'  # pdf_file -> doc_id
value_to_key_file = 'value_to_key.db'  # doc_id -> pdf_file


def load_doc_mapping_by_key(pdf_file):
    """Load a single document mapping using the pdf_file (key)."""
    try:
        with shelve.open(key_to_value_file, flag='r') as db:
            # Return the doc_id for the given pdf_file, or None if not found
            return db.get(str(pdf_file), None)
    except (FileNotFoundError, KeyError) as e:
        # If the file doesn't exist or the key is not found, handle it gracefully
        print(f"Error loading mapping for '{pdf_file}' by key: {e}")
        return None

def load_doc_mapping_by_value(doc_id):
    """Load a single document mapping using the doc_id (value)."""
    try:
        with shelve.open(value_to_key_file, flag='r') as db:
            # Return the pdf_file for the given doc_id, or None if not found
            return db.get(str(doc_id), None)
    except (FileNotFoundError, KeyError) as e:
        # If the file doesn't exist or the key is not found, handle it gracefully
        print(f"Error loading mapping for doc_id '{doc_id}' by value: {e}")
        return None

def save_doc_mapping(pdf_file, doc_id):
    doc_id -= 1
    """Save a single document mapping in both key-to-value and value-to-key databases."""
    try:
        # Key to value (pdf_file -> doc_id)
        with shelve.open(key_to_value_file, flag='c') as key_db:
            key_db[pdf_file] = doc_id  # Save or update the mapping
        
        # Value to key (doc_id -> pdf_file)
        with shelve.open(value_to_key_file, flag='c') as value_db:
            value_db[str(doc_id)] = pdf_file  # Save or update the reverse mapping
        
        print(f"Mapping for '{pdf_file}' -> {doc_id} saved successfully.")
        
    except PermissionError as e:
        print(f"Permission error while saving mapping: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving mapping: {e}")

def delete_doc_mapping_by_key(pdf_file):
    """Delete a specific document mapping using the pdf_file (key)."""
    try:
        # Key to value (pdf_file -> doc_id)
        with shelve.open(key_to_value_file, flag='c') as key_db:
            if pdf_file in key_db:
                doc_id = key_db.pop(pdf_file)  # Delete the mapping for the given file
                print(f"Deleted mapping for '{pdf_file}' -> {doc_id}.")
            else:
                print(f"No mapping found for '{pdf_file}' in key-to-value database.")
        
        # Value to key (doc_id -> pdf_file)
        with shelve.open(value_to_key_file, flag='c') as value_db:
            # If the key is deleted, ensure the reverse mapping is also deleted
            with shelve.open(key_to_value_file, flag='r') as key_db:
                doc_id = key_db.get(pdf_file, None)
                if doc_id and doc_id in value_db:
                    value_db.pop(doc_id)  # Delete the reverse mapping for the doc_id
                    print(f"Deleted mapping for doc_id {doc_id} -> '{pdf_file}' from value-to-key database.")
                else:
                    print(f"No reverse mapping found for doc_id {doc_id} in value-to-key database.")
    
    except KeyError as e:
        print(f"Error: The key '{pdf_file}' was not found in the database: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while deleting mapping: {e}")

def delete_doc_mapping_by_value(doc_id):
    """Delete a specific document mapping using the doc_id (value)."""
    try:
        # Value to key (doc_id -> pdf_file)
        with shelve.open(value_to_key_file, flag='c') as value_db:
            if doc_id in value_db:
                pdf_file = value_db.pop(doc_id)  # Delete the reverse mapping for the doc_id
                print(f"Deleted mapping for doc_id {doc_id} -> '{pdf_file}'.")
            else:
                print(f"No mapping found for doc_id {doc_id} in value-to-key database.")
        
        # Key to value (pdf_file -> doc_id)
        with shelve.open(key_to_value_file, flag='c') as key_db:
            # If the value is deleted, ensure the key mapping is also deleted
            with shelve.open(value_to_key_file, flag='r') as value_db:
                pdf_file = value_db.get(doc_id, None)
                if pdf_file and pdf_file in key_db:
                    key_db.pop(pdf_file)  # Delete the mapping for the given file
                    print(f"Deleted mapping for '{pdf_file}' -> {doc_id} from key-to-value database.")
                else:
                    print(f"No reverse mapping found for '{pdf_file}' in key-to-value database.")
    
    except KeyError as e:
        print(f"Error: The doc_id '{doc_id}' was not found in the database: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while deleting mapping: {e}")

def ensure_db_exists():
    global key_to_value_file, value_to_key_file
    """Ensure the shelve databases are created if they don't exist."""
    # Ensure the key-to-value database exists
    if not os.path.exists(key_to_value_file):
        with shelve.open(key_to_value_file, flag='c') as db:
            print(f"Created new database file: {key_to_value_file}")
    
    # Ensure the value-to-key database exists
    if not os.path.exists(value_to_key_file):
        print("Not exists", value_to_key_file)
        with shelve.open(value_to_key_file, flag='c') as db:
            print(f"Created new database file: {value_to_key_file}")

def check_key_exists(pdf_file):
    """Check if a pdf_file exists in the key-to-value mapping (key file)."""
    try:
        with shelve.open(key_to_value_file, flag='r') as db:
            return pdf_file in db  # Return True if the key exists, else False
    except (FileNotFoundError, KeyError) as e:
        print(f"Error checking existence of '{pdf_file}': {e}")
        return False

def check_value_exists(doc_id):
    """Check if a doc_id exists in the value-to-key mapping (value file)."""
    try:
        with shelve.open(value_to_key_file, flag='r') as db:
            return str(doc_id) in db  # Return True if the value exists, else False
    except (FileNotFoundError, KeyError) as e:
        print(f"Error checking existence of doc_id '{doc_id}': {e}")
        return False

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows requests from other origins

@app.route('/generate-solution', methods=['POST'])
def generate_solution():
    data = request.json
    query = data.get('query', '')
    category = data.get('category', '')
    
    temp_solutions = [
        f"Solution 1 for query: '{query}' in category: '{category}'.",
        f"Solution 2 for query: '{query}' in category: '{category}'.",
        f"Solution 3 for query: '{query}' in category: '{category}'."
    ]
    
    solutions = query_cluster(query=query) or temp_solutions
    # Example solutions (replace this with actual logic)

    return jsonify({"solutions": solutions})

     
def main():
    start_time = time.time()
    load_last_document_id()
    global doc_id_counter
    print(f"Starting with document ID: {doc_id_counter}")
    print(f"Time taken to load last document ID: {time.time() - start_time:.2f} seconds")
    
    #ensure_db_exists()
    #load_and_cluster_embeddings_by_id(1)
    
    print(load_doc_mapping_by_value(1))
    """
    for i in range(1,30):
        #load_and_cluster_embeddings_by_id(i)
        print(load_doc_mapping_by_value(i))
    """
    start_time = time.time()
    """
    query_cluster(query = "Particle-based representations of radiance fields such as 3D Gaussian Splat\
ting have found great success for reconstructing and re-rendering of complex\
 scenes. Most existing methods render particles via rasterization, projecting\
 them to screen space tiles for processing in a sorted order")
    """
    print(f"Time taken to retrieve ID: {time.time() - start_time:.8f} seconds")
    
    pdf_folder = "D:/pyfiles/Spirit-AGI-main/temp5/AI_Papers_2"
    skip_until = 0
    skip = 1
    
    app.run(debug=True)
    """
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            try:
                pdf_path = os.path.join(pdf_folder, pdf_file)
                print(f"Processing PDF: {pdf_path}")

                # Process the current PDF file
                start_time = time.time()
                if not check_key_exists(pdf_file):
                    process_pdf(pdf_path)  # Process and embed PDF content
                    # Save mapping for the processed file immediately
                    save_doc_mapping(pdf_file, doc_id_counter)
                    
                    # Increment the document ID after processing
                    #doc_id_counter += 1
                    save_last_document_id()
                    print(f"Time taken to process PDF: {time.time() - start_time:.2f} seconds")
                else:
                    print(f"Skipping: {pdf_file}, with value, {load_doc_mapping_by_key(pdf_file)}")
                
            except Exception as e:
                print(f"Error: {e}\nSkipping document")

    print(f"Document mappings saved. Total time taken: {time.time() - start_time:.2f} seconds")
    """
if __name__ == '__main__':
    main()