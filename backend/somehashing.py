import mmap
import os
import struct
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import os
from werkzeug.utils import secure_filename
import json
import signal
import time
import duckdb
from collections import defaultdict
from flask_cors import CORS
import fitz  # PyMuPDF
import traceback
import requests


GEMINI_API_KEY = 'AIzaSyB6VxiPZ9dtCbk0Ph15ooI4-9GL6AJmHRg'  # Replace with your Gemini API key
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

nltk.download('punkt')
nltk.download('stopwords')

class MMapChainedHashTable:
    TABLE_SIZE = 100_000  # number of buckets
    SLOT_SIZE = 8         # each slot stores 8-byte offset to chain head (0 = empty)
    KEY_SIZE = 32         # fixed length for keys (padded/truncated)
    NODE_HEADER_SIZE = KEY_SIZE + 4 + 4 + 8
    # Node layout: key(32) + count_of_items(4) + items_region_size(4) + next_node_offset(8)
    # Items follow immediately after header, variable length: count_of_items * 4-byte integers (example items)

    DATA_REGION_SIZE = DATA_REGION_SIZE = 500 * 1024 * 1024  # 2 GB
    FILE_SIZE = SLOT_SIZE * TABLE_SIZE + DATA_REGION_SIZE

    def __init__(self, filename="mmap_chain_hash.dat"):
        self.filename = filename
        self._init_file()
        self.f = open(self.filename, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), self.FILE_SIZE)
        self.data_start = self.SLOT_SIZE * self.TABLE_SIZE
        self.next_free_offset = self.data_start
        # In a real implementation, you'd want to persist and load next_free_offset or free list!

    def _init_file(self):
        if not os.path.exists(self.filename):
            print(f"Creating file {self.filename} size {self.FILE_SIZE}")
            with open(self.filename, "wb") as f:
                f.truncate(self.FILE_SIZE)

    def _hash(self, key):
        return sum(ord(c) for c in key) % self.TABLE_SIZE

    def _read_slot(self, index):
        offset = index * self.SLOT_SIZE
        return struct.unpack('Q', self.mm[offset:offset+self.SLOT_SIZE])[0]

    def _write_slot(self, index, offset_val):
        offset = index * self.SLOT_SIZE
        self.mm[offset:offset+self.SLOT_SIZE] = struct.pack('Q', offset_val)

    def _read_node(self, offset):
        header = self.mm[offset:offset+self.NODE_HEADER_SIZE]
        key_bytes = header[:self.KEY_SIZE]
        count = struct.unpack('I', header[self.KEY_SIZE:self.KEY_SIZE+4])[0]
        items_size = struct.unpack('I', header[self.KEY_SIZE+4:self.KEY_SIZE+8])[0]
        next_node_offset = struct.unpack('Q', header[self.KEY_SIZE+8:self.KEY_SIZE+16])[0]

        key = key_bytes.rstrip(b'\x00').decode('utf-8')
        items_offset = offset + self.NODE_HEADER_SIZE
        items_bytes = self.mm[items_offset:items_offset+items_size]
        # Assume items are 4-byte integers
        items = [struct.unpack('I', items_bytes[i:i+4])[0] for i in range(0, len(items_bytes), 4)]

        return {
            "key": key,
            "count": count,
            "items": items,
            "next": next_node_offset,
            "offset": offset,
            "size": self.NODE_HEADER_SIZE + items_size
        }

    def _write_node(self, offset, key, items, next_node_offset):
        key_bytes = key.encode('utf-8')[:self.KEY_SIZE]
        key_bytes = key_bytes.ljust(self.KEY_SIZE, b'\x00')
        count = len(items)
        items_size = count * 4
        header = key_bytes + struct.pack('I', count) + struct.pack('I', items_size) + struct.pack('Q', next_node_offset)
        items_bytes = b''.join(struct.pack('I', i) for i in items)
        self.mm[offset:offset+len(header)] = header
        self.mm[offset+len(header):offset+len(header)+items_size] = items_bytes

    def _alloc_node(self, size):
        if self.next_free_offset + size > self.FILE_SIZE:
            raise Exception("Out of mmap data space!")
        off = self.next_free_offset
        self.next_free_offset += size
        return off

    def get(self, key):
        bucket = self._hash(key)
        node_offset = self._read_slot(bucket)
        while node_offset != 0:
            node = self._read_node(node_offset)
            if node["key"] == key:
                return node["items"]
            node_offset = node["next"]
        return None

    def insert(self, key, items):
        bucket = self._hash(key)
        node_offset = self._read_slot(bucket)

        merged_items = set(items)
        prev_offset = 0
        chain_head = node_offset

        # Traverse the chain and collect items for this key
        while node_offset != 0:
            node = self._read_node(node_offset)
            if node["key"] == key:
                merged_items.update(node["items"])
            else:
                prev_offset = node_offset  # Track last non-matching node
            node_offset = node["next"]

        # Remove all nodes for this key by rebuilding the chain without them
        node_offset = chain_head
        new_chain_head = 0
        last_node_offset = 0

        while node_offset != 0:
            node = self._read_node(node_offset)
            next_offset = node["next"]

            if node["key"] != key:
                if new_chain_head == 0:
                    new_chain_head = node_offset
                else:
                    prev_node = self._read_node(last_node_offset)
                    self._write_node(last_node_offset, prev_node["key"], prev_node["items"], node_offset)
                last_node_offset = node_offset

            node_offset = next_offset

        if last_node_offset != 0:
            # Ensure the new tail of the chain points to null
            prev_node = self._read_node(last_node_offset)
            self._write_node(last_node_offset, prev_node["key"], prev_node["items"], 0)

        # Now insert the new merged node at head
        new_node_items = list(merged_items)
        new_node_size = self.NODE_HEADER_SIZE + len(new_node_items) * 4
        new_node_offset = self._alloc_node(new_node_size)
        self._write_node(new_node_offset, key, new_node_items, new_chain_head)
        self._write_slot(bucket, new_node_offset)

    def remove_doc_id(self, key, doc_id):
        bucket = self._hash(key)
        node_offset = self._read_slot(bucket)
        prev_offset = 0
        while node_offset != 0:
            node = self._read_node(node_offset)
            if node["key"] == key:
                new_items = [i for i in node["items"] if i != doc_id]
                if len(new_items) == 0:
                    # Remove node from chain
                    if prev_offset == 0:
                        self._write_slot(bucket, node["next"])
                    else:
                        prev_node = self._read_node(prev_offset)
                        self._write_node(prev_offset, prev_node["key"], prev_node["items"], node["next"])
                    return
                else:
                    # Write updated node
                    self._write_node(node_offset, key, new_items, node["next"])
                    return
            prev_offset = node_offset
            node_offset = node["next"]
    def close(self):
        self.mm.flush()
        self.mm.close()
        self.f.close()


# Text preprocessing utils
class MergeSort():
    def __init__(self):
        pass

    def merge_sort(self, arr):
        if len(arr) <= 1:
            return arr  # base case: already sorted

        # Split the array into two halves
        mid = len(arr) // 2
        left_half = self.merge_sort(arr[:mid])
        right_half = self.merge_sort(arr[mid:])

        # Merge the sorted halves
        return self.merge(left_half, right_half)

    def merge(self, left, right):
        sorted_arr = []
        i = j = 0

        # Compare elements from both halves and merge them
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                sorted_arr.append(left[i])
                i += 1
            else:
                sorted_arr.append(right[j])
                j += 1

        # Append remaining elements (if any)
        sorted_arr.extend(left[i:])
        sorted_arr.extend(right[j:])

        return sorted_arr
class SequentialSearch:
    def __init__(self):
        pass

    def split(self, text, delimiter=None):
        if not isinstance(text, str):
            raise TypeError("Input 'text' must be a string.")

        if not text:
            return []

        results = []
        current_word_start = 0

        if delimiter == None or delimiter == '':
            is_in_whitespace_block = True
            for i in range(len(text)):
                if text[i].isspace():
                    if not is_in_whitespace_block:
                        results.append(text[current_word_start:i])
                        is_in_whitespace_block = True
                    current_word_start = i + 1
                else:
                    if is_in_whitespace_block:
                        current_word_start = i
                        is_in_whitespace_block = False
            if not is_in_whitespace_block:
                results.append(text[current_word_start:])
        else:
            delimiter_length = len(delimiter)
            i = 0
            while i <= len(text) - delimiter_length:
                if text[i:i + delimiter_length] == delimiter:
                    results.append(text[current_word_start:i])
                    current_word_start = i + delimiter_length
                    i += delimiter_length
                else:
                    i += 1
            results.append(text[current_word_start:])

        return results

# Tokenizer using whitespace
def tokenize(text):
    s = SequentialSearch()
    return s.split(text)

def preprocess_text(text):
    if not text:
        return []

    # Lowercase
    text = text.lower()

    # Split CamelCase words (e.g., 'QualificationsExperience' → 'Qualifications Experience')
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Replace certain punctuations with space
    text = re.sub(r'[:/\-\\]', ' ', text)

    # Remove non-alphanumeric characters except space
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = tokenize(text)

    # Remove punctuation and stopwords
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Optional stemming
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(t) for t in tokens]

    return tokens


# Insert tokens for a given doc_id, adding doc_id only once per token

def insert_doc_tokens(ht, tokens, doc_id):
    unique_tokens = set(tokens)
    for token in unique_tokens:
        ht.insert(token, [doc_id])


def process_text_string(ht, text, doc_id):
    tokens = preprocess_text(text)
    insert_doc_tokens(ht, tokens, doc_id)


# Example usage:
"""
if __name__ == "__main__":
    ht = MMapChainedHashTable()

    text1 = "Artificial intelligence is the future of technology."
    text2 = "Python is a popular programming language."
    text3 = "AI and Python often go hand in hand."

    process_text_string(ht, text1, doc_id=1)
    process_text_string(ht, text2, doc_id=2)
    process_text_string(ht, text3, doc_id=3)

    print("ai docs:", ht.get("ai"))         # Should show [1, 3]
    print("python docs:", ht.get("python")) # Should show [2, 3]
    print("python docs:", ht.get("programming")) # Should show [2, 3]
    ht.close()
"""

class MergeSort():
    def __init__(self):
        pass

    def merge_sort(self, arr):
        if len(arr) <= 1:
            return arr  # base case: already sorted

        # Split the array into two halves
        mid = len(arr) // 2
        left_half = self.merge_sort(arr[:mid])
        right_half = self.merge_sort(arr[mid:])

        # Merge the sorted halves
        return self.merge(left_half, right_half)

    def merge(self, left, right):
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i][1] >= right[j][1]:  # sort by score descending
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result


def score_jobs(ht, resume_tokens, top_n=1000):
    scores = defaultdict(int)
    
    for token in set(resume_tokens):  # unique tokens to avoid double counting
        doc_ids = ht.get(token)
        if doc_ids:
            for doc_id in doc_ids:
                scores[doc_id] += 1
    
    m = MergeSort()
    sorted_jobs = m.merge_sort(list(scores.items()))
    top_jobs = sorted_jobs[:top_n]
    
    return top_jobs

def truncate_text(text, max_len=300):
    if len(text) > max_len:
        return text[:max_len].rstrip() + "..."
    return text
def paginate_results(results, page_size=10):
    """Splits (doc_id, score) tuples into pages."""
    pages = {}
    for i in range(0, len(results), page_size):
        page_number = (i // page_size) + 1
        pages[page_number] = results[i:i + page_size]
    print("Pages: ", pages)
    return pages

def get_job_snippet(con, doc_id):
    result = con.execute("""
        SELECT title, description, location, company_name
        FROM linkedin_jobs
        WHERE id = ?
    """, [doc_id]).fetchone()
    if result:
        title, job_desc, location, company = result
        return {
            "doc_id": doc_id,
            "title": title,
            "score": None,  # optionally updated later
            "snippet": truncate_text(job_desc),
            "location" : location or None,
            "company_name" : company or None
        }
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_page(con, pages, page_num):
    # Example: send page 1 to client
    page_data = []
    
    for doc_id, score in pages.get(page_num, []):
        job = get_job_snippet(con, doc_id)
        if job:
            job["score"] = score
            page_data.append(job)
    return page_data

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

def close_process_by_pid(pid):
    """Kill a process by its PID (works on both Windows and Linux)."""
    try:
        print(f"Closing process with PID {pid}...")
        if os.name == 'nt':  # Windows
            os.kill(int(pid), 9)  # Hard kill on Windows
        else:  # Linux/Mac
            os.kill(int(pid), signal.SIGKILL)  # Proper kill on Unix-based systems
    except (ValueError, ProcessLookupError, PermissionError):
        print(f"Failed to close process {pid}. It may not exist or lack permissions.")

def extract_pid_from_error(error_message):
    """Extract PID from DuckDB error message."""
    match = re.search(r"PID (\d+)", error_message)
    return match.group(1) if match else None


def connect_to_duckdb(db_path):
    """Try connecting to DuckDB and handle locked database errors."""
    attempts = 3  # Number of retries
    for attempt in range(attempts):
        try:
            print(f"Attempt {attempt + 1}: Connecting to DuckDB at {db_path}...")
            conn = duckdb.connect(db_path)
            print("Connected successfully!")
            return conn
        except duckdb.IOException as e:
            error_msg = str(e)
            print("Database lock detected:", error_msg)

            # Extract PID from error message and close it
            pid = extract_pid_from_error(error_msg)
            if pid:
                close_process_by_pid(pid)
                time.sleep(1)  # Wait before retrying
            else:
                print("No PID found in error message.")
                raise  # Raise if the error is not related to PID lock

    print("Failed to connect after multiple attempts.")
    return None  # Return None if unable to connect
    
con = connect_to_duckdb('linkedin_jobs.db')
ht = MMapChainedHashTable()

if __name__ == "__main__":
    # Connect to your DuckDB database

    # Load the first 100 jobs with combined text fields
    jobs = con.execute("""
        SELECT id, 
               CONCAT_WS(' ',
                    title,
                    description,
                    skills_desc,
                    formatted_experience_level,
                    location,
                    company_name
                ) AS search_text
        FROM linkedin_jobs
        LIMIT 1000;
    """).fetchall()

    # Assume 'resume.pdf' is your resume file
    resume_text = extract_text_from_pdf("resume.pdf")
    # Preprocess the resume text
    resume_tokens = preprocess_text(resume_text)
    
    # Create and fill the inverted index
    
    from tqdm import tqdm
    if False:
        for id, search_text in tqdm(jobs, desc="Processing jobs"):
            process_text_string(ht, search_text, id)
    
    # Example keyword queries
    #print("Architecture docs:", ht.get("architecture"))
    # Get top matching jobs
    top_jobs = score_jobs(ht, resume_tokens)  # list of (doc_id, score)
    print("Top Jobs", top_jobs)
    # Paginate top jobs
    pages = paginate_results(top_jobs, page_size=10)
    page_data = get_page(con, pages, 1)
    

    # Example output for client
    import json
    if False:
        print(json.dumps({
            "page": page_num,
            "total_pages": len(pages),
            "results": page_data
        }, indent=2))

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

uuids = {}
user_pages = {}

@app.route('/job')
def get_job():
    job_id = request.args.get('id')
    if not job_id:
        return jsonify({'error': 'Missing job ID'}), 400

    try:
        job_id = int(job_id)

        result = con.execute("SELECT * FROM linkedin_jobs WHERE id = $1", [job_id]).fetchall()
        columns = [desc[0] for desc in con.description]

        if not result:
            return jsonify({'error': 'Job not found'}), 404

        job = dict(zip(columns, result[0]))
        return jsonify(job)

    except ValueError:
        return jsonify({'error': 'Invalid job ID'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print(f"Client connected with sid: {request.sid}")
    socketio.emit('assign_sid', {'sid': request.sid}, room=request.sid)

@app.route("/api/get_resume", methods=["POST"])
def get_resume():
    try:
        data = request.get_json()
        print("Received data: ", data)
        device_uuid = data.get("device_uuid")
        output = uuids[device_uuid]
        
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/get_page", methods=["POST"])
def get_page_api():
    data = request.get_json()
    print("Matching: ", data)
    sid = data.get("sid")
    device_uuid = data.get("device_uuid")
    print("Matching: ", sid, device_uuid)
    if not sid:
        return jsonify({"error": "Missing sid"}), 400
    if not device_uuid:
        return jsonify({"error": "Missing device_uuid"}), 400
    page = data.get("page")
    if not page:
        return jsonify({"error": "Missing page number"}), 400
    print("Matching: ", sid, page)

    if True:
        pages = user_pages[device_uuid]
        page_data = get_page(con, pages, page)
    #socketio.emit('assign_sid', {'sid': request.sid}, room=request.sid)
    return jsonify({
        "page": page,
        "total_pages": len(pages),
        "results": page_data
    })

@app.route("/api/match_jobs", methods=["POST"])
def match_jobs():
    data = request.get_json()
    print("Matching: ", data)
    sid = data.get("sid")
    device_uuid = data.get("device_uuid")
    print("Matching: ", sid, device_uuid)
    if not sid:
        return jsonify({"error": "Missing sid"}), 400
    uploaded_files = data.get("uploadedFiles")
    if not uploaded_files:
        return jsonify({"error": "No files uploaded"}), 400
    page = data.get("page")
    if not page:
        return jsonify({"error": "Missing page number"}), 400
    print("Matching: ", sid, uploaded_files, page)
    # Assume 'resume.pdf' is your resume file
    if True:
        resume_text = extract_text_from_pdf(f"uploads/{sid}/{uploaded_files}")
        # Preprocess the resume text
        resume_tokens = preprocess_text(resume_text)
        
        uuids[device_uuid] = {
            "resume_tokens": resume_tokens,
            "resume_text" : resume_text,
        }

        print(uuids[device_uuid])
        
        # Example keyword queries
        global ht
        print("Architecture docs:", ht.get("architecture"))
        # Get top matching jobs
        top_jobs = score_jobs(ht, resume_tokens)  # list of (doc_id, score)
        #print("Top Jobs", top_jobs)
        # Paginate top jobs
        pages = paginate_results(top_jobs, page_size=10)
        user_pages[device_uuid] = pages
        page_data = get_page(con, pages, page)
    #socketio.emit('assign_sid', {'sid': request.sid}, room=request.sid)
    return jsonify({
        "page": page,
        "total_pages": len(pages),
        "results": page_data
    })

@app.route('/upload', methods=['POST'])
def upload():
    sid = request.form.get('sid')
    files = request.files.getlist('files')

    if not sid or not files:
        return jsonify({'message': 'Missing sid or files'}), 400

    sid_folder = os.path.join(UPLOAD_FOLDER, secure_filename(sid))
    os.makedirs(sid_folder, exist_ok=True)

    for file in files:
        filename = secure_filename(file.filename)
        file.save(os.path.join(sid_folder, filename))

    return jsonify({'message': f'{len(files)} file(s) saved to {sid_folder}'}), 200

@app.route('/api/ai_recommendation', methods=['POST'])
def ai_recommendation():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 1024,
                "topP": 1,
                "topK": 40
            }
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))

        if response.ok:
            result = response.json()
            # Defensive check for nested structure
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response")
            return jsonify({"result": text})
        else:
            return jsonify({"error": response.text}), response.status_code

    except Exception as e:
        print("❌ Exception occurred in /api/ai_recommendation:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)