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

import mmap, os, struct

class MMapChainedHashTable:
    # ────────────────────────────────────────────────────────────
    # tunables
    TABLE_SIZE      = 100_000
    SLOT_SIZE       = 8
    KEY_SIZE        = 32
    MAX_ITEMS       = 64
    DATA_REGION_MB  = 500

    # ────────────────────────────────────────────────────────────
    # derived constants
    NODE_HEADER_SIZE = KEY_SIZE + 4 + 8 + 8  # key + count + next_same + next_other
    NODE_SIZE        = NODE_HEADER_SIZE + MAX_ITEMS * 4
    DATA_REGION_SIZE = DATA_REGION_MB * 1024 * 1024
    FILE_SIZE        = SLOT_SIZE * TABLE_SIZE + DATA_REGION_SIZE

    def __init__(self, filename="mmap_chain_hash.dat"):
        self.filename = filename
        self._init_file()
        self.f  = open(self.filename, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), self.FILE_SIZE)
        self.data_start      = self.SLOT_SIZE * self.TABLE_SIZE
        self.next_free_off   = self.data_start

    def _init_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, "wb") as f:
                f.truncate(self.FILE_SIZE)

    def _hash(self, key: str) -> int:
        return sum(key.encode("utf-8")) % self.TABLE_SIZE

    def _slot_off(self, index: int) -> int:
        return index * self.SLOT_SIZE

    def _read_u64(self, off: int) -> int:
        return struct.unpack_from("Q", self.mm, off)[0]

    def _write_u64(self, off: int, val: int) -> None:
        struct.pack_into("Q", self.mm, off, val)

    def _read_node(self, off: int):
        key_bytes = self.mm[off : off + self.KEY_SIZE]
        count     = struct.unpack_from("I", self.mm, off + self.KEY_SIZE)[0]
        same_off  = struct.unpack_from("Q", self.mm, off + self.KEY_SIZE + 4)[0]
        other_off = struct.unpack_from("Q", self.mm, off + self.KEY_SIZE + 12)[0]
        key       = key_bytes.rstrip(b"\x00").decode("utf-8")
        base      = off + self.NODE_HEADER_SIZE
        items     = [struct.unpack_from("I", self.mm, base + i*4)[0] for i in range(count)]
        return {"key": key, "count": count, "same": same_off, "other": other_off, "off": off, "items": items}

    def _write_empty_node(self, off: int, key: str, same: int = 0, other: int = 0):
        key_b = key.encode("utf-8")[:self.KEY_SIZE].ljust(self.KEY_SIZE, b"\x00")
        struct.pack_into(f"{self.KEY_SIZE}sIQQ", self.mm, off, key_b, 0, same, other)

    def _append_item_to_node(self, node_off: int, doc_id: int):
        count_off = node_off + self.KEY_SIZE
        count     = struct.unpack_from("I", self.mm, count_off)[0]
        if count >= self.MAX_ITEMS:
            raise ValueError("node already full")
        items_base = node_off + self.NODE_HEADER_SIZE
        struct.pack_into("I", self.mm, items_base + count*4, doc_id)
        struct.pack_into("I", self.mm, count_off, count + 1)

    def _alloc_node(self) -> int:
        if self.next_free_off + self.NODE_SIZE > self.FILE_SIZE:
            raise RuntimeError("out of data region")
        off = self.next_free_off
        self.next_free_off += self.NODE_SIZE
        return off

    def insert(self, key: str, doc_ids):
        if isinstance(doc_ids, int): doc_ids = [doc_ids]

        bucket     = self._hash(key)
        slot_off   = self._slot_off(bucket)
        node_off   = self._read_u64(slot_off)
        prev_other = 0

        while node_off:
            node = self._read_node(node_off)
            if node["key"] == key:
                break
            prev_other = node_off
            node_off   = node["other"]

        if node_off:
            same_off = node_off
            prev_same = 0
            while same_off:
                n = self._read_node(same_off)
                if n["count"] < self.MAX_ITEMS:
                    for d in doc_ids[:]:
                        if d in n["items"]:
                            doc_ids.remove(d)
                            continue
                        if n["count"] == self.MAX_ITEMS:
                            break
                        self._append_item_to_node(same_off, d)
                    if not doc_ids: return
                prev_same = same_off
                same_off = n["same"]

            while doc_ids:
                new_off = self._alloc_node()
                self._write_empty_node(new_off, key)
                for d in doc_ids[:self.MAX_ITEMS]:
                    self._append_item_to_node(new_off, d)
                doc_ids = doc_ids[self.MAX_ITEMS:]
                self._write_u64(prev_same + self.KEY_SIZE + 4, new_off)
                prev_same = new_off
            return

        while doc_ids:
            new_off = self._alloc_node()
            self._write_empty_node(new_off, key)
            for d in doc_ids[:self.MAX_ITEMS]:
                self._append_item_to_node(new_off, d)
            doc_ids = doc_ids[self.MAX_ITEMS:]

            if prev_other == 0:
                self._write_u64(slot_off, new_off)
            else:
                self._write_u64(prev_other + self.KEY_SIZE + 12, new_off)
            prev_other = new_off

    def get(self, key: str):
        bucket     = self._hash(key)
        node_off   = self._read_u64(self._slot_off(bucket))
        items = []
        while node_off:
            node = self._read_node(node_off)
            if node["key"] == key:
                while node:
                    items.extend(node["items"])
                    node = self._read_node(node["same"]) if node["same"] else None
                break
            node_off = node["other"]
        return items if items else None
    def remove_doc_id(self, key: str, doc_id: int):
        bucket   = self._hash(key)
        node_off = self._read_u64(self._slot_off(bucket))

        while node_off:
            node = self._read_node(node_off)
            if node["key"] == key:
                while node:
                    items = node["items"]
                    try:
                        i = items.index(doc_id)
                        last_idx = node["count"] - 1
                        items_base = node["off"] + self.NODE_HEADER_SIZE
                        if i != last_idx:
                            last_val = items[last_idx]
                            struct.pack_into("I", self.mm, items_base + i*4, last_val)
                        struct.pack_into("I", self.mm, node["off"] + self.KEY_SIZE, last_idx)
                        return
                    except ValueError:
                        node = self._read_node(node["same"]) if node["same"] else None
                return
            node_off = node["other"]
    def close(self):
        self.mm.flush(); self.mm.close(); self.f.close()




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
    if True:
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

@app.route("/api/get_page_api", methods=["POST"])
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