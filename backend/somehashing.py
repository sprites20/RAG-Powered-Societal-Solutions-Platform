import mmap
import os
import struct
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('punkt')
nltk.download('stopwords')

class MMapChainedHashTable:
    TABLE_SIZE = 100_000  # number of buckets
    SLOT_SIZE = 8         # each slot stores 8-byte offset to chain head (0 = empty)
    KEY_SIZE = 32         # fixed length for keys (padded/truncated)
    NODE_HEADER_SIZE = KEY_SIZE + 4 + 4 + 8
    # Node layout: key(32) + count_of_items(4) + items_region_size(4) + next_node_offset(8)
    # Items follow immediately after header, variable length: count_of_items * 4-byte integers (example items)

    DATA_REGION_SIZE = 500 * 1024 * 1024  # 500 MB for nodes and items
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
        prev_offset = 0
        while node_offset != 0:
            node = self._read_node(node_offset)
            if node["key"] == key:
                # Update: append unique items
                new_items = node["items"][:]
                for item in items:
                    if item not in new_items:
                        new_items.append(item)
                # Write back updated node (assume size fits original node size for simplicity)
                if len(new_items) <= node["count"]:  # If smaller or same size, just overwrite
                    self._write_node(node_offset, key, new_items, node["next"])
                else:
                    # For simplicity, allocate new node, insert, remove old by linking prev
                    new_node_size = self.NODE_HEADER_SIZE + len(new_items)*4
                    new_node_offset = self._alloc_node(new_node_size)
                    self._write_node(new_node_offset, key, new_items, node["next"])
                    if prev_offset == 0:
                        self._write_slot(bucket, new_node_offset)
                    else:
                        # update prev node's next pointer to new node
                        prev_node = self._read_node(prev_offset)
                        self._write_node(prev_offset, prev_node["key"], prev_node["items"], new_node_offset)
                    # (old node remains garbage, free list needed in production)
                return

            prev_offset = node_offset
            node_offset = node["next"]

        # Not found: insert new node at head of chain
        new_node_size = self.NODE_HEADER_SIZE + len(items)*4
        new_node_offset = self._alloc_node(new_node_size)
        chain_head = self._read_slot(bucket)
        self._write_node(new_node_offset, key, items, chain_head)
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
    tokens = word_tokenize(text)

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
import duckdb
from collections import defaultdict

def score_jobs(ht, resume_tokens, top_n=10):
    scores = defaultdict(int)
    
    for token in set(resume_tokens):  # unique tokens to avoid double counting
        doc_ids = ht.get(token)
        if doc_ids:
            for doc_id in doc_ids:
                scores[doc_id] += 1
    
    # Sort job ids by score descending and pick top_n
    top_jobs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
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
    return pages

def get_job_snippet(con, doc_id):
    result = con.execute("""
        SELECT title, description 
        FROM linkedin_jobs
        WHERE id = ?
    """, [doc_id]).fetchone()
    if result:
        title, job_desc = result
        return {
            "doc_id": doc_id,
            "title": title,
            "score": None,  # optionally updated later
            "snippet": truncate_text(job_desc)
        }
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
    
if __name__ == "__main__":
    # Connect to your DuckDB database
    con = duckdb.connect("linkedin_jobs.db")

    # Load the first 100 jobs with combined text fields
    jobs = con.execute("""
        SELECT id, 
               CONCAT_WS(' ',
                    title,
                    description,
                    skills_desc,
                    formatted_experience_level,
                    formatted_work_type,
                    location,
                    CASE WHEN remote_allowed = TRUE THEN 'remote' ELSE '' END,
                    work_type,
                    company_name
                ) AS search_text
        FROM linkedin_jobs
        LIMIT 100;
    """).fetchall()

    # Assume 'resume.pdf' is your resume file
    resume_text = extract_text_from_pdf("resume.pdf")
    # Preprocess the resume text
    resume_tokens = preprocess_text(resume_text)

    # Create and fill the inverted index
    ht = MMapChainedHashTable()
    """
    for id, search_text in jobs:
        print("Processing id:", id)
        process_text_string(ht, search_text, id)
    """
    # Example keyword queries
    print("Architecture docs:", ht.get("architecture"))
    # Get top matching jobs
    top_jobs = score_jobs(ht, resume_tokens)  # list of (doc_id, score)

    # Paginate top jobs
    pages = paginate_results(top_jobs, page_size=10)

    # Example: send page 1 to client
    page_num = 1
    page_data = []

    for doc_id, score in pages.get(page_num, []):
        job = get_job_snippet(con, doc_id)
        if job:
            job["score"] = score
            page_data.append(job)

    # Example output for client
    import json
    print(json.dumps({
        "page": page_num,
        "total_pages": len(pages),
        "results": page_data
    }, indent=2))

    ht.close()