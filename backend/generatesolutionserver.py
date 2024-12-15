from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows requests from other origins

@app.route('/generate-solution', methods=['POST'])
def generate_solution():
    data = request.json
    query = data.get('query', '')
    category = data.get('category', '')

    # Example solutions (replace this with actual logic)
    solutions = [
        f"Solution 1 for query: '{query}' in category: '{category}'.",
        f"Solution 2 for query: '{query}' in category: '{category}'.",
        f"Solution 3 for query: '{query}' in category: '{category}'."
    ]

    return jsonify({"solutions": solutions})

if __name__ == '__main__':
    app.run(debug=True)
