import os
import json
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dataclasses import asdict
from datetime import datetime

# Import the marketplace classes from your existing main.py file
from main import DataMarketplace, PinataIPFSClient, AIDataCurator, Web3SmartContract

# --- INITIALIZATION ---

app = Flask(__name__, static_folder='.', static_url_path='')

# 💡 IMPORTANT: CORS is required to allow your index.html (browser)
# to communicate with this Flask server.
CORS(app)

# Define a folder for temporary file uploads
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper to handle objects that are not JSON serializable by default
def json_default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

# --- Initialize the Marketplace just once when the server starts ---
print("🚀 Initializing marketplace components for the server...")

PINATA_API_KEY = os.getenv("PINATA_API_KEY", "your_pinata_api_key")
PINATA_SECRET = os.getenv("PINATA_SECRET", "your_pinata_secret")
RPC_URL = os.getenv("RPC_URL", "https://sepolia.infura.io/v3/your_project_id")
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "0x" + "0" * 64)

ipfs_client = PinataIPFSClient(PINATA_API_KEY, PINATA_SECRET)
ai_curator = AIDataCurator()
smart_contract = Web3SmartContract(RPC_URL, PRIVATE_KEY)
marketplace = DataMarketplace(ipfs_client, ai_curator, smart_contract)

print("✅ Marketplace server is ready.")


# --- API ENDPOINTS ---

@app.route('/')
def serve_index():
    # This will serve your index.html file
    return send_from_directory('.', 'index.html')

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Returns all datasets in the marketplace."""
    query = request.args.get('query', '')
    min_quality = float(request.args.get('min_quality', 0))
    
    search_results = marketplace.search_datasets(query=query, min_quality=min_quality)
    
    # Convert dataclass objects to dictionaries for JSON response
    datasets_dict = [asdict(d) for d in search_results]
    
    return json.dumps(datasets_dict, default=json_default)

@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """Handles new dataset uploads from the web form."""
    if 'datasetFile' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['datasetFile']
    title = request.form.get('title')
    description = request.form.get('description')
    owner_address = request.form.get('ownerAddress')

    if file.filename == '' or not title or not description or not owner_address:
        return jsonify({"error": "Missing form data"}), 400

    if file:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        # Use the marketplace's upload logic
        dataset_id = marketplace.upload_dataset(
            file_path=temp_path,
            title=title,
            description=description,
            owner_address=owner_address
        )
        
        # Clean up the temporarily saved file
        os.remove(temp_path)

        if dataset_id:
            new_dataset = marketplace.get_dataset_info(dataset_id)
            return json.dumps(asdict(new_dataset), default=json_default), 201
        else:
            return jsonify({"error": "Failed to process and upload dataset"}), 500

@app.route('/api/purchase/<string:dataset_id>', methods=['POST'])
def purchase_dataset(dataset_id):
    """Processes a purchase request for a given dataset."""
    data = request.get_json()
    buyer_address = data.get('buyerAddress')
    
    if not buyer_address:
        return jsonify({"error": "Buyer address is required"}), 400
        
    success = marketplace.purchase_data_access(
        dataset_id=dataset_id,
        buyer_address=buyer_address
    )
    
    if success:
        return jsonify({"message": "Purchase successful", "dataset_id": dataset_id}), 200
    else:
        return jsonify({"error": "Purchase failed. Dataset not found or error occurred."}), 404

# --- RUN THE SERVER ---

if __name__ == '__main__':
    # Using port 5000 by default. Debug=True allows auto-reloading on code changes.
    app.run(debug=True, port=5000)