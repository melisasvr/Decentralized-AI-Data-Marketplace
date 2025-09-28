# Decentralized AI Data Marketplace 🚀
 - A blockchain-powered platform for buying, selling, and discovering high-quality datasets, curated by AI and stored on IPFS.
 -  The backend combines Flask for API handling and core Python logic for marketplace operations, with a simple HTML/JS frontend. 🤖📊

## 📝 Description
- This project creates a decentralized data marketplace where users can upload datasets, which are analyzed by AI for quality, stored on IPFS via Pinata, and transacted via Ethereum smart contracts.
- The backend is split into:
- app.py: Flask server providing RESTful API endpoints for dataset management and frontend interaction.
- main.py: Core Python logic for the marketplace, including IPFS integration, AI-powered data curation, and blockchain transactions.
- Frontend: HTML with Tailwind CSS and JavaScript, offering a user-friendly interface to browse, upload, and purchase datasets.

## Key technologies:
- Backend: Flask (app.py) and Python (main.py) 🧪
- Blockchain: Web3.py & Ethereum (Sepolia testnet) 🔗
- Storage: IPFS (via Pinata) ☁️
- AI Curation: Pandas, NumPy for data quality assessment 📈
- Frontend: HTML/JS with Tailwind CSS 🎨

## ✨ Features
- Dataset Upload & Analysis 📤: Upload CSV/JSON files, get AI-generated quality scores, tags, schema, and suggested pricing.
- Search & Browse 🔍: Filter datasets by query and minimum quality score via the API.
- Purchase Access 💳: Simulate blockchain purchases with buyer/seller addresses.
- Wallet Integration 🔑: Connect MetaMask for user authentication.
- Analytics Dashboard 📊: View static marketplace stats (extendable to dynamic).
- Modals & UI Feedback 🖥️: Processing modals for uploads and detailed dataset views.
- Demo Mode 🎮: Run a sample e-commerce dataset demo in main.py.

## 🛠️ Prerequisites
- Python 3.8+ 🐍
- Node.js (for JS dependencies, minimal) 🌐
- MetaMask wallet for testing wallet connections 💼
- API Keys:
- Pinata (for IPFS) 🔑
- Infura or Alchemy (for Ethereum RPC) 🌐

## 📦 Installation
1. Clone the repository:
- `git clone https://github.com/yourusername/yourrepo.git`
- `cd yourrepo`
2. Install Python dependencies:
- `pip install flask flask-cors werkzeug pandas numpy web3 requests`
3. Set up the uploads folder:
`mkdir uploads`

## ⚙️ Configuration
- Set environment variables or update app.py and main.py with:
- PINATA_API_KEY: Your Pinata API key ☁️
- PINATA_SECRET: Your Pinata secret key 🔒
- RPC_URL: Ethereum RPC endpoint (e.g., https://sepolia.infura.io/v3/your_project_id) 🔗
- PRIVATE_KEY: Ethereum private key (use a test account!) 🗝️
- Demo mode uses placeholder keys, but production requires real credentials.

## 🚀 Running the Application
1. Start the Flask server:
- `python app.py`
- Runs on http://localhost:5000 🌍
- Debug mode enables auto-reloading 🔄
- Serves the frontend (index.html) and API endpoints.
2. Open `http://localhost:5000` in your browser 🖥️
- Connect your wallet 🔗
- Upload datasets 📤
- Browse and purchase datasets 🔍💳
3. Run the demo script (optional, tests core logic in main.py):
- `python main.py`
- Generates a sample e-commerce dataset, uploads it, and simulates a purchase 🎉

## 📖 Usage
- Frontend Interface 🖼️
- Connect Wallet: Click "🔗 Connect Wallet" to link MetaMask.
- Upload Data: Go to "📤 Upload Dataset", fill in details, and upload a CSV/JSON file.
- Browse Datasets: Search and filter datasets, view details, and purchase.
- My Data: View datasets you uploaded (filtered by wallet address).
- Analytics: See static marketplace trends (e.g., 67% high-quality data).

## Backend Components ⚙️
- app.py (Flask API):
- GET /api/datasets?query=<query>&min_quality=<score>: Search datasets.
- POST /api/upload: Upload a dataset (multipart form: file, title, description, ownerAddress).
- POST /api/purchase/<dataset_id>: Purchase a dataset (JSON: buyerAddress).
- main.py (Core Logic):
- Handles IPFS uploads/downloads via PinataIPFSClient.
- Performs AI analysis with AIDataCurator (quality scores, tags, pricing).
- Manages blockchain transactions via Web3SmartContract.
- Stores dataset metadata in marketplace.json.

## Example Workflow 👟
1. Upload a dataset via the UI.
2. AI analyzes it, assigns a quality score, and suggests a price.
3. The dataset is pinned to IPFS and registered on the blockchain.
4. Search and purchase it using another wallet address.

## 🐛 Troubleshooting
- IPFS Upload Fails: Verify Pinata keys; demo mode uses mock hashes.
- Blockchain Errors: Check RPC URL and private key; use Sepolia testnet.
- File Not Found: Ensure files are in the uploads folder with correct permissions.
- CORS Issues: Flask-CORS is enabled; confirm the browser allows localhost requests.
- Demo Fails: Run python main.py to debug; check for missing dependencies.

## 🤝 Contributing
- Contributions welcome! Fork the repo, make changes, and submit a pull request. Open an issue for major updates. 🌟

## 📄 License
- MIT License - free to use and modify! 📜

- Built with ❤️ by [Melisa Sever]. Questions? Open an issue! 🚀
