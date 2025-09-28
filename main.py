# Decentralized AI-Curated Data Marketplace - FIXED VERSION
# Core implementation with IPFS integration, AI curation, and smart contract interaction

import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
from pathlib import Path
import pickle
from web3 import Web3
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Data structure definitions
@dataclass
class DatasetMetadata:
    """Metadata for a dataset in the marketplace"""
    dataset_id: str
    title: str
    description: str
    owner_address: str
    ipfs_hash: str
    file_size: int
    upload_timestamp: datetime
    tags: List[str]
    quality_score: float
    suggested_price: float
    data_type: str
    schema: Dict
    access_count: int = 0
    rating: float = 0.0

@dataclass
class QualityMetrics:
    """Quality assessment metrics for datasets"""
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    uniqueness_score: float
    timeliness_score: float
    overall_score: float

class PinataIPFSClient:
    """Pinata IPFS client for production use"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.pinata.cloud"
        
    def add_file(self, file_path: str, metadata: Dict = None) -> str:
        """Upload file to IPFS via Pinata"""
        url = f"{self.base_url}/pinning/pinFileToIPFS"
        
        headers = {
            'pinata_api_key': self.api_key,
            'pinata_secret_api_key': self.secret_key
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                return self._generate_mock_hash(file_path)
                
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f, 'application/octet-stream')}
                
                # Add metadata if provided
                data = {}
                if metadata:
                    data['pinataMetadata'] = json.dumps(metadata)
                    data['pinataOptions'] = json.dumps({"cidVersion": 1})
                
                # Only try real upload if we have real API keys
                if self.api_key != "your_pinata_api_key" and self.secret_key != "your_pinata_secret":
                    response = requests.post(url, files=files, data=data, headers=headers, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ File uploaded to IPFS: {result['IpfsHash']}")
                        return result['IpfsHash']
                    else:
                        print(f"⚠️ Pinata upload failed: {response.status_code}")
                        return self._generate_mock_hash(file_path)
                else:
                    print("🔧 Using demo mode - mock IPFS upload")
                    return self._generate_mock_hash(file_path)
                    
        except requests.exceptions.RequestException as e:
            print(f"🔧 Network error, using mock hash: {str(e)[:50]}...")
            return self._generate_mock_hash(file_path)
        except Exception as e:
            print(f"⚠️ IPFS upload error: {str(e)[:100]}...")
            return self._generate_mock_hash(file_path)
    
    def get_file(self, ipfs_hash: str, output_path: str) -> bool:
        """Download file from IPFS via gateway"""
        try:
            gateway_url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
            response = requests.get(gateway_url, timeout=30)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ File downloaded from IPFS: {ipfs_hash}")
                return True
            else:
                print(f"⚠️ IPFS download failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"⚠️ IPFS download error: {str(e)[:100]}...")
            return False
    
    def _generate_mock_hash(self, file_path: str) -> str:
        """Generate mock IPFS hash for demo"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            mock_hash = "Qm" + hashlib.sha256(content).hexdigest()[:44]
        except:
            # If file doesn't exist, generate based on filename
            mock_hash = "Qm" + hashlib.sha256(file_path.encode()).hexdigest()[:44]
        
        print(f"🔧 Using mock IPFS hash: {mock_hash}")
        return mock_hash

class AIDataCurator:
    """AI-powered data analysis and curation engine"""
    
    def __init__(self):
        self.quality_weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'accuracy': 0.20,
            'uniqueness': 0.15,
            'timeliness': 0.20
        }
    
    def analyze_dataset(self, file_path: str) -> Tuple[List[str], QualityMetrics, float, str, Dict]:
        """Comprehensive AI analysis of dataset"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                return self._analyze_missing_file(file_path)
            
            # Load and analyze the dataset
            if file_path.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    data_type = "tabular"
                except Exception as e:
                    print(f"⚠️ Error reading CSV: {str(e)[:100]}...")
                    return self._analyze_binary_file(file_path)
                    
            elif file_path.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        df = pd.json_normalize(data)
                    else:
                        df = pd.DataFrame([data])
                    data_type = "json"
                except Exception as e:
                    print(f"⚠️ Error reading JSON: {str(e)[:100]}...")
                    return self._analyze_binary_file(file_path)
            else:
                # Handle other file types
                return self._analyze_binary_file(file_path)
            
            # Generate tags
            tags = self._generate_tags(df)
            
            # Calculate quality metrics
            quality = self._assess_quality(df)
            
            # Suggest pricing
            price = self._suggest_pricing(df, quality)
            
            # Generate schema
            schema = self._generate_schema(df)
            
            return tags, quality, price, data_type, schema
            
        except Exception as e:
            print(f"⚠️ Error analyzing dataset: {str(e)[:100]}...")
            return self._analyze_binary_file(file_path)
    
    def _generate_tags(self, df: pd.DataFrame) -> List[str]:
        """Generate relevant tags based on data content"""
        tags = []
        
        try:
            # Column-based tags
            columns = [str(col).lower() for col in df.columns]
            
            # Detect common data patterns
            for col in columns:
                if any(keyword in col for keyword in ['date', 'time', 'timestamp']):
                    tags.append('temporal')
                elif any(keyword in col for keyword in ['price', 'cost', 'amount', 'value', 'money']):
                    tags.append('financial')
                elif any(keyword in col for keyword in ['location', 'address', 'geo', 'city', 'country']):
                    tags.append('geospatial')
                elif any(keyword in col for keyword in ['user', 'customer', 'person', 'name']):
                    tags.append('user-data')
            
            # Data type tags
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            text_cols = len(df.select_dtypes(include=['object']).columns)
            
            if numeric_cols > text_cols:
                tags.append('numerical')
            elif text_cols > numeric_cols:
                tags.append('textual')
            else:
                tags.append('mixed')
            
            # Size-based tags
            if len(df) > 10000:
                tags.append('large-dataset')
            elif len(df) < 100:
                tags.append('small-dataset')
            else:
                tags.append('medium-dataset')
                
        except Exception as e:
            print(f"⚠️ Error generating tags: {str(e)[:50]}...")
            tags = ['unknown', 'data']
        
        return list(set(tags)) if tags else ['data']  # Remove duplicates, ensure at least one tag
    
    def _assess_quality(self, df: pd.DataFrame) -> QualityMetrics:
        """Assess data quality across multiple dimensions"""
        try:
            # Completeness: percentage of non-null values
            total_cells = len(df) * len(df.columns)
            non_null_cells = df.count().sum()
            completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
            
            # Consistency: check for data type consistency
            consistency = 0
            for col in df.columns:
                try:
                    if df[col].dtype == 'object':
                        # Check string format consistency
                        non_null_values = df[col].dropna().astype(str)
                        if len(non_null_values) > 0:
                            lengths = non_null_values.str.len()
                            mean_length = lengths.mean()
                            if mean_length > 0:
                                consistency += max(0, (1 - lengths.std() / mean_length)) * 100
                            else:
                                consistency += 50
                        else:
                            consistency += 0
                    else:
                        consistency += 90  # Numeric columns are generally consistent
                except Exception:
                    consistency += 50
            
            consistency = min(consistency / len(df.columns), 100) if len(df.columns) > 0 else 0
            
            # Accuracy: basic validation checks
            accuracy = 85.0  # Placeholder - would implement domain-specific validation
            
            # Uniqueness: check for duplicate rows
            if len(df) > 0:
                uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
            else:
                uniqueness = 100
            
            # Timeliness: assume recent data is more valuable
            timeliness = 80.0  # Placeholder - would check timestamps if available
            
            # Calculate overall score
            overall = (
                completeness * self.quality_weights['completeness'] +
                consistency * self.quality_weights['consistency'] +
                accuracy * self.quality_weights['accuracy'] +
                uniqueness * self.quality_weights['uniqueness'] +
                timeliness * self.quality_weights['timeliness']
            )
            
        except Exception as e:
            print(f"⚠️ Error assessing quality: {str(e)[:50]}...")
            # Return default values
            completeness = consistency = accuracy = uniqueness = timeliness = overall = 50.0
        
        return QualityMetrics(
            completeness, consistency, accuracy, uniqueness, timeliness, overall
        )
    
    def _suggest_pricing(self, df: pd.DataFrame, quality: QualityMetrics) -> float:
        """AI-driven pricing suggestion based on data characteristics"""
        try:
            # Base price factors
            size_factor = min(len(df) / 1000, 10) if len(df) > 0 else 0.1  # Max 10x for size
            column_factor = min(len(df.columns) / 10, 3) if len(df.columns) > 0 else 0.1  # Max 3x for features
            quality_factor = max(quality.overall_score / 100, 0.1)  # Ensure minimum factor
            
            # Base price in tokens/ETH equivalent
            base_price = 0.01
            suggested_price = base_price * size_factor * column_factor * quality_factor
            
            return round(max(suggested_price, 0.001), 6)  # Minimum price of 0.001
        except Exception as e:
            print(f"⚠️ Error suggesting pricing: {str(e)[:50]}...")
            return 0.01
    
    def _generate_schema(self, df: pd.DataFrame) -> Dict:
        """Generate schema description for the dataset"""
        try:
            schema = {
                'columns': {},
                'row_count': len(df),
                'column_count': len(df.columns),
                'data_types': {}
            }
            
            for col in df.columns:
                try:
                    schema['columns'][str(col)] = {
                        'type': str(df[col].dtype),
                        'null_count': int(df[col].isnull().sum()),
                        'unique_count': int(df[col].nunique())
                    }
                    schema['data_types'][str(col)] = str(df[col].dtype)
                except Exception:
                    schema['columns'][str(col)] = {
                        'type': 'unknown',
                        'null_count': 0,
                        'unique_count': 0
                    }
                    schema['data_types'][str(col)] = 'unknown'
            
        except Exception as e:
            print(f"⚠️ Error generating schema: {str(e)[:50]}...")
            schema = {'columns': {}, 'row_count': 0, 'column_count': 0, 'data_types': {}}
        
        return schema
    
    def _analyze_binary_file(self, file_path: str) -> Tuple[List[str], QualityMetrics, float, str, Dict]:
        """Analyze non-tabular files"""
        try:
            if os.path.exists(file_path):
                file_size = Path(file_path).stat().st_size
            else:
                file_size = 0
                
            file_ext = Path(file_path).suffix.lower()
            
            tags = ['binary']
            if file_ext:
                tags.append(f'{file_ext[1:]}-file')
            else:
                tags.append('unknown-format')
            
            # Basic quality assessment for binary files
            quality = QualityMetrics(100, 80, 70, 100, 80, 86)
            
            # Price based on file size
            price = max(0.001, file_size / (1024 * 1024) * 0.1)  # $0.1 per MB
            
            schema = {
                'file_type': 'binary',
                'file_size': file_size,
                'extension': file_ext
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing binary file: {str(e)[:50]}...")
            tags = ['unknown']
            quality = QualityMetrics(50, 50, 50, 50, 50, 50)
            price = 0.001
            schema = {'file_type': 'unknown', 'file_size': 0, 'extension': ''}
        
        return tags, quality, price, 'binary', schema
    
    def _analyze_missing_file(self, file_path: str) -> Tuple[List[str], QualityMetrics, float, str, Dict]:
        """Handle missing file case"""
        tags = ['missing-file']
        quality = QualityMetrics(0, 0, 0, 0, 0, 0)
        price = 0.001
        schema = {'file_type': 'missing', 'file_size': 0, 'extension': ''}
        return tags, quality, price, 'unknown', schema

class Web3SmartContract:
    """Production Web3 smart contract interface"""
    
    def __init__(self, rpc_url: str, private_key: str, nft_contract_address: str = None, marketplace_contract_address: str = None):
        self.rpc_url = rpc_url
        self.private_key = private_key
        self.nft_contract_address = nft_contract_address
        self.marketplace_contract_address = marketplace_contract_address
        self.connected = False
        self.w3 = None
        self.account = None
        
        # Simple NFT contract ABI (you'd deploy actual contracts)
        self.nft_abi = [
            {
                "inputs": [{"name": "_to", "type": "address"}, {"name": "_tokenURI", "type": "string"}],
                "name": "mint",
                "outputs": [{"name": "tokenId", "type": "uint256"}],
                "type": "function"
            }
        ]
        
        # Marketplace contract ABI
        self.marketplace_abi = [
            {
                "inputs": [
                    {"name": "_datasetId", "type": "string"},
                    {"name": "_price", "type": "uint256"},
                    {"name": "_duration", "type": "uint256"}
                ],
                "name": "createAccessContract",
                "outputs": [{"name": "contractId", "type": "bytes32"}],
                "type": "function"
            }
        ]
        
        self._initialize_web3()
    
    def _initialize_web3(self):
        """Initialize Web3 connection safely"""
        try:
            # Only try real connection if we have real credentials
            if (self.rpc_url != "https://sepolia.infura.io/v3/your_project_id" and 
                self.private_key != "0x" + "0" * 64):
                
                self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
                self.account = self.w3.eth.account.from_key(self.private_key)
                
                # Test connection with timeout
                self.connected = self.w3.is_connected()
                
                if self.connected:
                    print(f"🔗 Connected to blockchain: {self.connected}")
                    print(f"🏦 Account address: {self.account.address}")
                    
                    balance = self.w3.from_wei(self.w3.eth.get_balance(self.account.address), 'ether')
                    print(f"💰 Account balance: {balance:.6f} ETH")
                else:
                    print("⚠️ No blockchain connection - using demo mode")
            else:
                print("🔧 Demo credentials detected - using demo mode")
                self.connected = False
                
                # Create mock account for demo
                self.account = type('MockAccount', (), {
                    'address': '0x742d35cc6634C0532925a3b8D87b16A9D4c5c5c5'
                })()
                print(f"🏦 Demo account address: {self.account.address}")
                
        except Exception as e:
            self.connected = False
            print(f"⚠️ Blockchain connection failed: {str(e)[:100]}...")
            
            # Create mock account for demo
            self.account = type('MockAccount', (), {
                'address': '0x742d35cc6634C0532925a3b8D87b16A9D4c5c5c5'
            })()
            print(f"🏦 Demo account address: {self.account.address}")
            print("⚠️ Using demo mode for all transactions")
    
    def mint_data_nft(self, metadata: DatasetMetadata, owner_address: str) -> str:
        """Mint NFT representing data ownership"""
        if not self.connected:
            return self._mock_mint_nft(metadata, owner_address)
            
        try:
            if not self.nft_contract_address:
                print("⚠️ No NFT contract deployed, using mock minting")
                return self._mock_mint_nft(metadata, owner_address)
            
            contract = self.w3.eth.contract(
                address=self.nft_contract_address,
                abi=self.nft_abi
            )
            
            # Create metadata URI (would upload to IPFS)
            metadata_uri = f"ipfs://metadata/{metadata.dataset_id}.json"
            
            # Build transaction
            txn = contract.functions.mint(
                owner_address, metadata_uri
            ).build_transaction({
                'from': self.account.address,
                'gas': 200000,
                'gasPrice': self.w3.to_wei('20', 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed_txn = self.account.sign_transaction(txn)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            print(f"🎯 NFT minted successfully!")
            print(f"📋 Transaction hash: {receipt.transactionHash.hex()}")
            
            return receipt.transactionHash.hex()
            
        except Exception as e:
            print(f"⚠️ NFT minting failed: {str(e)[:100]}...")
            return self._mock_mint_nft(metadata, owner_address)
    
    def create_data_access_contract(self, dataset_id: str, price: float, duration: int) -> str:
        """Create smart contract for data access"""
        if not self.connected:
            return self._mock_create_contract(dataset_id, price, duration)
            
        try:
            if not self.marketplace_contract_address:
                print("⚠️ No marketplace contract deployed, using mock contract creation")
                return self._mock_create_contract(dataset_id, price, duration)
            
            contract = self.w3.eth.contract(
                address=self.marketplace_contract_address,
                abi=self.marketplace_abi
            )
            
            price_wei = self.w3.to_wei(price, 'ether')
            
            txn = contract.functions.createAccessContract(
                dataset_id, price_wei, duration
            ).build_transaction({
                'from': self.account.address,
                'gas': 150000,
                'gasPrice': self.w3.to_wei('20', 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            signed_txn = self.account.sign_transaction(txn)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            print(f"📜 Access contract created!")
            print(f"📋 Transaction hash: {receipt.transactionHash.hex()}")
            
            return receipt.transactionHash.hex()
            
        except Exception as e:
            print(f"⚠️ Contract creation failed: {str(e)[:100]}...")
            return self._mock_create_contract(dataset_id, price, duration)
    
    def _mock_mint_nft(self, metadata: DatasetMetadata, owner_address: str) -> str:
        """Mock NFT minting for demo"""
        token_id = hashlib.sha256(f"{metadata.dataset_id}{owner_address}".encode()).hexdigest()[:16]
        print(f"🎯 Mock NFT minted for dataset {metadata.dataset_id}")
        print(f"🏷️ Token ID: {token_id}")
        print(f"👤 Owner: {owner_address}")
        return token_id
    
    def _mock_create_contract(self, dataset_id: str, price: float, duration: int) -> str:
        """Mock contract creation for demo"""
        contract_id = hashlib.sha256(f"{dataset_id}{price}{duration}".encode()).hexdigest()[:16]
        print(f"📜 Mock access contract created for dataset {dataset_id}")
        print(f"🆔 Contract ID: {contract_id}")
        print(f"💰 Price: {price:.6f} ETH")
        print(f"⏰ Duration: {duration} days")
        return contract_id

class DataMarketplace:
    """Main marketplace orchestrator"""
    
    def __init__(self, ipfs_client: PinataIPFSClient, ai_curator: AIDataCurator, 
                 smart_contract: Web3SmartContract):
        self.ipfs_client = ipfs_client
        self.ai_curator = ai_curator
        self.smart_contract = smart_contract
        self.datasets: Dict[str, DatasetMetadata] = {}
        self.marketplace_db = "marketplace_data.json"
        self._load_marketplace_data()
    
    def upload_dataset(self, file_path: str, title: str, description: str, 
                      owner_address: str) -> str:
        """Upload and process a new dataset"""
        
        print(f"🚀 Processing dataset: {title}")
        print("=" * 60)
        
        try:
            # Upload to IPFS with metadata
            print("📤 Uploading to IPFS...")
            metadata = {
                "name": title,
                "description": description,
                "owner": owner_address,
                "uploadDate": datetime.now().isoformat()
            }
            ipfs_hash = self.ipfs_client.add_file(file_path, metadata)
            
            # AI analysis
            print("🤖 Running AI analysis...")
            tags, quality, price, data_type, schema = self.ai_curator.analyze_dataset(file_path)
            
            # Create metadata
            dataset_id = hashlib.sha256(f"{title}{owner_address}{datetime.now()}".encode()).hexdigest()[:16]
            
            # Get file size safely
            try:
                file_size = Path(file_path).stat().st_size if os.path.exists(file_path) else 0
            except Exception:
                file_size = 0
            
            dataset_metadata = DatasetMetadata(
                dataset_id=dataset_id,
                title=title,
                description=description,
                owner_address=owner_address,
                ipfs_hash=ipfs_hash,
                file_size=file_size,
                upload_timestamp=datetime.now(),
                tags=tags,
                quality_score=quality.overall_score,
                suggested_price=price,
                data_type=data_type,
                schema=schema
            )
            
            # Mint NFT
            print("🎯 Minting data NFT...")
            token_id = self.smart_contract.mint_data_nft(dataset_metadata, owner_address)
            
            # Store in marketplace
            self.datasets[dataset_id] = dataset_metadata
            self._save_marketplace_data()
            
            print("\n🎉 Dataset successfully uploaded!")
            print("=" * 60)
            print(f"🆔 Dataset ID: {dataset_id}")
            print(f"🗂️ IPFS Hash: {ipfs_hash}")
            print(f"⭐ Quality Score: {quality.overall_score:.2f}/100")
            print(f"💰 Suggested Price: {price:.6f} ETH")
            print(f"🏷️ Tags: {', '.join(tags)}")
            print(f"📊 Data Type: {data_type}")
            print(f"📦 File Size: {file_size:,} bytes")
            
            return dataset_id
            
        except Exception as e:
            print(f"⚠️ Error uploading dataset: {str(e)[:100]}...")
            return ""
    
    def search_datasets(self, query: str = "", tags: List[str] = None, 
                       min_quality: float = 0, max_price: float = float('inf')) -> List[DatasetMetadata]:
        """Search datasets based on criteria"""
        results = []
        
        try:
            for dataset in self.datasets.values():
                # Text search
                if query:
                    search_text = (dataset.title + " " + dataset.description).lower()
                    if query.lower() not in search_text:
                        continue
                
                # Tag filter
                if tags and not any(tag.lower() in [t.lower() for t in dataset.tags] for tag in tags):
                    continue
                
                # Quality filter
                if dataset.quality_score < min_quality:
                    continue
                
                # Price filter
                if dataset.suggested_price > max_price:
                    continue
                
                results.append(dataset)
            
            return sorted(results, key=lambda x: x.quality_score, reverse=True)
        except Exception as e:
            print(f"⚠️ Error searching datasets: {str(e)[:100]}...")
            return []
    
    def purchase_data_access(self, dataset_id: str, buyer_address: str, duration_days: int = 30) -> bool:
        """Purchase access to a dataset"""
        try:
            if dataset_id not in self.datasets:
                print("⚠️ Dataset not found!")
                return False
            
            dataset = self.datasets[dataset_id]
            
            print(f"\n💳 Processing purchase...")
            print("=" * 40)
            print(f"📊 Dataset: {dataset.title}")
            print(f"💰 Price: {dataset.suggested_price:.6f} ETH")
            print(f"👤 Buyer: {buyer_address}")
            print(f"⏰ Duration: {duration_days} days")
            
            # Create access contract
            contract_id = self.smart_contract.create_data_access_contract(
                dataset_id, dataset.suggested_price, duration_days
            )
            
            # Update access count
            dataset.access_count += 1
            self._save_marketplace_data()
            
            print(f"\n✅ Access granted successfully!")
            print(f"🆔 Access contract: {contract_id}")
            print(f"🗂️ You can now download from: {dataset.ipfs_hash}")
            
            return True
        except Exception as e:
            print(f"⚠️ Error processing purchase: {str(e)[:100]}...")
            return False
    
    def get_dataset_info(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get detailed information about a dataset"""
        return self.datasets.get(dataset_id)
    
    def _load_marketplace_data(self):
        """Load marketplace data from storage"""
        try:
            if os.path.exists(self.marketplace_db):
                with open(self.marketplace_db, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for dataset_data in data.values():
                        # Convert timestamp string back to datetime
                        try:
                            dataset_data['upload_timestamp'] = datetime.fromisoformat(
                                dataset_data['upload_timestamp']
                            )
                        except Exception:
                            dataset_data['upload_timestamp'] = datetime.now()
                        
                        try:
                            metadata = DatasetMetadata(**dataset_data)
                            self.datasets[metadata.dataset_id] = metadata
                        except Exception as e:
                            print(f"⚠️ Error loading dataset: {str(e)[:50]}...")
                            continue
        except Exception as e:
            print(f"⚠️ Error loading marketplace data: {str(e)[:100]}...")
    
    def _save_marketplace_data(self):
        """Save marketplace data to storage"""
        try:
            data_to_save = {}
            for dataset_id, metadata in self.datasets.items():
                try:
                    metadata_dict = asdict(metadata)
                    # Convert datetime to string for JSON serialization
                    metadata_dict['upload_timestamp'] = metadata.upload_timestamp.isoformat()
                    data_to_save[dataset_id] = metadata_dict
                except Exception as e:
                    print(f"⚠️ Error serializing dataset {dataset_id}: {str(e)[:50]}...")
                    continue
            
            with open(self.marketplace_db, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error saving marketplace data: {str(e)[:100]}...")

# Demo usage with production-ready components
def demo_marketplace():
    """Demonstrate marketplace functionality"""
    
    print("🌍 DECENTRALIZED AI-CURATED DATA MARKETPLACE")
    print("=" * 60)
    print("🚀 Initializing components...\n")
    
    try:
        # Configuration - replace with your actual credentials
        PINATA_API_KEY = os.getenv("PINATA_API_KEY", "your_pinata_api_key")
        PINATA_SECRET = os.getenv("PINATA_SECRET", "your_pinata_secret")
        RPC_URL = os.getenv("RPC_URL", "https://sepolia.infura.io/v3/your_project_id")
        PRIVATE_KEY = os.getenv("PRIVATE_KEY", "0x" + "0" * 64)  # Demo key
        
        # Initialize components
        print("📡 Connecting to IPFS (Pinata)...")
        ipfs_client = PinataIPFSClient(PINATA_API_KEY, PINATA_SECRET)
        
        print("🤖 Initializing AI curator...")
        ai_curator = AIDataCurator()
        
        print("🔗 Connecting to blockchain...")
        smart_contract = Web3SmartContract(RPC_URL, PRIVATE_KEY)
        
        print("🏪 Starting marketplace...")
        marketplace = DataMarketplace(ipfs_client, ai_curator, smart_contract)
        
        print("\n" + "=" * 60)
        print("📊 CREATING SAMPLE DATASET")
        print("=" * 60)
        
        # Create sample dataset with realistic e-commerce data
        print("🔧 Generating sample e-commerce data...")
        
        # Create sample data with proper error handling
        np.random.seed(42)  # For reproducible results
        n_samples = 1000
        
        sample_data = pd.DataFrame({
            'user_id': range(n_samples),
            'purchase_amount': np.round(np.random.uniform(10, 1000, n_samples), 2),
            'purchase_date': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'category': np.random.choice(['electronics', 'clothing', 'books', 'food'], n_samples),
            'location': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
            'user_age': np.random.randint(18, 80, n_samples),
            'payment_method': np.random.choice(['credit', 'debit', 'paypal', 'crypto'], n_samples)
        })
        
        sample_file = 'sample_ecommerce_data.csv'
        
        try:
            sample_data.to_csv(sample_file, index=False)
            print(f"✅ Sample data created: {sample_file}")
        except Exception as e:
            print(f"⚠️ Error creating sample file: {str(e)[:100]}...")
            return None
        
        # Upload dataset
        dataset_id = marketplace.upload_dataset(
            file_path=sample_file,
            title="Premium E-commerce Transaction Dataset",
            description="Comprehensive e-commerce transaction data with user demographics, purchase patterns, and location insights. Perfect for ML models and business analytics.",
            owner_address="0xDataProvider123456789abcdef"
        )
        
        if not dataset_id:
            print("⚠️ Failed to upload dataset")
            return marketplace
        
        print(f"\n" + "=" * 60)
        print("🔍 MARKETPLACE SEARCH & DISCOVERY")
        print("=" * 60)
        
        # Search datasets
        print("🔎 Searching for 'ecommerce' datasets with quality > 70...")
        results = marketplace.search_datasets(query="ecommerce", min_quality=70)
        
        print(f"\n📋 Found {len(results)} matching datasets:")
        print("-" * 50)
        
        for i, dataset in enumerate(results, 1):
            print(f"\n{i}. 📊 {dataset.title}")
            print(f"   ⭐ Quality Score: {dataset.quality_score:.1f}/100")
            print(f"   💰 Price: {dataset.suggested_price:.6f} ETH")
            print(f"   🏷️ Tags: {', '.join(dataset.tags)}")
            print(f"   📦 Size: {dataset.file_size:,} bytes")
            print(f"   👥 Access Count: {dataset.access_count}")
            print(f"   🗂️ IPFS: {dataset.ipfs_hash[:20]}...")
        
        # Purchase access
        print(f"\n" + "=" * 60)
        print("💳 SIMULATING DATA PURCHASE")
        print("=" * 60)
        
        success = marketplace.purchase_data_access(
            dataset_id, 
            "0xBuyer456789abcdef123456", 
            30
        )
        
        print(f"\n" + "=" * 60)
        print("📈 MARKETPLACE STATISTICS")
        print("=" * 60)
        
        if marketplace.datasets:
            total_datasets = len(marketplace.datasets)
            total_value = sum(d.suggested_price for d in marketplace.datasets.values())
            avg_quality = np.mean([d.quality_score for d in marketplace.datasets.values()])
            total_accesses = sum(d.access_count for d in marketplace.datasets.values())
            
            print(f"📊 Total Datasets: {total_datasets}")
            print(f"💰 Total Market Value: {total_value:.6f} ETH")
            print(f"⭐ Average Quality Score: {avg_quality:.1f}/100")
            print(f"👥 Total Data Accesses: {total_accesses}")
        else:
            print("⚠️ No datasets found in marketplace")
        
        # Show detailed dataset info
        print(f"\n" + "=" * 60)
        print("🔍 DETAILED DATASET INFORMATION")
        print("=" * 60)
        
        dataset_info = marketplace.get_dataset_info(dataset_id)
        if dataset_info:
            print(f"🆔 Dataset ID: {dataset_info.dataset_id}")
            print(f"📊 Title: {dataset_info.title}")
            print(f"📝 Description: {dataset_info.description}")
            print(f"👤 Owner: {dataset_info.owner_address}")
            print(f"🗂️ IPFS Hash: {dataset_info.ipfs_hash}")
            print(f"📅 Upload Date: {dataset_info.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📦 File Size: {dataset_info.file_size:,} bytes")
            print(f"📊 Data Type: {dataset_info.data_type}")
            print(f"🏷️ Tags: {', '.join(dataset_info.tags)}")
            print(f"⭐ Quality Score: {dataset_info.quality_score:.2f}/100")
            print(f"💰 Price: {dataset_info.suggested_price:.6f} ETH")
            print(f"👥 Access Count: {dataset_info.access_count}")
            
            print(f"\n📋 Schema Information:")
            print(f"   📏 Rows: {dataset_info.schema.get('row_count', 'N/A'):,}")
            print(f"   📊 Columns: {dataset_info.schema.get('column_count', 'N/A')}")
            
            if 'columns' in dataset_info.schema and dataset_info.schema['columns']:
                print(f"   📤 Column Details:")
                for col_name, col_info in list(dataset_info.schema['columns'].items())[:3]:
                    unique_count = col_info.get('unique_count', 0)
                    col_type = col_info.get('type', 'unknown')
                    print(f"      • {col_name}: {col_type} ({unique_count} unique)")
                
                if len(dataset_info.schema['columns']) > 3:
                    remaining = len(dataset_info.schema['columns']) - 3
                    print(f"      ... and {remaining} more columns")
        
        # Cleanup
        try:
            if os.path.exists(sample_file):
                os.remove(sample_file)
                print(f"🗑️ Cleaned up sample file: {sample_file}")
        except Exception as e:
            print(f"⚠️ Could not clean up sample file: {str(e)[:50]}...")
        
        print(f"\n🎉 Demo completed successfully!")
        print("=" * 60)
        
        return marketplace
    
    except Exception as e:
        print(f"⚠️ Demo failed with error: {str(e)[:200]}...")
        print("🔧 This is likely due to missing dependencies or configuration issues")
        print("📝 Make sure you have installed: pandas, numpy, web3, requests")
        return None


# Additional utility functions for better error handling
def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['pandas', 'numpy', 'web3', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ Missing required packages: {', '.join(missing_packages)}")
        print(f"📦 Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def safe_demo():
    """Run demo with comprehensive error handling"""
    print("🔍 Checking dependencies...")
    
    if not check_dependencies():
        return None
    
    try:
        return demo_marketplace()
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
        return None
    except Exception as e:
        print(f"\n⚠️ Unexpected error: {str(e)}")
        print("🐛 This might be a configuration or environment issue")
        return None


if __name__ == "__main__":
    marketplace = safe_demo()