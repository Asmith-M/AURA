import asyncio
import httpx
import json
from datetime import datetime
import uuid

async def test_ledger_integration():
    """Test the ledger integration with the sentinel system"""
    print("Testing Ledger Integration...")
    
    # Test ledger health
    print("\n1. Testing ledger health...")
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8001/health")
        print(f"   Ledger health: {response.status_code} - {response.json()}")
    
    # Test logging a decision
    print("\n2. Testing decision logging...")
    test_data = {
        'hospital_id': 1,
        'update_hash': 'a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890',
        'verdict': 'APPROVED',
        'evidence_hash': 'f0e9d8c7b6a543210fedcba9876543210fedcba9876543210fedcba987654321',
        'anomaly_score': 0.123,
        'normalized_score': 0.234,
        'confidence': 0.95,
        'metadata': {
            'test': True,
            'timestamp': datetime.now().isoformat(),
            'test_id': str(uuid.uuid4())
        }
    }
    
    try:
        response = await client.post("http://localhost:8001/ledger/log_verdict", json=test_data)
        print(f"   Log verdict: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Transaction ID: {result['transaction_id']}")
            tx_id = result['transaction_id']
        else:
            print(f"   Error: {response.text}")
            return
    except Exception as e:
        print(f"   Error logging verdict: {str(e)}")
        return
    
    # Test getting transaction
    print(f"\n3. Testing transaction retrieval...")
    try:
        response = await client.get(f"http://localhost:8001/ledger/transaction/{tx_id}")
        print(f"   Get transaction: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Retrieved: {result['verdict']} for hospital {result['hospital_id']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error getting transaction: {str(e)}")
    
    # Test verification
    print(f"\n4. Testing transaction verification...")
    try:
        response = await client.get(f"http://localhost:8001/ledger/verify/{tx_id}")
        print(f"   Verify transaction: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Valid: {result['valid']}")
            if result['valid']:
                print(f"   Transaction details retrieved successfully")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error verifying transaction: {str(e)}")
    
    # Test hospital history
    print(f"\n5. Testing hospital history...")
    try:
        response = await client.get("http://localhost:8001/ledger/history/1")
        print(f"   Hospital history: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Transactions for hospital 1: {result['total_count']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error getting history: {str(e)}")
    
    # Test statistics
    print(f"\n6. Testing statistics...")
    try:
        response = await client.get("http://localhost:8001/ledger/stats")
        print(f"   Stats: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Total transactions: {result['total_transactions']}")
            print(f"   Approved: {result['approved']}")
            print(f"   Rejected: {result['rejected']}")
            print(f"   Approval rate: {result['approval_rate']:.2%}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error getting stats: {str(e)}")
    
    # Test search functionality
    print(f"\n7. Testing search functionality...")
    try:
        response = await client.get("http://localhost:8001/ledger/search?hospital_id=1&verdict=APPROVED")
        print(f"   Search: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Found {result['total_found']} approved transactions for hospital 1")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error searching: {str(e)}")
    
    print("\n✅ Ledger integration test completed!")

async def test_sentinel_ledger_workflow():
    """Test the complete workflow from sentinel to ledger"""
    print("\nTesting Complete Sentinel-Ledger Workflow...")
    
    # First, test that both services are running
    async with httpx.AsyncClient() as client:
        # Check sentinel
        try:
            response = await client.get("http://localhost:8000/sentinel/health")
            print(f"   Sentinel health: {response.status_code}")
        except:
            print("   ⚠️  Sentinel not running - please start sentinel API first")
            return
        
        # Check ledger
        try:
            response = await client.get("http://localhost:8001/health")
            print(f"   Ledger health: {response.status_code}")
        except:
            print("   ⚠️  Ledger not running - please start ledger API first")
            return
    
    print("   Complete workflow test requires both APIs to be running")
    print("   Please start both services before running this test")

if __name__ == "__main__":
    print("="*60)
    print("AURA - Ledger Integration Test")
    print("="*60)
    
    asyncio.run(test_ledger_integration())
    asyncio.run(test_sentinel_ledger_workflow())
    
    print("\n" + "="*60)
    print("Ledger integration testing completed!")
    print("Next steps:")
    print("1. Start ledger API: python -m uvicorn ledger_stub.main:app --host 0.0.0.0 --port 8001")
    print("2. Start sentinel API: python -m uvicorn sentinel.main:app --host 0.0.0.0 --port 8000")
    print("3. Test the complete workflow")
    print("="*60)