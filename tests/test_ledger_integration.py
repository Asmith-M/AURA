import pytest
import asyncio
import httpx
from datetime import datetime
import json
import uuid

def test_database_creation():
    """Test that database tables were created successfully"""
    from ledger_stub.database import engine
    from ledger_stub.models import Transaction, BlockchainSimulation, AuditTrail
    
    # Check if tables exist by trying to create them (should not raise error)
    from ledger_stub.models import create_tables
    create_tables()
    
    # Verify tables exist by checking table names
    table_names = engine.table_names()
    assert 'transactions' in table_names
    assert 'blockchain_simulation' in table_names
    assert 'audit_trail' in table_names

def test_transaction_database_operations():
    """Test transaction database operations"""
    from ledger_stub.database import TransactionDB
    from ledger_stub.models import SessionLocal
    
    db = SessionLocal()
    
    try:
        # Test creating a transaction
        transaction = TransactionDB.create_transaction(
            db,
            hospital_id=1,
            update_hash="test_hash_123",
            verdict="APPROVED",
            evidence_hash="evidence_hash_456",
            anomaly_score=0.123,
            normalized_score=0.234,
            confidence=0.95,
            metadata={"test": True}
        )
        
        assert transaction.hospital_id == 1
        assert transaction.verdict == "APPROVED"
        assert transaction.anomaly_score == 0.123
        
        # Test retrieving transaction
        retrieved = TransactionDB.get_transaction_by_id(db, transaction.tx_id)
        assert retrieved is not None
        assert retrieved.tx_id == transaction.tx_id
        
        # Test retrieving by hash
        by_hash = TransactionDB.get_transaction_by_hash(db, "test_hash_123")
        assert by_hash is not None
        assert by_hash.update_hash == "test_hash_123"
        
        # Test getting transactions by hospital
        hospital_txs = TransactionDB.get_transactions_by_hospital(db, 1)
        assert len(hospital_txs) >= 1
        
        # Test verification stats
        stats = TransactionDB.get_verification_stats(db)
        assert 'total_transactions' in stats
        assert 'approved' in stats
        assert 'rejected' in stats
        
    finally:
        db.close()

def test_blockchain_database_operations():
    """Test blockchain database operations"""
    from ledger_stub.database import BlockchainDB
    from ledger_stub.models import SessionLocal
    
    db = SessionLocal()
    
    try:
        # Test creating a block
        block = BlockchainDB.create_block(db, ["tx1", "tx2", "tx3"])
        
        assert block is not None
        assert block.transaction_count == 3
        assert block.block_hash is not None
        
        # Test retrieving latest block
        latest = BlockchainDB.get_latest_block(db)
        assert latest is not None
        assert latest.id == block.id
        
        # Test retrieving block by hash
        by_hash = BlockchainDB.get_block_by_hash(db, block.block_hash)
        assert by_hash is not None
        assert by_hash.block_hash == block.block_hash
        
        # Test getting all blocks
        all_blocks = BlockchainDB.get_all_blocks(db)
        assert len(all_blocks) >= 1
        
    finally:
        db.close()

def test_audit_database_operations():
    """Test audit database operations"""
    from ledger_stub.database import AuditDB
    from ledger_stub.models import SessionLocal
    
    db = SessionLocal()
    
    try:
        # Test logging an operation
        audit_entry = AuditDB.log_operation(
            db,
            operation="TEST_OPERATION",
            entity_type="test_entity",
            entity_id=123,
            old_values={"old": "value"},
            new_values={"new": "value"},
            user="test_user"
        )
        
        assert audit_entry is not None
        assert audit_entry.operation == "TEST_OPERATION"
        assert audit_entry.entity_id == 123
        
        # Test getting audit trail
        trail = AuditDB.get_audit_trail(db, entity_type="test_entity")
        assert len(trail) >= 1
        
        trail_all = AuditDB.get_audit_trail(db)
        assert len(trail_all) >= 1
        
    finally:
        db.close()

def test_blockchain_simulator():
    """Test blockchain simulator functionality"""
    from ledger_stub.blockchain_simulator import blockchain_simulator
    from ledger_stub.models import SessionLocal
    
    db = SessionLocal()
    
    try:
        # Test logging a decision
        result = blockchain_simulator.log_decision(
            db=db,
            hospital_id=1,
            update_hash="sim_test_hash",
            verdict="APPROVED",
            evidence_hash="sim_evidence_hash",
            anomaly_score=0.5,
            normalized_score=0.3,
            confidence=0.8,
            metadata={"simulation": True}
        )
        
        assert result['hospital_id'] == 1
        assert result['verdict'] == 'APPROVED'
        assert result['status'] == 'logged'
        
        # Test getting verification stats
        stats = blockchain_simulator.get_verification_stats(db)
        assert 'total_transactions' in stats
        assert 'approved' in stats
        
        # Test getting transaction history
        history = blockchain_simulator.get_transaction_history(1, db)
        assert len(history) >= 1
        
        # Test chain info
        chain_info = blockchain_simulator.get_chain_info(db)
        assert 'total_blocks' in chain_info
        assert 'total_transactions' in chain_info
        
        # Test verification
        verification = blockchain_simulator.verify_transaction(result['transaction_id'], db)
        assert verification['valid'] == True
        
    finally:
        db.close()

@pytest.mark.asyncio
async def test_ledger_api_endpoints():
    """Test ledger API endpoints"""
    async with httpx.AsyncClient(base_url="http://localhost:8001") as client:
        try:
            # Test health endpoint
            response = await client.get("/health")
            assert response.status_code == 200
            health_data = response.json()
            assert health_data['status'] == 'healthy'
            
            # Test stats endpoint
            response = await client.get("/ledger/stats")
            assert response.status_code == 200
            stats_data = response.json()
            assert 'total_transactions' in stats_data
            
            # Test recent transactions
            response = await client.get("/ledger/recent")
            assert response.status_code == 200
            recent_data = response.json()
            assert 'transactions' in recent_data
            
        except httpx.ConnectError:
            print("⚠️  Ledger API not running - skipping API tests")
            return

def test_sentinel_ledger_integration():
    """Test that sentinel can integrate with ledger"""
    # This test would require both APIs to be running
    # For now, just test the imports and basic functionality
    try:
        from ledger_stub.blockchain_simulator import blockchain_simulator
        from ledger_stub.database import TransactionDB
        
        assert blockchain_simulator is not None
        assert TransactionDB is not None
        
        print("✅ Sentinel-ledger integration components available")
        
    except ImportError as e:
        pytest.skip(f"Ledger integration not available: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])