from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import hashlib
import uuid

from .models import get_db
from .database import TransactionDB, BlockchainDB, AuditDB
from .models import Transaction, BlockchainSimulation

class BlockchainSimulator:
    """Simulates Hyperledger Fabric functionality for AURA ledger"""
    
    def __init__(self):
        self.transaction_db = TransactionDB()
        self.blockchain_db = BlockchainDB()
        self.audit_db = AuditDB()
        self.chain = []  # In-memory chain for simulation
        self.pending_transactions = []  # Pending transactions
        self.block_size_limit = 10  # Max transactions per block
    
    def log_decision(self, 
                    hospital_id: int, 
                    update_hash: str, 
                    verdict: str,
                    evidence_hash: Optional[str] = None,
                    anomaly_score: Optional[float] = None,
                    normalized_score: Optional[float] = None,
                    confidence: Optional[float] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    db: Any = None) -> Dict[str, Any]:
        """Log a decision to the ledger"""
        try:
            # Create transaction record
            transaction = self.transaction_db.create_transaction(
                db, hospital_id, update_hash, verdict, 
                evidence_hash, anomaly_score, normalized_score, confidence, metadata
            )
            
            # Add to pending transactions
            self.pending_transactions.append(transaction.tx_id)
            
            # Try to create a block if we have enough pending transactions
            if len(self.pending_transactions) >= self.block_size_limit:
                self._create_block(db)
            
            result = {
                'transaction_id': transaction.tx_id,
                'hospital_id': transaction.hospital_id,
                'update_hash': transaction.update_hash,
                'verdict': transaction.verdict,
                'evidence_hash': transaction.evidence_hash,
                'anomaly_score': transaction.anomaly_score,
                'timestamp': transaction.timestamp.isoformat(),
                'status': 'logged'
            }
            
            print(f"Decision logged: {transaction.tx_id} - {transaction.verdict}")
            return result
            
        except Exception as e:
            print(f"Error logging decision: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error logging decision: {str(e)}")
    
    def _create_block(self, db: Any):
        """Create a new block with pending transactions"""
        if not self.pending_transactions:
            return
        
        # Get previous block hash
        previous_block = self.blockchain_db.get_latest_block(db)
        previous_hash = previous_block.block_hash if previous_block else "GENESIS"
        
        # Create block with current pending transactions
        current_transactions = self.pending_transactions[:self.block_size_limit]
        remaining_transactions = self.pending_transactions[self.block_size_limit:]
        
        # Create block
        block = self.blockchain_db.create_block(db, current_transactions, previous_hash)
        
        # Update pending transactions
        self.pending_transactions = remaining_transactions
        
        print(f"Block created: {block.block_hash} with {len(current_transactions)} transactions")
    
    def get_transaction_history(self, hospital_id: int, db: Any) -> List[Dict[str, Any]]:
        """Get transaction history for a hospital"""
        transactions = self.transaction_db.get_transactions_by_hospital(db, hospital_id)
        
        history = []
        for tx in transactions:
            history.append({
                'transaction_id': tx.tx_id,
                'hospital_id': tx.hospital_id,
                'update_hash': tx.update_hash,
                'verdict': tx.verdict,
                'evidence_hash': tx.evidence_hash,
                'anomaly_score': tx.anomaly_score,
                'normalized_score': tx.normalized_score,
                'confidence': tx.confidence,
                'timestamp': tx.timestamp.isoformat(),
                'metadata': json.loads(tx.meta_data) if tx.metadata else None
            })
        
        return sorted(history, key=lambda x: x['timestamp'], reverse=True)
    
    def get_verification_stats(self, db: Any) -> Dict[str, Any]:
        """Get verification statistics"""
        stats = self.transaction_db.get_verification_stats(db)
        
        # Add blockchain info
        latest_block = self.blockchain_db.get_latest_block(db)
        total_blocks = len(self.blockchain_db.get_all_blocks(db))
        
        stats['latest_block_hash'] = latest_block.block_hash if latest_block else None
        stats['total_blocks'] = total_blocks
        stats['pending_transactions'] = len(self.pending_transactions)
        
        return stats
    
    def verify_transaction(self, tx_id: str, db: Any) -> Dict[str, Any]:
        """Verify a transaction exists and is valid"""
        transaction = self.transaction_db.get_transaction_by_id(db, tx_id)
        
        if not transaction:
            return {'valid': False, 'error': 'Transaction not found'}
        
        # Verify evidence hash exists and is valid (if present)
        if transaction.evidence_hash:
            # In a real system, we would verify the hash against the stored evidence
            # For simulation, we assume it's valid if it exists
            evidence_valid = len(transaction.evidence_hash) == 64  # SHA256 length
        else:
            evidence_valid = True
        
        return {
            'valid': True,
            'transaction': {
                'transaction_id': transaction.tx_id,
                'hospital_id': transaction.hospital_id,
                'update_hash': transaction.update_hash,
                'verdict': transaction.verdict,
                'evidence_hash': transaction.evidence_hash,
                'anomaly_score': transaction.anomaly_score,
                'timestamp': transaction.timestamp.isoformat(),
                'evidence_valid': evidence_valid
            }
        }
    
    def get_chain_info(self, db: Any) -> Dict[str, Any]:
        """Get blockchain information"""
        all_blocks = self.blockchain_db.get_all_blocks(db)
        
        blocks_info = []
        for block in all_blocks:
            block_info = {
                'block_hash': block.block_hash,
                'previous_hash': block.previous_hash,
                'transaction_count': block.transaction_count,
                'transactions': json.loads(block.transactions) if block.transactions else [],
                'timestamp': block.timestamp.isoformat(),
                'difficulty': block.difficulty
            }
            blocks_info.append(block_info)
        
        return {
            'total_blocks': len(blocks_info),
            'total_transactions': sum(block['transaction_count'] for block in blocks_info),
            'blocks': blocks_info,
            'pending_transactions': len(self.pending_transactions)
        }

# Global instance
blockchain_simulator = BlockchainSimulator()