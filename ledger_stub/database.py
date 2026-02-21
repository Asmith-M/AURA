from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import json
import hashlib

from .models import Transaction, BlockchainSimulation, AuditTrail, SessionLocal
from .models import get_db as get_session

class TransactionDB:
    """Database operations for transactions"""
    
    @staticmethod
    def create_transaction(db: Session, 
                          hospital_id: int, 
                          update_hash: str, 
                          verdict: str,
                          evidence_hash: Optional[str] = None,
                          anomaly_score: Optional[float] = None,
                          normalized_score: Optional[float] = None,
                          confidence: Optional[float] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Transaction:
        """Create a new transaction record"""
        tx_id = f"TX_{uuid.uuid4().hex[:12].upper()}"
        
        transaction = Transaction(
            tx_id=tx_id,
            hospital_id=hospital_id,
            update_hash=update_hash,
            verdict=verdict.upper(),
            evidence_hash=evidence_hash,
            anomaly_score=anomaly_score,
            normalized_score=normalized_score,
            confidence=confidence,
            meta_data=json.dumps(metadata) if metadata else None
        )
        
        db.add(transaction)
        try:
            db.commit()
            db.refresh(transaction)
            
            # Add to audit trail
            audit_entry = AuditTrail(
                operation="CREATE",
                entity_type="transaction",
                entity_id=transaction.id,
                new_values=json.dumps({
                    'tx_id': transaction.tx_id,
                    'hospital_id': transaction.hospital_id,
                    'verdict': transaction.verdict
                })
            )
            db.add(audit_entry)
            db.commit()
            
            return transaction
        except IntegrityError:
            db.rollback()
            raise ValueError(f"Transaction with hash {update_hash} already exists")
    
    @staticmethod
    def get_transaction_by_id(db: Session, tx_id: str) -> Optional[Transaction]:
        """Get transaction by ID"""
        return db.query(Transaction).filter(Transaction.tx_id == tx_id).first()
    
    @staticmethod
    def get_transaction_by_hash(db: Session, update_hash: str) -> Optional[Transaction]:
        """Get transaction by update hash"""
        return db.query(Transaction).filter(Transaction.update_hash == update_hash).first()
    
    @staticmethod
    def get_transactions_by_hospital(db: Session, hospital_id: int) -> List[Transaction]:
        """Get all transactions for a hospital"""
        return db.query(Transaction).filter(Transaction.hospital_id == hospital_id).all()
    
    @staticmethod
    def get_recent_transactions(db: Session, limit: int = 100) -> List[Transaction]:
        """Get recent transactions"""
        return db.query(Transaction).order_by(Transaction.timestamp.desc()).limit(limit).all()
    
    @staticmethod
    def get_verification_stats(db: Session) -> Dict[str, int]:
        """Get verification statistics"""
        total = db.query(Transaction).count()
        approved = db.query(Transaction).filter(Transaction.verdict == "APPROVED").count()
        rejected = db.query(Transaction).filter(Transaction.verdict == "REJECTED").count()
        errors = db.query(Transaction).filter(Transaction.verdict == "ERROR").count()
        
        return {
            'total_transactions': total,
            'approved': approved,
            'rejected': rejected,
            'errors': errors,
            'approval_rate': approved / total if total > 0 else 0,
            'rejection_rate': rejected / total if total > 0 else 0
        }
    
    @staticmethod
    def search_transactions(db: Session, 
                           hospital_id: Optional[int] = None,
                           verdict: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           limit: int = 100) -> List[Transaction]:
        """Search transactions with filters"""
        query = db.query(Transaction)
        
        if hospital_id:
            query = query.filter(Transaction.hospital_id == hospital_id)
        if verdict:
            query = query.filter(Transaction.verdict == verdict.upper())
        if start_date:
            query = query.filter(Transaction.timestamp >= start_date)
        if end_date:
            query = query.filter(Transaction.timestamp <= end_date)
        
        return query.order_by(Transaction.timestamp.desc()).limit(limit).all()

class BlockchainDB:
    """Database operations for blockchain simulation"""
    
    @staticmethod
    def create_block(db: Session, 
                    transactions: List[str], 
                    previous_hash: str = None) -> BlockchainSimulation:
        """Create a new block in the blockchain"""
        block_hash = hashlib.sha256(
            f"{previous_hash}{str(transactions)}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()
        
        block = BlockchainSimulation(
            block_hash=block_hash,
            previous_hash=previous_hash,
            transaction_count=len(transactions),
            transactions=json.dumps(transactions),
            nonce=0,  # Will be computed in real blockchain
            difficulty=4
        )
        
        db.add(block)
        db.commit()
        db.refresh(block)
        
        # Add to audit trail
        audit_entry = AuditTrail(
            operation="CREATE_BLOCK",
            entity_type="block",
            entity_id=block.id,
            new_values=json.dumps({
                'block_hash': block.block_hash,
                'transaction_count': block.transaction_count
            })
        )
        db.add(audit_entry)
        db.commit()
        
        return block
    
    @staticmethod
    def get_latest_block(db: Session) -> Optional[BlockchainSimulation]:
        """Get the latest block in the chain"""
        return db.query(BlockchainSimulation).order_by(BlockchainSimulation.timestamp.desc()).first()
    
    @staticmethod
    def get_block_by_hash(db: Session, block_hash: str) -> Optional[BlockchainSimulation]:
        """Get block by hash"""
        return db.query(BlockchainSimulation).filter(BlockchainSimulation.block_hash == block_hash).first()
    
    @staticmethod
    def get_all_blocks(db: Session) -> List[BlockchainSimulation]:
        """Get all blocks in chronological order"""
        return db.query(BlockchainSimulation).order_by(BlockchainSimulation.timestamp.asc()).all()

class AuditDB:
    """Database operations for audit trail"""
    
    @staticmethod
    def log_operation(db: Session, 
                     operation: str, 
                     entity_type: str, 
                     entity_id: int,
                     old_values: Optional[Dict[str, Any]] = None,
                     new_values: Optional[Dict[str, Any]] = None,
                     user: str = "system") -> AuditTrail:
        """Log an operation in the audit trail"""
        audit_entry = AuditTrail(
            operation=operation.upper(),
            entity_type=entity_type.lower(),
            entity_id=entity_id,
            old_values=json.dumps(old_values) if old_values else None,
            new_values=json.dumps(new_values) if new_values else None,
            user=user
        )
        
        db.add(audit_entry)
        db.commit()
        db.refresh(audit_entry)
        
        return audit_entry
    
    @staticmethod
    def get_audit_trail(db: Session, 
                       entity_type: Optional[str] = None,
                       operation: Optional[str] = None,
                       limit: int = 100) -> List[AuditTrail]:
        """Get audit trail entries with optional filters"""
        query = db.query(AuditTrail)
        
        if entity_type:
            query = query.filter(AuditTrail.entity_type == entity_type.lower())
        if operation:
            query = query.filter(AuditTrail.operation == operation.upper())
        
        return query.order_by(AuditTrail.timestamp.desc()).limit(limit).all()

def get_db_session():
    """Convenience function to get database session"""
    return SessionLocal()

# Global instances
transaction_db = TransactionDB()
blockchain_db = BlockchainDB()
audit_db = AuditDB()