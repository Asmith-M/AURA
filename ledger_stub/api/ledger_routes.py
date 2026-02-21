from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

from models import get_db
from database import TransactionDB, BlockchainDB, AuditDB
from blockchain_simulator import blockchain_simulator

router = APIRouter(prefix="/ledger", tags=["ledger"])

@router.post("/log_verdict")
async def log_verdict(
    hospital_id: int,
    update_hash: str,
    verdict: str,
    evidence_hash: Optional[str] = None,
    anomaly_score: Optional[float] = None,
    normalized_score: Optional[float] = None,
    confidence: Optional[float] = None,
    metadata: Optional[Dict] = None,
    db: any = Depends(get_db)
):
    """Log a detection verdict to the ledger"""
    if verdict.upper() not in ["APPROVED", "REJECTED", "ERROR"]:
        raise HTTPException(status_code=400, detail="Invalid verdict. Must be APPROVED, REJECTED, or ERROR")
    
    result = blockchain_simulator.log_decision(
        hospital_id, update_hash, verdict, 
        evidence_hash, anomaly_score, normalized_score, confidence, metadata, db
    )
    
    return result

@router.get("/history/{hospital_id}")
async def get_hospital_history(
    hospital_id: int,
    limit: int = Query(100, ge=1, le=1000),
    db: any = Depends(get_db)
):
    """Get transaction history for a specific hospital"""
    history = blockchain_simulator.get_transaction_history(hospital_id, db)
    
    # Limit results
    limited_history = history[:limit]
    
    return {
        'hospital_id': hospital_id,
        'transactions': limited_history,
        'total_count': len(limited_history)
    }

@router.get("/transaction/{tx_id}")
async def get_transaction(
    tx_id: str,
    db: any = Depends(get_db)
):
    """Get details of a specific transaction"""
    transaction = TransactionDB.get_transaction_by_id(db, tx_id)
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return {
        'transaction_id': transaction.tx_id,
        'hospital_id': transaction.hospital_id,
        'update_hash': transaction.update_hash,
        'verdict': transaction.verdict,
        'evidence_hash': transaction.evidence_hash,
        'anomaly_score': transaction.anomaly_score,
        'normalized_score': transaction.normalized_score,
        'confidence': transaction.confidence,
        'timestamp': transaction.timestamp.isoformat(),
        'metadata': json.loads(transaction.metadata) if transaction.metadata else None
    }

@router.get("/verify/{tx_id}")
async def verify_transaction(
    tx_id: str,
    db: any = Depends(get_db)
):
    """Verify a transaction exists and is valid"""
    verification = blockchain_simulator.verify_transaction(tx_id, db)
    
    if not verification['valid']:
        raise HTTPException(status_code=404, detail=verification.get('error', 'Verification failed'))
    
    return verification

@router.get("/stats")
async def get_verification_stats(
    db: any = Depends(get_db)
):
    """Get verification statistics"""
    stats = blockchain_simulator.get_verification_stats(db)
    
    return stats

@router.get("/chain")
async def get_chain_info(
    db: any = Depends(get_db)
):
    """Get blockchain information"""
    chain_info = blockchain_simulator.get_chain_info(db)
    
    return chain_info

@router.get("/search")
async def search_transactions(
    hospital_id: Optional[int] = Query(None),
    verdict: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: any = Depends(get_db)
):
    """Search transactions with filters"""
    transactions = TransactionDB.search_transactions(
        db, hospital_id, verdict, start_date, end_date, limit
    )
    
    results = []
    for tx in transactions:
        results.append({
            'transaction_id': tx.tx_id,
            'hospital_id': tx.hospital_id,
            'update_hash': tx.update_hash,
            'verdict': tx.verdict,
            'evidence_hash': tx.evidence_hash,
            'anomaly_score': tx.anomaly_score,
            'timestamp': tx.timestamp.isoformat()
        })
    
    return {
        'transactions': results,
        'total_found': len(results)
    }

@router.get("/blocks")
async def get_blocks(
    limit: int = Query(100, ge=1, le=1000),
    db: any = Depends(get_db)
):
    """Get blockchain blocks"""
    all_blocks = BlockchainDB.get_all_blocks(db)
    limited_blocks = all_blocks[-limit:] if len(all_blocks) > limit else all_blocks
    
    blocks_info = []
    for block in limited_blocks:
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
        'blocks': blocks_info,
        'total_blocks': len(all_blocks),
        'returned_blocks': len(blocks_info)
    }

@router.get("/recent")
async def get_recent_transactions(
    limit: int = Query(50, ge=1, le=1000),
    db: any = Depends(get_db)
):
    """Get recent transactions"""
    recent_txs = TransactionDB.get_recent_transactions(db, limit)
    
    results = []
    for tx in recent_txs:
        results.append({
            'transaction_id': tx.tx_id,
            'hospital_id': tx.hospital_id,
            'update_hash': tx.update_hash,
            'verdict': tx.verdict,
            'evidence_hash': tx.evidence_hash,
            'anomaly_score': tx.anomaly_score,
            'timestamp': tx.timestamp.isoformat()
        })
    
    return {
        'transactions': results,
        'total_count': len(results)
    }

@router.get("/health")
async def ledger_health():
    """Health check for the ledger system"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'component': 'ledger_stub',
        'database_connected': True,
        'pending_transactions': len(blockchain_simulator.pending_transactions)
    }