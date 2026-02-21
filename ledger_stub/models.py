from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import os

Base = declarative_base()

class Transaction(Base):
    """Transaction table for recording model update decisions"""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True, index=True)
    tx_id = Column(String, unique=True, index=True)
    hospital_id = Column(Integer, nullable=False)
    update_hash = Column(String, nullable=False)
    verdict = Column(String, nullable=False)
    evidence_hash = Column(String)
    anomaly_score = Column(Float)
    normalized_score = Column(Float)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    meta_data = Column(Text)

class BlockchainSimulation(Base):
    """Blockchain simulation table for ledger operations"""
    __tablename__ = 'blockchain_simulation'
    
    id = Column(Integer, primary_key=True, index=True)
    block_hash = Column(String, unique=True, index=True)
    previous_hash = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    transaction_count = Column(Integer)
    transactions = Column(Text)  # JSON string of transaction IDs
    nonce = Column(Integer)
    difficulty = Column(Integer, default=4)

class AuditTrail(Base):
    """Audit trail for tracking all operations"""
    __tablename__ = 'audit_trail'
    
    id = Column(Integer, primary_key=True, index=True)
    operation = Column(String, nullable=False)  # CREATE, UPDATE, DELETE, READ
    entity_type = Column(String, nullable=False)  # transaction, block, etc.
    entity_id = Column(Integer)  # ID of affected entity
    old_values = Column(Text)  # Previous values as JSON
    new_values = Column(Text)  # New values as JSON
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = Column(String, default="system")  # User who performed operation

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./transactions.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for getting DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")

# Initialize database
create_tables()