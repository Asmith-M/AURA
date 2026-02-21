from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.ledger_routes import router as ledger_router

# Create FastAPI app
app = FastAPI(
    title="AURA Ledger Stub API",
    description="Blockchain simulation for federated learning security ledger",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include ledger routes
app.include_router(ledger_router)

@app.get("/")
async def root():
    """Root endpoint for ledger stub"""
    return {
        "message": "AURA Ledger Stub API",
        "description": "Blockchain simulation for federated learning security",
        "version": "0.1.0",
        "endpoints": [
            "/ledger/log_verdict",
            "/ledger/history/{hospital_id}",
            "/ledger/transaction/{tx_id}",
            "/ledger/stats",
            "/ledger/chain",
            "/ledger/search",
            "/ledger/blocks",
            "/ledger/recent"
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ledger_stub",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)