import logging
import os
from datetime import datetime
from typing import Dict, Any
import json

class FLLogger:
    """Custom logger for Federated Learning operations"""
    
    def __init__(self, log_dir="./fl_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file logger
        self.logger = logging.getLogger("FL_Server")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(log_dir, f"fl_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_round_start(self, round_num: int, num_clients: int):
        """Log the start of a training round"""
        message = f"Starting Round {round_num} with {num_clients} clients"
        self.logger.info(message)
    
    def log_aggregation(self, round_num: int, metrics: Dict[str, Any]):
        """Log aggregation results"""
        message = f"Round {round_num} aggregation completed - Metrics: {metrics}"
        self.logger.info(message)
    
    def log_client_update(self, client_id: int, round_num: int, metrics: Dict[str, Any]):
        """Log client update"""
        message = f"Client {client_id} round {round_num} - Metrics: {metrics}"
        self.logger.info(message)
    
    def log_security_check(self, round_num: int, suspicious_clients: list):
        """Log security check results"""
        if suspicious_clients:
            message = f"Round {round_num} - Suspicious clients detected: {suspicious_clients}"
            self.logger.warning(message)
        else:
            message = f"Round {round_num} - No suspicious activity detected"
            self.logger.info(message)

# Global logger instance
fl_logger = FLLogger()