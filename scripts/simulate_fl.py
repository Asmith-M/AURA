import subprocess
import time
import threading
import sys
import os

def start_server():
    """Start the FL server in a separate thread"""
    print("Starting FL Server...")
    # UPDATED: Use -m flag and dot notation
    try:
        subprocess.run([sys.executable, "-m", "fl_server.server", "--rounds", "3"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Server error: {e}")

def start_client(client_id):
    """Start a FL client"""
    print(f"Starting Client {client_id}...")
    # UPDATED: Use -m flag and dot notation for the client modules
    try:
        subprocess.run([
            sys.executable, "-m", f"fl_client.client_{client_id}", 
            "--hospital_id", str(client_id)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Client {client_id} crashed: {e}")

def main():
    """Simulate FL training with multiple clients"""
    print("Starting FL simulation...")
    
    # Start server in background
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait a moment for server to start and bind to the port
    time.sleep(3)
    
    # Start multiple clients
    client_threads = []
    for i in range(1, 4):  # Start 3 clients
        client_thread = threading.Thread(target=start_client, args=(i,))
        client_thread.daemon = True
        client_thread.start()
        client_threads.append(client_thread)
        time.sleep(1.5)  # Stagger client starts to avoid connection spikes
    
    # Wait for server to complete
    server_thread.join()
    
    print("FL simulation completed!")

if __name__ == "__main__":
    main()