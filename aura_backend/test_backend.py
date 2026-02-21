"""
Test script for AURA Backend
Tests both demo and real modes
"""
import pytest
import os
import sys
import time
import json
import requests
from pathlib import Path

pytestmark = pytest.mark.skip(reason="Manual integration script; not a pytest test module.")

BASE_URL = "http://localhost:8000"

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def test_system_endpoints():
    """Test system endpoints"""
    print_header("Testing System Endpoints")
    
    # Test mode endpoint
    print("\n1. Testing /system/mode...")
    try:
        response = requests.get(f"{BASE_URL}/system/mode")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Mode: {data['mode']}")
        print(f"   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Test health endpoint
    print("\n2. Testing /system/health...")
    try:
        response = requests.get(f"{BASE_URL}/system/health")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Mode: {data['mode']}")
        print(f"   Detector: {data['detector_loaded']}")
        print(f"   Ledger: {data['ledger_connected']}")
        print(f"   Golden Set: {data['golden_set_loaded']}")
        print(f"   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")

def test_demo_submission():
    """Test demo mode submission"""
    print_header("Testing Demo Mode Submission")
    
    print("\nSubmitting 3 demo requests...")
    
    for i in range(3):
        print(f"\n{i+1}. Submitting from HOSP{i+1}...")
        try:
            response = requests.post(
                f"{BASE_URL}/sentinel/submit_update",
                data={"hospital_id": f"HOSP{i+1}"}
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Session ID: {data['session_id']}")
                print(f"   Verdict: {data['verdict']}")
                print(f"   Anomaly Score: {data['anomaly_analysis']['anomaly_score']}")
                print(f"   Processing Time: {data.get('processing_time', 'N/A')}s")
                print(f"   ✓ Success")
                
                # Verify response schema
                required_keys = [
                    'session_id', 'hospital_id', 'timestamp', 'model_profile',
                    'golden_test', 'fingerprint', 'shap_analysis',
                    'anomaly_analysis', 'verdict', 'ledger'
                ]
                
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    print(f"   ⚠ Missing keys: {missing_keys}")
                else:
                    print(f"   ✓ All required keys present")
                
                return data['session_id']  # Return for session test
            else:
                print(f"   ✗ Failed: {response.text}")
                
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
    
    return None

def test_session_retrieval(session_id):
    """Test session retrieval"""
    if not session_id:
        print("\n⚠ Skipping session test (no session ID)")
        return
    
    print_header("Testing Session Retrieval")
    
    print(f"\n1. Retrieving session {session_id}...")
    try:
        response = requests.get(f"{BASE_URL}/session/{session_id}")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Verdict: {data['verdict']}")
            print(f"   Hospital: {data['hospital_id']}")
            print(f"   ✓ Success")
        else:
            print(f"   ✗ Failed: {response.text}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print("\n2. Listing recent sessions...")
    try:
        response = requests.get(f"{BASE_URL}/sessions/list?limit=5")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Count: {data['count']}")
            print(f"   ✓ Success")
        else:
            print(f"   ✗ Failed: {response.text}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")

def test_ledger_endpoints():
    """Test ledger endpoints"""
    print_header("Testing Ledger Endpoints")
    
    print("\n1. Getting recent transactions...")
    try:
        response = requests.get(f"{BASE_URL}/ledger/transactions?limit=5")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Count: {data['count']}")
            print(f"   ✓ Success")
        else:
            print(f"   ✗ Failed: {response.text}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")

def test_statistics():
    """Test statistics endpoint"""
    print_header("Testing Statistics")
    
    print("\n1. Getting summary statistics...")
    try:
        response = requests.get(f"{BASE_URL}/stats/summary")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Mode: {data['mode']}")
            print(f"   Session Stats: {data['session_stats']}")
            print(f"   ✓ Success")
        else:
            print(f"   ✗ Failed: {response.text}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")

def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  AURA Backend Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    print(f"\nBase URL: {BASE_URL}")
    print("\nMake sure the backend is running before proceeding!")
    print("  Demo mode: python main.py")
    print("  Or: uvicorn main:app --reload")
    
    input("\nPress Enter to start tests...")
    
    # Run tests
    test_system_endpoints()
    session_id = test_demo_submission()
    test_session_retrieval(session_id)
    test_ledger_endpoints()
    test_statistics()
    
    print_header("Test Suite Complete")
    print("\n✓ All tests executed\n")

if __name__ == "__main__":
    run_all_tests()
