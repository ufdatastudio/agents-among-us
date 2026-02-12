#!/usr/bin/env python3
"""
TEST SCRIPT - Verify Backend Integration
Run this to test each component before full deployment
"""

import os
import sys
import json
import requests
import time
import subprocess

# Configuration
BASE_URL = "http://localhost:3000"
BACKEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_flask_running():
    """Test 1: Check if Flask is running"""
    print_header("TEST 1: Flask Server Running")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print("✅ Flask server is running!")
            print(f"   Backend Path: {data.get('backend_path')}")
            print(f"   Data Dir: {data.get('data_dir')}")
            return True
        else:
            print(f"❌ Flask returned {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask!")
        print("   Run: python frontend/app.py")
        return False

def test_pages_load():
    """Test 2: Check if all pages load"""
    print_header("TEST 2: Page Routes")
    
    pages = {
        'Home': '/',
        'Config': '/config',
        'Game': '/game',
        'Stats': '/stats'
    }
    
    all_good = True
    for name, route in pages.items():
        try:
            response = requests.get(f"{BASE_URL}{route}", timeout=2)
            if response.status_code == 200:
                print(f"✅ {name} page loads")
            else:
                print(f"❌ {name} page failed: {response.status_code}")
                all_good = False
        except Exception as e:
            print(f"❌ {name} page error: {e}")
            all_good = False
    
    return all_good

def test_live_state_file():
    """Test 3: Check if backend creates live_state.json"""
    print_header("TEST 3: Backend Creates live_state.json")
    
    live_state_path = os.path.join(BACKEND_PATH, 'live_state.json')
    
    if os.path.exists(live_state_path):
        print(f"✅ File exists: {live_state_path}")
        
        try:
            with open(live_state_path, 'r') as f:
                data = json.load(f)
            
            print(f"✅ Valid JSON")
            print(f"   Keys: {list(data.keys())}")
            
            if 'game_info' in data:
                print(f"   Game ID: {data['game_info'].get('game_id')}")
                print(f"   Round: {data['game_info'].get('round')}")
            
            if 'agents' in data:
                print(f"   Agents: {len(data['agents'])}")
            
            return True
            
        except json.JSONDecodeError:
            print("❌ File exists but is not valid JSON")
            return False
    else:
        print(f"⚠️  File does not exist yet: {live_state_path}")
        print("   This is normal if no game has run yet")
        return True  # Not a failure

def test_stats_csv_exists():
    """Test 4: Check if stats.csv files exist in logs"""
    print_header("TEST 4: Backend Creates stats.csv")
    
    import glob
    pattern = os.path.join(BACKEND_PATH, 'logs', '*', 'Game_*_Run0', 'stats.csv')
    csv_files = glob.glob(pattern)
    
    if csv_files:
        print(f"✅ Found {len(csv_files)} stats.csv files")
        print(f"   Example: {csv_files[0]}")
        
        # Try to read one
        try:
            import pandas as pd
            df = pd.read_csv(csv_files[0])
            print(f"✅ CSV is readable")
            print(f"   Columns: {list(df.columns)[:5]}...")
            print(f"   Rows: {len(df)}")
            return True
        except Exception as e:
            print(f"❌ Cannot read CSV: {e}")
            return False
    else:
        print("⚠️  No stats.csv files found in logs/")
        print("   This is normal if no games have been run yet")
        return True

def test_stats_api():
    """Test 5: Check if stats API works"""
    print_header("TEST 5: Stats API Endpoints")
    
    try:
        # Test refresh
        print("Testing: POST /api/stats/refresh")
        response = requests.post(f"{BASE_URL}/api/stats/refresh", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Refresh works - Added {data.get('new_games', 0)} games")
        else:
            print(f"❌ Refresh failed: {response.status_code}")
            return False
        
        # Test get all
        print("Testing: GET /api/stats/all")
        response = requests.get(f"{BASE_URL}/api/stats/all", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Get all works - {len(data)} rows")
            if len(data) > 0:
                print(f"   First row game_id: {data[0].get('game_id')}")
        else:
            print(f"❌ Get all failed: {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Stats API error: {e}")
        return False

def test_game_state_api():
    """Test 6: Check if game state API works"""
    print_header("TEST 6: Game State API")
    
    try:
        response = requests.get(f"{BASE_URL}/api/game_state", timeout=2)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Game state API works")
            print(f"   Status: {data.get('status', 'unknown')}")
            
            if 'game_info' in data:
                print(f"   Game is running!")
                print(f"   Round: {data['game_info'].get('round')}")
            
            return True
        else:
            print(f"❌ Game state API failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Game state API error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "█"*60)
    print("  AGENTS AMONG US - BACKEND INTEGRATION TEST")
    print("█"*60)
    
    tests = [
        ("Flask Running", test_flask_running),
        ("Pages Load", test_pages_load),
        ("live_state.json", test_live_state_file),
        ("stats.csv Files", test_stats_csv_exists),
        ("Stats API", test_stats_api),
        ("Game State API", test_game_state_api)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {name}")
    
    print(f"\n{'='*60}")
    print(f"  TOTAL: {passed}/{total} tests passed")
    print(f"{'='*60}\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready to deploy!")
        return 0
    else:
        print("⚠️  Some tests failed. Check errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests())