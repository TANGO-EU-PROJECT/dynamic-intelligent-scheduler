#!/usr/bin/env python3
"""
Quick test script for TornadoVM ML Inference Engine
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

try:
    from engine.inference_engine import TornadoVMInferenceEngine
    
    # Test with sample data
    json_input = {
        "testWorkload": {
            "BACKEND": "PTX",
            "DEVICE_ID": "0:2",
            "DEVICE": "GeForce GTX 1650",
            "Global Memory Loads": "15",
            "Global Memory Stores": "6",
            "Local Memory Loads": "0",
            "Local Memory Stores": "0",
            "Total Loops": "2",
            "Parallel Loops": "1",
            "Cast Operations": "2",
            "Vector Operations": "0",
            "Integer & Float Operations": "57"
        }
    }
    
    print("Testing TornadoVM ML Inference Engine...")
    engine = TornadoVMInferenceEngine()
    result = engine.predict_from_json(json_input)
    
    print("✓ Test successful!")
    print(f"Predicted Hardware: {result['predicted_device'].upper()}")
    print(f"Confidence: {max(result['confidence_scores'].values()):.3f}")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)
