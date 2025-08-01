#!/usr/bin/env python3
"""
Simple example of using TornadoVM ML Inference Engine with JSON input
"""

from engine.inference_engine import TornadoVMInferenceEngine

def main():
    # Your JSON input (from features.json)
    json_input = {
        "nBody": {
            "BACKEND": "PTX",
            "DEVICE_ID": "0:2",
            "DEVICE": "GeForce GTX 1650",
            "Global Memory Loads": "15",
            "Global Memory Stores": "6",
            "Constant Memory Loads": "0",
            "Constant Memory Stores": "0",
            "Local Memory Loads": "0",
            "Local Memory Stores": "0",
            "Private Memory Loads": "20",
            "Private Memory Stores": "20",
            "Total Loops": "2",
            "Parallel Loops": "1",
            "If Statements": "2",
            "Integer Comparison": "2",
            "Float Comparison": "0",
            "Switch Statements": "0",
            "Switch Cases": "0",
            "Vector Operations": "0",
            "Integer & Float Operations": "57",
            "Boolean Operations": "9",
            "Cast Operations": "2",
            "Float Math Functions": "1",
            "Integer Math Functions": "0"
        }
    }
    
    # Initialize engine
    engine = TornadoVMInferenceEngine()
    
    # Get prediction
    result = engine.predict_from_json(json_input)
    
    # Display results
    print("TornadoVM ML Prediction Results")
    print("=" * 40)
    print(f"Workload: nBody")
    print(f"Predicted Hardware: {result['predicted_device'].upper()}")
    print(f"Device Code: {result['device_code']}")
    
    print("\nConfidence Scores:")
    for classifier, score in result['confidence_scores'].items():
        print(f"  {classifier}: {score:.3f}")

if __name__ == "__main__":
    main() 