#!/usr/bin/env python3
"""
Test script for TornadoVM ML Inference Engine with JSON input format
"""

import json
from engine.inference_engine import TornadoVMInferenceEngine

def load_json_file(filename: str) -> dict:
    """Load JSON file and return dictionary."""
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    # Initialize the inference engine
    print("Initializing TornadoVM ML Inference Engine...")
    try:
        engine = TornadoVMInferenceEngine()
        print("✓ Engine initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize engine: {e}")
        print("Make sure you have the model files:")
        print("  - IGPUvsCPU_final.joblib")
        print("  - GPUvsCPU_final.joblib") 
        print("  - GPUvsIGPU_final.joblib")
        print("  - ./Final Artifacts/features.txt")
        return
    
    # Load the JSON input
    print("\nLoading JSON input...")
    try:
        json_data = load_json_file("tests/features.json")
        print("✓ JSON loaded successfully")
        print(f"  Workload: {list(json_data.keys())[0]}")
    except Exception as e:
        print(f"✗ Failed to load JSON: {e}")
        return
    
    # Test prediction from JSON
    print("\nRunning prediction...")
    try:
        result = engine.predict_from_json(json_data)
        
        print("✓ Prediction completed successfully")
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        # Display workload info
        workload_name = list(json_data.keys())[0]
        workload_data = json_data[workload_name]
        print(f"Workload: {workload_name}")
        print(f"Backend: {workload_data.get('BACKEND', 'N/A')}")
        print(f"Device: {workload_data.get('DEVICE', 'N/A')}")
        print(f"Device ID: {workload_data.get('DEVICE_ID', 'N/A')}")
        
        print(f"\nPredicted Hardware: {result['predicted_device'].upper()}")
        print(f"Device Code: {result['device_code']}")
        
        print("\nConfidence Scores:")
        for classifier, score in result['confidence_scores'].items():
            print(f"  {classifier.replace('_', ' ').title()}: {score:.3f}")
        
        print("\nClassifier Decisions:")
        for decision, value in result['classifier_decisions'].items():
            status = "✓" if value else "✗"
            print(f"  {decision.replace('_', ' ').title()}: {status}")
        
        print("\nParsed Features:")
        for feature, value in result['parsed_features'].items():
            print(f"  {feature}: {value}")
            
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return
    
    # Test with multiple JSON inputs
    print("\n" + "="*60)
    print("BATCH PREDICTION TEST")
    print("="*60)
    
    # Create additional test cases
    test_cases = [
        json_data,  # Original nBody case
        {
            "matrixMultiplication": {
                "BACKEND": "PTX",
                "DEVICE_ID": "0:2", 
                "DEVICE": "GeForce GTX 1650",
                "Global Memory Loads": "1000",
                "Global Memory Stores": "500",
                "Local Memory Loads": "200",
                "Local Memory Stores": "100",
                "Total Loops": "100",
                "Parallel Loops": "80",
                "Cast Operations": "10",
                "Vector Operations": "50",
                "Integer & Float Operations": "5000"
            }
        },
        {
            "blackScholes": {
                "BACKEND": "PTX",
                "DEVICE_ID": "0:2",
                "DEVICE": "GeForce GTX 1650", 
                "Global Memory Loads": "100",
                "Global Memory Stores": "50",
                "Local Memory Loads": "50",
                "Local Memory Stores": "25",
                "Total Loops": "20",
                "Parallel Loops": "5",
                "Cast Operations": "5",
                "Vector Operations": "2",
                "Integer & Float Operations": "1000"
            }
        }
    ]
    
    try:
        batch_results = engine.batch_predict_from_json(test_cases)
        
        for i, result in enumerate(batch_results):
            if "error" in result:
                print(f"Case {i+1}: ERROR - {result['error']}")
            else:
                workload_name = list(result['input_json'].keys())[0]
                print(f"Case {i+1}: {workload_name} → {result['predicted_device'].upper()}")
                
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")

if __name__ == "__main__":
    main() 