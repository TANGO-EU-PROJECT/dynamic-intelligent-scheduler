#!/usr/bin/env python3
"""
TornadoVM ML Inference Engine - Command Line Interface
Usage: python engine.py <json_file> <input_size>
Example: python engine.py features.json 1024
"""

import sys
import json
import argparse
from inference_engine import TornadoVMInferenceEngine

def load_json_file(filename: str) -> dict:
    """Load JSON file and return dictionary."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{filename}': {e}")
        sys.exit(1)

def main():
    """Main function for command-line inference engine."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="TornadoVM ML Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python engine.py features.json 64
  python engine.py workload.json 1024
  python engine.py -v features.json 256
        """
    )
    
    parser.add_argument(
        "json_file", 
        help="Path to JSON file containing workload features"
    )
    
    parser.add_argument(
        "input_size", 
        type=int,
        help="Input size (number of threads/elements)"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Verbose output with detailed information"
    )
    
    parser.add_argument(
        "--model-dir", 
        default="./ML",
        help="Directory containing model files (default: ./)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input size
    if args.input_size <= 0:
        print("Error: Input size must be positive.")
        sys.exit(1)
    
    print("TornadoVM ML Inference Engine")
    print("=" * 40)
    
    # Initialize the inference engine
    try:
        engine = TornadoVMInferenceEngine(model_dir=args.model_dir)
        print("✓ Engine initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize engine: {e}")
        print("\nMake sure you have the model files:")
        print(f"  - {args.model_dir}/IGPUvsCPU_final.joblib")
        print(f"  - {args.model_dir}/GPUvsCPU_final.joblib") 
        print(f"  - {args.model_dir}/GPUvsIGPU_final.joblib")
        print(f"  - {args.model_dir}/Final Artifacts/features.txt")
        sys.exit(1)
    
    # Load the JSON input
    try:
        json_data = load_json_file(args.json_file)
        print(f"✓ JSON loaded successfully from '{args.json_file}'")
    except Exception as e:
        print(f"✗ Failed to load JSON: {e}")
        sys.exit(1)
    
    # Update the JSON data with the provided input size
    workload_name = list(json_data.keys())[0]
    json_data[workload_name]["threads"] = str(args.input_size)
    
    if args.verbose:
        print(f"  Workload: {workload_name}")
        print(f"  Input Size: {args.input_size}")
    
    # Run prediction
    try:
        result = engine.predict_from_json(json_data)
        print("✓ Prediction completed successfully")
        
        # Display results
        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)
        
        # Display workload info
        workload_data = json_data[workload_name]
        print(f"Workload: {workload_name}")
        print(f"Input Size: {args.input_size}")
        print(f"Backend: {workload_data.get('BACKEND', 'N/A')}")
        print(f"Device: {workload_data.get('DEVICE', 'N/A')}")
        print(f"Device ID: {workload_data.get('DEVICE_ID', 'N/A')}")
        
        print(f"\nPredicted Hardware: {result['predicted_device'].upper()}")
        print(f"Device Code: {result['device_code']}")
        
        if args.verbose:
            print(f"\nConfidence Scores:")
            for classifier, score in result['confidence_scores'].items():
                print(f"  {classifier.replace('_', ' ').title()}: {score:.3f}")
            
            print(f"\nClassifier Decisions:")
            for decision, value in result['classifier_decisions'].items():
                status = "✓" if value else "✗"
                print(f"  {decision.replace('_', ' ').title()}: {status}")
            
            print(f"\nParsed Features:")
            for feature, value in result['parsed_features'].items():
                print(f"  {feature}: {value}")
        else:
            # Simple output
            print(f"Confidence: {max(result['confidence_scores'].values()):.3f}")
            
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Workload: {workload_name}")
    print(f"Input Size: {args.input_size}")
    print(f"Recommended Hardware: {result['predicted_device'].upper()}")
    
    # Provide reasoning based on input size
    if args.input_size < 128:
        print("Reasoning: Small input size suggests CPU optimization")
    elif args.input_size < 1024:
        print("Reasoning: Medium input size suggests iGPU optimization")
    else:
        print("Reasoning: Large input size suggests GPU optimization")

if __name__ == "__main__":
    main() 