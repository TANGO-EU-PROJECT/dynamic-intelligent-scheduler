# TornadoVM ML Inference Engine

This inference engine provides offline hardware prediction for TornadoVM computational tasks using the trained machine learning models.

## Requirements

### Files Needed
1. **Model Files** (saved from training):
   - `IGPUvsCPU_final.joblib`
   - `GPUvsCPU_final.joblib` 
   - `GPUvsIGPU_final.joblib`
   - `./Final Artifacts/features.txt`

2. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Input Formats

### Format 1: Direct Feature Dictionary
Provide exactly these 10 features for each prediction:

| Feature | Description | Example Value |
|---------|-------------|---------------|
| `threads` | Number of threads | 64 |
| `global_memory_loads` | Global memory load operations | 1000 |
| `global_memory_stores` | Global memory store operations | 500 |
| `local_memory_loads` | Local memory load operations | 200 |
| `local_memory_stores` | Local memory store operations | 100 |
| `total_loops` | Total number of loops | 50 |
| `parallel_loops` | Number of parallel loops | 25 |
| `cast_operations` | Type casting operations | 10 |
| `vector_operations` | Vector operations | 5 |
| `total_integer_operations` | Total integer arithmetic operations | 2000 |

### Format 2: JSON Input (Recommended)
Provide JSON data in the format from TornadoVM profiling:

```json
{
    "workloadName": {
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
```

**Note**: The engine automatically maps JSON field names to required features and uses default values for missing fields.

## Usage

### Basic Usage with JSON Input
```python
from inference_engine import TornadoVMInferenceEngine

# Initialize engine
engine = TornadoVMInferenceEngine()

# Your JSON input
json_input = {
    "nBody": {
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

# Get prediction
result = engine.predict_from_json(json_input)
print(f"Predicted Device: {result['predicted_device']}")
```

### Basic Usage with Direct Features
```python
from inference_engine import TornadoVMInferenceEngine

# Initialize engine
engine = TornadoVMInferenceEngine()

# Provide features directly
features = {
    "threads": 64,
    "global_memory_loads": 1000,
    "global_memory_stores": 500,
    "local_memory_loads": 200,
    "local_memory_stores": 100,
    "total_loops": 50,
    "parallel_loops": 25,
    "cast_operations": 10,
    "vector_operations": 5,
    "total_integer_operations": 2000
}

# Get prediction
result = engine.predict_hardware(features)
print(f"Predicted Device: {result['predicted_device']}")
```

### Output Format
The engine returns a dictionary with:
- `predicted_device`: 'cpu', 'igpu', or 'gpu'
- `confidence_scores`: Probabilities from each classifier
- `classifier_decisions`: Binary decisions from each classifier
- `raw_probabilities`: Raw probability outputs
- `device_code`: 3-digit binary code representing decisions
- `input_json`: Original JSON input (when using JSON format)
- `parsed_features`: Parsed features used for prediction (when using JSON format)

### Example Output
```python
{
    "predicted_device": "gpu",
    "confidence_scores": {
        "igpu_vs_cpu": 0.85,
        "gpu_vs_cpu": 0.92,
        "gpu_vs_igpu": 0.78
    },
    "classifier_decisions": {
        "igpu_fit": True,
        "gpu_fit": True,
        "gpu_igpu_fit": True
    },
    "raw_probabilities": [0.85, 0.92, 0.78],
    "device_code": "111",
    "input_json": {...},  # Original JSON
    "parsed_features": {...}  # Parsed features
}
```

## Decision Logic

The engine uses a 3-stage voting system:

1. **Classifier 1** (iGPU vs CPU): Threshold = 0.15
2. **Classifier 2** (GPU vs CPU): Threshold = 0.4  
3. **Classifier 3** (GPU vs iGPU): Threshold = 0.67

Device codes are mapped as:
- `000`, `001` → CPU
- `100`, `101`, `110` → iGPU
- `010`, `011`, `111` → GPU

## Batch Processing

### JSON Batch Processing
```python
# Multiple JSON predictions
json_list = [json1, json2, json3]
results = engine.batch_predict_from_json(json_list)
```

### Feature Batch Processing
```python
# Multiple feature predictions
features_list = [features1, features2, features3]
results = engine.batch_predict(features_list)
```

## Error Handling

The engine validates:
- All 10 required features are provided (direct format)
- JSON format is valid and contains required fields
- All values are numeric
- Model files exist and are loadable

## Feature Importance

```python
# Get feature importance scores
importance = engine.get_feature_importance("all")
```

## Running Examples

```bash
# Test with your JSON input
PYTHONPATH=$(pwd) python tests/test_json_input.py

# Simple JSON example
PYTHONPATH=$(pwd) python tests/simple_json_example.py

# Run inference engine directly
PYTHONPATH=$(pwd) python engine/inference_engine.py
```

## JSON Field Mapping

The engine automatically maps these JSON fields to required features:

| JSON Field | Required Feature |
|------------|------------------|
| `Global Memory Loads` | `global_memory_loads` |
| `Global Memory Stores` | `global_memory_stores` |
| `Local Memory Loads` | `local_memory_loads` |
| `Local Memory Stores` | `local_memory_stores` |
| `Total Loops` | `total_loops` |
| `Parallel Loops` | `parallel_loops` |
| `Cast Operations` | `cast_operations` |
| `Vector Operations` | `vector_operations` |
| `Integer & Float Operations` | `total_integer_operations` |

**Note**: `threads` is set to a default value of 64 since it's not typically in the JSON output.

## Integration Notes

- **Threading**: The engine is thread-safe for concurrent predictions
- **Memory**: Models are loaded once during initialization
- **Performance**: Predictions are fast (< 1ms per prediction)
- **Scalability**: Supports batch processing for multiple tasks
- **JSON Support**: Automatically handles TornadoVM profiling JSON format 