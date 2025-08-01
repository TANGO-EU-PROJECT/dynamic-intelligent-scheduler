# TornadoVM ML Inference Engine

A machine learning-based hardware prediction system for TornadoVM computational tasks.

## Quick Start

1. **Install dependencies:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

2. **Test the installation:**
   ```bash
   PYTHONPATH=$(pwd) python tests/test_inference.py
   ```

3. **Run inference:**
   ```bash
   PYTHONPATH=$(pwd) python engine/engine.py tests/features.json 16
   ```

## Usage

### Command Line Interface

```bash
python3 engine.py <json_file> <input_size>
```

**Examples:**
```bash
# Basic usage
PYTHONPATH=$(pwd) python3 engine/engine.py tests/features.json 64

# Verbose output
PYTHONPATH=$(pwd) python3 engine/engine.py tests/features.json 1024

# Different input sizes
PYTHONPATH=$(pwd) python3 engine/engine.py tests/features.json 256
PYTHONPATH=$(pwd) python3 engine/engine.py tests/features.json 2048
```

### JSON Input Format

Your JSON file should contain workload features:

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

**Note:** The `threads` field will be automatically set to your input size.

### Output

The engine predicts the optimal hardware (CPU, iGPU, or GPU) and provides:

- **Predicted Hardware:** CPU, iGPU, or GPU
- **Confidence Scores:** From each classifier
- **Device Code:** 3-digit binary code representing decisions
- **Reasoning:** Explanation based on input size

### Input Size Guidelines

| Input Size | Typical Prediction | Reasoning |
|------------|-------------------|-----------|
| < 128      | CPU               | Small workloads favor CPU |
| 128-1024   | iGPU              | Medium workloads favor iGPU |
| > 1024     | GPU               | Large workloads favor GPU |

## Files Included

- `inference_engine.py` - Core inference engine
- `engine.py` - Command-line interface
- `utils/utilities.py` - Utility functions
- `utils/plotting_utilities.py` - Visualization utilities
- `requirements.txt` - Python dependencies
- `*.joblib` - Trained model files
- `ML/Final Artifacts/features.txt` - Feature definitions
- `tests/features.json` - Example input file
- `tests/test_inference.py` - Quick test script
- `install.sh` - Installation script

## Requirements

- Python 3.7+
- pip3
- Internet connection (for dependency installation)

## Troubleshooting

1. **"Module not found" errors:**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **"Model files not found" errors:**
   Make sure all `.joblib` files are in the current directory.

3. **"JSON file not found" errors:**
   Check that your JSON file exists and is readable.

## Integration

To integrate with TornadoVM:

```bash
# In your TornadoVM script
PYTHONPATH=$(pwd) python3 engine/engine.py workload_features.json $INPUT_SIZE
```

The engine will return the predicted hardware that you can use for scheduling.

## Support

For issues or questions, check the model training notebook or contact the development team.
