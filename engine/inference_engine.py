import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class TornadoVMInferenceEngine:
    """
    Inference engine for TornadoVM ML task scheduler.
    Predicts optimal hardware (CPU, iGPU, GPU) for computational tasks.
    """
    def __init__(self, model_dir: str = "./ML", mode: str = "performance"):
        """
        Initialize the inference engine.
        
        Args:
            model_dir: Directory containing the saved model files
        """
        self.model_dir = model_dir
        self.mode = mode.lower()

        self.model_dir = Path(model_dir).resolve()
        default_path = Path("./ML").resolve()

        #if self.model_dir == default_path:
        subfolder = "Energy-Trained-Models" if self.mode == "energy" else "Performance-Trained-Models"
        classifier_dir = self.model_dir / subfolder
        #else:
        #    classifier_dir = self.model_dir

        if self.mode == "energy":
            # Load the six trained classifiers
            self.classifier_1 = joblib.load(f"{classifier_dir}/clf1_energy_etc.pkl")
            self.classifier_2 = joblib.load(f"{classifier_dir}/clf2_energy_etc.pkl")
            self.classifier_3 = joblib.load(f"{classifier_dir}/clf3_energy_etc.pkl")
            self.classifier_4 = joblib.load(f"{classifier_dir}/clf4_energy_etc.pkl")
            self.classifier_5 = joblib.load(f"{classifier_dir}/clf5_energy_etc.pkl")
            self.classifier_6 = joblib.load(f"{classifier_dir}/clf6_energy_etc.pkl")

            # Power mode thresholds (from test results)
            self.thresholds = {
                "igpu_cpu": 0.0,      # clf1: regression threshold
                "gpu_cpu": 0.4,       # clf2: classification threshold
                "gpu_igpu": 0.67,     # clf3: classification threshold
                "java_cpu": 0.5,      # clf4: classification threshold
                "java_gpu": 0.5,      # clf5: classification threshold
                "java_igpu": 0.5      # clf6: classification threshold
            }
        else:
            # Load the three trained classifiers
            print("Using the performance classifiers")
            self.classifier_1 = joblib.load(f"{classifier_dir}/IGPUvsCPU_final.joblib")
            self.classifier_2 = joblib.load(f"{classifier_dir}/GPUvsCPU_final.joblib")
            self.classifier_3 = joblib.load(f"{classifier_dir}/GPUvsIGPU_final.joblib")

            # Performance mode thresholds
            self.thresholds = {
                "igpu_cpu": 0.15,    # Classifier 1 threshold
                "gpu_cpu": 0.4,      # Classifier 2 threshold
                "gpu_igpu": 0.67     # Classifier 3 threshold
            }

        # Load feature names
        try:
            with open(f"{model_dir}/Final Artifacts/features.txt", 'r') as f:
                self.feature_names = json.load(f)
        except FileNotFoundError:
            # Fallback feature names if file doesn't exist
            self.feature_names = {
                "c1": ["threads", "global_memory_loads", "global_memory_stores", "local_memory_loads", "local_memory_stores", "total_loops", "parallel_loops", "cast_operations", "vector_operations", "total_integer_operations"],
                "c2": ["threads", "global_memory_loads", "global_memory_stores", "local_memory_loads", "local_memory_stores", "total_loops", "parallel_loops", "cast_operations", "vector_operations", "total_integer_operations"],
                "c3": ["threads", "global_memory_loads", "global_memory_stores", "local_memory_loads", "local_memory_stores", "total_loops", "parallel_loops", "cast_operations", "vector_operations", "total_integer_operations"]
            }
        
        # Required features (same for all classifiers)
        self.required_features = [
            "threads",
            "global_memory_loads", 
            "global_memory_stores",
            "local_memory_loads",
            "local_memory_stores", 
            "total_loops",
            "parallel_loops",
            "cast_operations",
            "vector_operations",
            "total_integer_operations"
        ]
        
        # Mapping from JSON field names to required features
        self.field_mapping = {
            "threads": "threads",
            "Global Memory Loads": "global_memory_loads",
            "Global Memory Stores": "global_memory_stores", 
            "Local Memory Loads": "local_memory_loads",
            "Local Memory Stores": "local_memory_stores",
            "Total Loops": "total_loops",
            "Parallel Loops": "parallel_loops", 
            "Cast Operations": "cast_operations",
            "Vector Operations": "vector_operations",
            "Integer & Float Operations": "total_integer_operations"
        }

    def parse_json_input(self, json_data: Dict) -> Dict[str, float]:
        """
        Parse JSON input format and convert to required feature format.
        
        Args:
            json_data: Dictionary containing workload data in JSON format
            
        Returns:
            Dictionary with required features for prediction
        """
        # Extract the first workload (assuming single workload per JSON)
        workload_name = list(json_data.keys())[0]
        workload_data = json_data[workload_name]
        
        # Map JSON fields to required features
        features = {}
        
        # Map the available fields (including threads)
        for json_field, required_field in self.field_mapping.items():
            if json_field in workload_data:
                # Convert string to float/int
                value = workload_data[json_field]
                if isinstance(value, str):
                    features[required_field] = float(value)
                else:
                    features[required_field] = float(value)
            else:
                # Use default value if field is missing
                if required_field == "threads":
                    features[required_field] = 64.0  # Default threads
                else:
                    features[required_field] = 0.0
        
        return features
    
    def predict_from_json(self, json_data: Dict) -> Dict[str, any]:
        """
        Predict hardware directly from JSON input format.
        
        Args:
            json_data: Dictionary containing workload data in JSON format
            
        Returns:
            Dictionary containing prediction results
        """
        # Parse JSON to required format
        features = self.parse_json_input(json_data)
        
        # Get prediction
        return  self.predict_hardware(features)
    
    def validate_input(self, features: Dict[str, float]) -> bool:
        """
        Validate that all required features are provided.
        
        Args:
            features: Dictionary of feature names to values
            
        Returns:
            True if valid, raises ValueError if not
        """
        missing_features = set(self.required_features) - set(features.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Check for non-numeric values
        for feature in self.required_features:
            if feature not in features:
                print(f"❌ Missing required feature: {feature}")
                return False
        return True

    def predict_hardware(self, features: Dict[str, float], available_devices: List[str] = None) -> Dict[str, any]:
        """
        Predict optimal hardware for given features.
        Optional filtering by available devices.
        """
        # Validate input
        if not self.validate_input(features):
            raise ValueError("Invalid input features")

        # Convert features to array for prediction
        feature_array = np.array([features[feature] for feature in self.required_features]).reshape(1, -1)

        if self.mode == "energy":
            result = self._predict_power_mode(feature_array, features)
        else:
            result = self._predict_performance_mode(feature_array, features)

        return result

    def _predict_power_mode(self, feature_array: np.ndarray, features: Dict[str, float]) -> Dict[str, any]:
        """
        Power mode prediction using 6 classifiers.
        """
        # Get predictions from all 6 classifiers
        prob_1 = self.classifier_1.predict(feature_array)[0]  # clf1: regression
        prob_2 = self.classifier_2.predict_proba(feature_array)[0][1]  # clf2: classification
        prob_3 = self.classifier_3.predict_proba(feature_array)[0][1]  # clf3: classification
        prob_4 = self.classifier_4.predict_proba(feature_array)[0][1]  # clf4: classification
        prob_5 = self.classifier_5.predict_proba(feature_array)[0][1]  # clf5: classification
        prob_6 = self.classifier_6.predict_proba(feature_array)[0][1]  # clf6: classification

        # Apply thresholds
        igpu_fit = prob_1 < self.thresholds["igpu_cpu"]  # clf1: regression threshold
        gpu_fit = prob_2 > self.thresholds["gpu_cpu"]    # clf2: classification threshold
        gpu_igpu_fit = prob_3 > self.thresholds["gpu_igpu"]  # clf3: classification threshold
        java_cpu_fit = prob_4 > self.thresholds["java_cpu"]  # clf4: classification threshold
        java_gpu_fit = prob_5 > self.thresholds["java_gpu"]  # clf5: classification threshold
        java_igpu_fit = prob_6 > self.thresholds["java_igpu"]  # clf6: classification threshold

        # Determine base device (first 3 classifiers)
        device_code = ""
        device_code += "1" if igpu_fit else "0"
        device_code += "1" if gpu_fit else "0"
        device_code += "1" if gpu_igpu_fit else "0"

        # Map device code to base device
        base_device = self._map_device_code_to_device(device_code)

        # Determine if Java is better than base device
        java_better = False
        #if base_device == "cpu" and java_cpu_fit:
        #    java_better = True
        #elif base_device == "gpu" and java_gpu_fit:
        #    java_better = True
        #elif base_device == "igpu" and java_igpu_fit:
        #    java_better = True

        # Final device recommendation
        predicted_device = "java" if java_better else base_device

        return {
            "predicted_device": predicted_device,
            "base_device": base_device,
            "java_recommended": java_better,
            "confidence_scores": {
                "igpu_vs_cpu": prob_1,
                "gpu_vs_cpu": prob_2,
                "gpu_vs_igpu": prob_3,
                "java_vs_cpu": prob_4,
                "java_vs_gpu": prob_5,
                "java_vs_igpu": prob_6
            },
            "classifier_decisions": {
                "igpu_fit": igpu_fit,
                "gpu_fit": gpu_fit,
                "gpu_igpu_fit": gpu_igpu_fit,
                "java_cpu_fit": java_cpu_fit,
                "java_gpu_fit": java_gpu_fit,
                "java_igpu_fit": java_igpu_fit
            },
            "raw_probabilities": [prob_1, prob_2, prob_3, prob_4, prob_5, prob_6],
            "device_code": device_code,
            "mode": "energy"
        }

    def _predict_performance_mode(self, feature_array: np.ndarray, features: Dict[str, float]) -> Dict[str, any]:
        """
        Performance mode prediction using 3 classifiers.
        """
        # Get predictions from the 3 classifiers
        prob_1 = self.classifier_1.predict_proba(feature_array)[0][1]  # iGPU vs CPU
        prob_2 = self.classifier_2.predict_proba(feature_array)[0][1]  # GPU vs CPU
        prob_3 = self.classifier_3.predict_proba(feature_array)[0][1]  # GPU vs iGPU

        # Apply thresholds
        igpu_fit = prob_1 > self.thresholds["igpu_cpu"]
        gpu_fit = prob_2 > self.thresholds["gpu_cpu"]
        gpu_igpu_fit = prob_3 > self.thresholds["gpu_igpu"]

        # Determine device code
        device_code = ""
        device_code += "1" if igpu_fit else "0"
        device_code += "1" if gpu_fit else "0"
        device_code += "1" if gpu_igpu_fit else "0"

        # Map device code to device
        predicted_device = self._map_device_code_to_device(device_code)

        return {
            "predicted_device": predicted_device,
            "confidence_scores": {
                "igpu_vs_cpu": prob_1,
                "gpu_vs_cpu": prob_2,
                "gpu_vs_igpu": prob_3
            },
            "classifier_decisions": {
                "igpu_fit": igpu_fit,
                "gpu_fit": gpu_fit,
                "gpu_igpu_fit": gpu_igpu_fit
            },
            "raw_probabilities": [prob_1, prob_2, prob_3],
            "device_code": device_code,
            "mode": "performance"
        }


    def _map_device_code_to_device(self, device_code: str) -> str:
        """
        Map 3-digit device code to device name.
        """
        if device_code in ["000", "001"]:
            return "cpu"
        elif device_code in ["100", "101", "110"]:
            return "igpu"
        elif device_code in ["010", "011", "111"]:
            return "gpu"
        else:
            return "cpu"  # Default fallback
    
    def batch_predict(self, features_list: List[Dict[str, float]]) -> List[Dict[str, any]]:
        """
        Predict hardware for multiple tasks.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for features in features_list:
            try:
                result = self.predict_hardware(features)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        return results
    
    def batch_predict_from_json(self, json_list: List[Dict]) -> List[Dict[str, any]]:
        """
        Predict hardware for multiple JSON inputs.
        
        Args:
            json_list: List of JSON dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for json_data in json_list:
            try:
                result = self.predict_from_json(json_data)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        return results
    
    def get_feature_importance(self, classifier_name: str = "all") -> Dict[str, List[float]]:
        """
        Get feature importance scores for the classifiers.
        
        Args:
            classifier_name: 'all', 'igpu_cpu', 'gpu_cpu', 'gpu_igpu', 'java_cpu', 'java_gpu', 'java_igpu'
            
        Returns:
            Dictionary of feature importance scores
        """
        importance_dict = {}

        if self.mode == "energy":
            if classifier_name in ["all", "igpu_cpu"]:
                importance_dict["igpu_cpu"] = self.classifier_1.feature_importances_.tolist()
            if classifier_name in ["all", "gpu_cpu"]:
                importance_dict["gpu_cpu"] = self.classifier_2.feature_importances_.tolist()
            if classifier_name in ["all", "gpu_igpu"]:
                importance_dict["gpu_igpu"] = self.classifier_3.feature_importances_.tolist()
            if classifier_name in ["all", "java_cpy"]:
                importance_dict["gpu_igpu"] = self.classifier_4.feature_importances_.tolist()
            if classifier_name in ["all", "java_gpu"]:
                importance_dict["gpu_igpu"] = self.classifier_5.feature_importances_.tolist()
            if classifier_name in ["all", "java_igpu"]:
                importance_dict["gpu_igpu"] = self.classifier_6.feature_importances_.tolist()
        else:
            if classifier_name in ["all", "igpu_cpu"]:
                importance_dict["igpu_cpu"] = self.classifier_1.feature_importances_.tolist()
            if classifier_name in ["all", "gpu_cpu"]:
                importance_dict["gpu_cpu"] = self.classifier_2.feature_importances_.tolist()
            if classifier_name in ["all", "gpu_igpu"]:
                importance_dict["gpu_igpu"] = self.classifier_3.feature_importances_.tolist()
        
        return importance_dict


# Example usage
if __name__ == "__main__":
    # Initialize the inference engine
    engine = TornadoVMInferenceEngine()
    
    # Example input features (you need to provide these)
    example_features = {
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
    result = engine.predict_hardware(example_features)
    
    print("Prediction Result:")
    print(f"Predicted Device: {result['predicted_device']}")
    print(f"Confidence Scores: {result['confidence_scores']}")
    print(f"Classifier Decisions: {result['classifier_decisions']}")
    print(f"Device Code: {result['device_code']}") 