
import os
import pickle
import numpy as np
import pandas as pd
import traceback

class ModelHandler:
    def __init__(self, models_config, scalar_path=None, pca_path=None, columns_path=None):
        self.models_config = models_config
        self.models = {}
        self.scaler = None
        self.pca = None
        self.model_columns = None
        
        # Load Main Models
        self.load_models()
        
        # Load Preprocessing Objects
        self.load_artifact(scalar_path, 'scaler')
        self.load_artifact(pca_path, 'pca')
        self.load_artifact(columns_path, 'model_columns')

    def load_artifact(self, path, attr_name):
        if path and os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    setattr(self, attr_name, pickle.load(f))
                print(f"{attr_name} loaded from {path}")
            except Exception as e:
                print(f"Error loading {attr_name}: {e}")
        else:
            print(f"Warning: {attr_name} file not found at {path}")

    def load_models(self):
        for name, path in self.models_config.items():
            try:
                if not os.path.exists(path):
                    print(f"Model file not found for {name} at: {path}")
                    self.models[name] = None
                    continue

                with open(path, 'rb') as f:
                    loaded_obj = pickle.load(f)
                
                if isinstance(loaded_obj, dict):
                    self.models[name] = loaded_obj.get('model')
                    # Fallbacks if global artifacts missing
                    if not self.scaler and loaded_obj.get('scaler'):
                        self.scaler = loaded_obj.get('scaler')
                    if not self.pca and loaded_obj.get('pca'):
                        self.pca = loaded_obj.get('pca')
                else:
                    self.models[name] = loaded_obj
                
                print(f"Model '{name}' loaded successfully. Type: {type(self.models[name])}")
                
            except Exception as e:
                print(f"Error loading model '{name}': {e}")
                print(traceback.format_exc())
                self.models[name] = None

    def preprocess(self, input_features):
        """
        Process features: 9 Inputs -> Scaler -> PCA -> 8 Components
        """
        # 1. Enforce strict input array of 9 features
        # Input order checked by caller (Form) to be:
        # ['Age', 'Sex', 'Chest pain type', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
        
        # Convert to numpy array (reshape to 2D)
        features_array = np.array(input_features).reshape(1, -1)
        
        # Validate shape
        if features_array.shape[1] != 9:
             raise ValueError(f"Expected 9 features, got {features_array.shape[1]}")
            
        print(f"Debug: Input shape: {features_array.shape}")
        
        # 2. Scale
        if self.scaler:
            features_scaled = self.scaler.transform(features_array)
        else:
            print("Warning: Scaler not loaded!")
            features_scaled = features_array
            
        # 3. PCA
        if self.pca:
            features_pca = self.pca.transform(features_scaled)
            print(f"Debug: PCA Output Shape: {features_pca.shape}") 
            # Should be (1, 8)
        else:
            print("Warning: PCA not loaded!")
            features_pca = features_scaled
            
        return features_pca

    def predict(self, model_name, input_features):
        model = self.models.get(model_name)
        if not model:
            return "Model Not Loaded", 0.0

        try:
            # Preprocess (Scale + PCA)
            df_processed = self.preprocess(input_features)
            
            # Prediction
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(df_processed)[0][1]
            else:
                prob = 0.0
            
            pred = model.predict(df_processed)[0]
            
            # User requested 'Present' or 'Absent'
            result_str = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
            
            return result_str, prob
            
        except Exception as e:
            print(f"Prediction Error ({model_name}): {e}")
            print(traceback.format_exc())
            return f"Error: {str(e)}", 0.0
            


# Initialize handler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Models are in the parent directory of 'app'
PROJECT_ROOT = os.path.dirname(BASE_DIR)

LR_PATH = os.path.join(PROJECT_ROOT, "heart_modelrg.pkl")
RF_PATH = os.path.join(PROJECT_ROOT, "heart_model.pkl")

SCALER_PATH = os.path.join(PROJECT_ROOT, "scaler.pkl")
PCA_PATH = os.path.join(PROJECT_ROOT, "pca.pkl")
COLUMNS_PATH = os.path.join(PROJECT_ROOT, "model_columns.pkl")

MODELS_CONFIG = {
    "Logistic Regression": LR_PATH,
    "Random Forest": RF_PATH
}

model_handler = ModelHandler(MODELS_CONFIG, SCALER_PATH, PCA_PATH, COLUMNS_PATH)


