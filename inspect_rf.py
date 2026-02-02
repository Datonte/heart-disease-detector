
import pickle
import os
import sys

# Force encoding to handle potential issues
sys.stdout.reconfigure(encoding='utf-8')

file_path = r"C:\Users\user\Downloads\Heartfelt\heart_model.pkl"

if not os.path.exists(file_path):
    print("File not found.")
else:
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Loaded: {file_path}")
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print("Is a Dictionary with keys:", list(data.keys()))
            if 'model' in data:
                m = data['model']
                if hasattr(m, 'n_features_in_'):
                    print(f"Model n_features_in_: {m.n_features_in_}")
                if hasattr(m, 'feature_names_in_'):
                    print(f"Model feature_names_in_: {m.feature_names_in_}")
        else:
            print("Is likely a raw model object.")
            if hasattr(data, 'feature_names_in_'):
                print(f"Num Features: {len(data.feature_names_in_)}")
                print(f"Features: {list(data.feature_names_in_)}")
            else:
                print("Model does not have feature_names_in_")
            
            if hasattr(data, 'n_features_in_'):
                print(f"n_features_in_: {data.n_features_in_}")
                
    except Exception as e:
        print(f"Error: {e}")
