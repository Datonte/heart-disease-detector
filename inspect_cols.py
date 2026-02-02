
import pickle
import os

path = r"C:\Users\user\Downloads\Heartfelt\model_columns.pkl"

if os.path.exists(path):
    try:
        with open(path, 'rb') as f:
            cols = pickle.load(f)
        print(f"Total Columns: {len(cols)}")
        for i, c in enumerate(cols):
            print(f"{i}: {c}")
    except Exception as e:
        print("Error reading pickle:", e)
else:
    print("model_columns.pkl not found.")
