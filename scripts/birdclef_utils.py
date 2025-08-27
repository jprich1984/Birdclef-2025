# birdclef_utils.py

import os
import zipfile
import joblib
from google.colab import drive

def retrieve_and_process_birdclef_data(zip_filename='birdclef-2025.zip'):
    try:
        drive.mount('/content/drive')
        zip_filepath = f'/content/drive/MyDrive/Main_Birdclef/{zip_filename}'
        if not os.path.exists(zip_filepath):
            print(f"Error: Zip file not found at {zip_filepath}.")
            return None
        extraction_path = '/content/data'
        os.makedirs(extraction_path, exist_ok=True)
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            zf.extractall(extraction_path)
            print(f"Successfully extracted all files from {zip_filename} to {extraction_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def load_label_encoder(label_encoder_path='label_encoder.joblib'):
    try:
        label_encoder = joblib.load(label_encoder_path)
        print(f"Successfully loaded label_encoder from: {label_encoder_path}")
        return label_encoder
    except FileNotFoundError:
        print(f"Error: File not found at: {label_encoder_path}.")
        return None
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        return None
