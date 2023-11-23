import numpy as np
import joblib
import json
import os
import glob
import argparse
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# should match the training pre-processing
preprocess_image_size = (128,128)

def find_most_recent_image(directory):
    """Find the most recent file in the specified directory."""
    # List all files in the directory
    files = glob.glob(os.path.join(directory, '*.tif'))

    # Sort files by modification time in descending order
    files.sort(key=os.path.getmtime, reverse=True)

    # Return the most recent file, if any
    if files:
        return files[0]
    else:
        return None

# pre-process images
# def preprocess_image(np_image, size=preprocess_image_size):
def preprocess_image(most_recent_image, size=preprocess_image_size):
    """Load an image, resize, and flatten it."""
    image = Image.open(most_recent_image)
    image = image.resize(size)
    if image.mode != 'L':
        image = image.convert('L')
    np_image = np.array(image).flatten()
    return np_image

# Set up argument parser
parser = argparse.ArgumentParser(description='Run the Flask server for the Random Forest predictor.')
parser.add_argument('model_path', help='Path to the Random Forest model file')
parser.add_argument('label_to_idx_path', help='Path to the label-to-index JSON file')
parser.add_argument('image_folder_path', help='Path to the folder containing the images for the predictor.')

# Parse arguments
args = parser.parse_args()

# Assign arguments to variables
model_path = args.model_path
label_to_idx_path = args.label_to_idx_path
image_folder_path = args.image_folder_path

# Load model and label mapping
model = joblib.load(model_path)
with open(label_to_idx_path, 'r') as f:
    label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

@app.route('/predict', methods=['GET'])
def predict():

    # find latest image file
    most_recent_image = find_most_recent_image(image_folder_path)

    if not most_recent_image:
        return jsonify({'error': 'No image files found'}), 404

    # preprocessing should match worm_detector_RandomForest_train.py
    processed_image = preprocess_image(most_recent_image)
    prediction = model.predict([processed_image])[0]
    predicted_label = idx_to_label[prediction]
    return jsonify({'prediction': 1 if predicted_label == 'enough' else 0})
    # return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
