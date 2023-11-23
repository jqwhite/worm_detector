import numpy as np
import joblib
import json
import os
import glob
import argparse
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

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
    # np_image = np.array(image_list)
    # image = Image.fromarray(np_image)
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
# model = joblib.load('models/best_random_forest_model_more_data_2023-11-16_1527.pkl')
# with open('models/random_forest_labels_to_idx_more_data.json', 'r') as f:
#     label_to_idx = json.load(f)
#     idx_to_label = {v: k for k, v in label_to_idx.items()}
model = joblib.load(model_path)
with open(label_to_idx_path, 'r') as f:
    label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

# def convert_to_image(np_image):
#     if len(np_image.shape) == 2:  # Grayscale image
#         return Image.fromarray(np_image, 'L')
#     elif len(np_image.shape) == 3 and np_image.shape[2] == 3:  # RGB image
#         return Image.fromarray(np_image, 'RGB')
#     else:
#         raise ValueError("Unsupported image format")



@app.route('/predict', methods=['GET'])
def predict():
    # request_data = request.json
    # image_data = request_data['image_data']
    # image_shape = request_data['image_shape']

    # np_image = np.array(image_data).astype(np.uint8).reshape(image_shape)

    # # Convert numpy array to PIL Image
    # try:
    #     img = convert_to_image(np_image)
    # except ValueError as e:
    #     return jsonify({'error': str(e)}), 400  # Bad request

    # Further preprocessing...
    most_recent_image = find_most_recent_image(image_folder_path)

    if not most_recent_image:
        return jsonify({'error': 'No image files found'}), 404

    processed_image = preprocess_image(most_recent_image)#.reshape(1,-1)
    # prediction = model.predict(processed_image.reshape(1,-1))[0]
    prediction = model.predict([processed_image])[0]
    # print(prediction)
    predicted_label = idx_to_label[prediction]
    # return jsonify({'prediction': 1 if predicted_label == 'enough' else 0})
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
