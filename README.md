# worm_detector

## Overview
Image classifier to detect worms in Kyle's microfluidic device for automatic embryo extraction.
Intended to trigger the bleach squence if the classifier detects that there are enough worms in the device.

## Usage

1. Place a model file and a label-to-index file somewhere convenient (I use a `models/`  folder)

`best_random_forest_model_more_data_2023-11-16_1527.pkl`

`random_forest_labels_to_idx_more_data.json`

2. make a folder for temporary images to feed the predictor

`temp_images`

3. Start the predictor service

~~~
python worm_detector_RandomForest_predictor.py models/best_random_forest_model_more_data_2023-11-16_1527.pkl models/random_forest_labels_to_idx_more_data.json temp_images
~~~

4. Write an images to the temporary folder.
5. Query the predictor
~~~
url = 'http://127.0.0.1:5000/predict'  # URL where the Flask app is running
response = requests.get(url)
response.json()['prediction']
~~~


## Classes

Classes are:

- "enough" : there are enough worms in  the device to trigger the bleach.
- "bubble" : there is a bubble in the device.  
- "not enough" : not enough worms were detected to justify a bleach.

The latest classifier is a RandomForest trained on equal classes of ~200 of each.

## Training

Training is performed by `worm_detector_RandomForest_train.ipynb` notebook.
The model and label-to-index files are written to a folder.  Currently in the `models/` folder.  
The models are timestamped with UTC time.

## Prediction

- The predictor runs as a lightweight Flask web service in the background.
- It loads the model and label-to-index dictionary once. 
- While it is running, it monitors a temporary directory for `.tif` image files and runs a prediction when queried.
- When the http server recieves a `GET` query, it scans the temporary directory for the latest `.tif` image, loads the image, and runs the prediction using `model.predict()`.
- It returns **1** if it predicts the "enough" class, otherwise it returns **0**.



The server takes the model, label-to-index, and image folder paths as command-line parameters upon startup. 
It  should be started like this: 

~~~
python worm_detector_RandomForest_predictor.py models/best_random_forest_model_more_data_2023-11-16_1527.pkl models/random_forest_labels_to_idx_more_data.json temp_images
~~~

## LabView Info

From ChatGPT.

To query the predictor from LabView, you'll need to use LabView's HTTP Client VIs (Virtual Instruments) to perform the equivalent of a GET request to your Flask server and then parse the JSON response to get the prediction. Here's a general outline of how you can do it:

### Step 1: Set Up HTTP Client in LabView
1. **Open a New VI**: In LabView, start by opening a new or existing VI (Virtual Instrument).

2. **HTTP Client Palette**: Navigate to the HTTP Client palette. You can find this under `Data Communication -> Protocols -> HTTP Client` in the functions palette.

3. **HTTP Get VI**: Place an `HTTP Get` VI on the block diagram. This VI is used to make GET requests.

4. **Configure URL**: Wire a string constant to the URL input of the `HTTP Get` VI and enter your Flask app's URL (e.g., `http://127.0.0.1:5000/predict`).

### Step 2: Handle the Response
1. **Parse JSON**: After making the GET request, you'll receive the response from the server. Use the `Unflatten from JSON` VI (found under `Data Communication -> Format & Parse -> JSON`) to parse the JSON response.

2. **Extract Prediction**: The JSON response will contain the prediction data, which you can extract using the appropriate JSON parsing functions in LabView.

3. **Error Handling**: Make sure to include error handling for network errors or issues with the HTTP request.

### Step 3: Display or Use the Prediction
1. **Display the Result**: You can display the prediction on the front panel using indicators, or use the data as needed in your LabView application.

2. **Test the Setup**: Run your VI to test the interaction with your Flask app. Ensure that the Flask app is running and accessible from where the LabView application is running.

### Additional Considerations
- **Network Accessibility**: Ensure that the Flask app is network-accessible from the machine where the LabView application is running.
- **Firewall and Security Settings**: Check any firewall or security settings that might block the LabView application from accessing the Flask app.
- **Data Format Consistency**: Ensure that the data format expected by LabView matches the format of the JSON response from the Flask app.

By following these steps, you should be able to query your Flask-based predictor from LabView and process the response to get the prediction results.






