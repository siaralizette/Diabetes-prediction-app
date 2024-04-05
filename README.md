
## Project description

**The Diabetes Prediction** App is a Python application that leverages machine learning to predict the likelihood of an individual having diabetes based on input features such as pregnancies, glucose levels, blood pressure, and more. The app provides a user-friendly graphical interface built with PyQt6, allowing users to input relevant data and receive a prediction. The underlying machine learning model, powered by scikit-learn's RandomForestClassifier, has been trained on a dataset to make accurate diabetes predictions.


## Prerequisites

matplotlib==3.8.2

pandas==2.1.4

PyQt6==6.6.1

PyQt6-Qt6==6.6.1

scikit-learn==1.3.2

seaborn~=0.13.2


## Installation

Clone the repository from yourgit: git clone https://mygit.th-deg.de/ss03006/Recomendation

Navigate to the project directory: cd <project_directory>

Create a virtual environment: python -m venv venv

Activate the virtual environment:

On Windows: venv\Scripts\activate

On Unix or MacOS: source venv/bin/activate

Install dependencies: pip install -r requirements.txt


## Basic usage


**Run the App:**

Execute the main script: python <main_script_name>.py

First a Window shows the data analysis, Click "OK"

The Diabetes Prediction App window will open and also the Data visualization

Input values for selected features (e.g., pregnancies, glucose) in the provided fields.

Click the "Predict Diabetes" button to see the probability of being diabetic.

The app displays the prediction result and a pie chart illustrating the probability distribution.


**Interpretation:**

The "Predict Diabetes" button triggers the machine learning model to make predictions based on the input data.

The result is displayed as the probability of being diabetic.

The accompanying pie chart visually represents the probability distribution between being diabetic and not being diabetic.

**Note:**

Ensure that the required libraries are installed in the virtual environment (refer to the requirements.txt file).

Handle any potential errors, such as providing numeric values for input fields.
This application provides a simple and interactive way for users to explore diabetes predictions using machine learning.


## Implementation of requests

**Data Import:**

Data import is handled by the import_data method. It reads the data from a CSV file ('diabetes.csv').

**Data Analysis:**

Data analysis is performed in the analyze_data method.

The info, describe, and corr methods of the DataFrame are used to display information, summary statistics, and correlation matrix, respectively.

Additional data visualization is done in the visualize_data method, where histograms and a correlation heatmap are displayed in a separate window.

**Input Widgets:**

Input widgets are created in the create_input_widget method.

Different types of input widgets are used, including QSpinBox, QDoubleSpinBox, QSlider, and QDial.

**Scikit Training Model Algorithm:**

The Scikit-learn Random Forest Classifier is used for training the model in the train_model method.

**Output Canvas for Data Visualization:**

Two output canvases are used for data visualization. One is a hidden FigureCanvas for potential future use, and the other is in the separate visualization_window that displays histograms and a correlation heatmap.

**Statistical Metrics:**

The statistical metrics (count, mean, std) are included in the show_data_analysis method.

**Interactive Reaction to Input Parameter Changes:**

The predict_diabetes method is triggered when the "Predict Diabetes" button is clicked.

It collects the input values, predicts diabetes probabilities using the trained model, and updates the result display and pie chart.

**New Prediction with Visualization:**

The predict_diabetes method updates the prediction result and triggers the plot_probabilities_pie method to update the pie chart based on the new probabilities.
