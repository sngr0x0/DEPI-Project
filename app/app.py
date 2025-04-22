# app.py
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from retrain import retraining

app = Flask(__name__)

# Constants
MODEL_PATH = "best_model.pkl"
HISTORICAL_DATA = "../data/historical_data.csv"
NEW_DATA = "../data/new_data.csv"
TRAINING_DATA = "../data/training_data.csv"

# Load the model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Model file not found at {MODEL_PATH}. Please place your model file in the specified location.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/admin')
def admin():
    model_metrics = {}
    model_status = "Not Available"
    
    # Check if model exists and get metrics
    if model is not None and hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
        # Try to extract metrics from model if available
        try:
            # Load test data or results if stored
            if os.path.exists('best_model_metrics.pkl'):
                model_metrics = joblib.load('best_model_metrics.pkl')
                
                # Set status indicators based on metrics
                if 'r2' in model_metrics:
                    r2 = model_metrics['r2']
                    if r2 > 0.8:
                        model_status = "Excellent"
                    elif r2 > 0.7:
                        model_status = "Good"
                    elif r2 > 0.5:
                        model_status = "Decent"
                    else:
                        model_status = "Poor"
            else:
                model_metrics = {
                    "note": "Model loaded but detailed metrics not available"
                }
                model_status = "Unknown"
        except Exception as e:
            model_metrics = {"error": str(e)}
            model_status = "Error"
    
    # Check if historical data exists and get sample
    sample_data = None
    if os.path.exists(HISTORICAL_DATA):
        try:
            sample_data = pd.read_csv(HISTORICAL_DATA, encoding="ISO-8859-1", nrows=5)
        except Exception as e:
            sample_data = f"Error loading sample data: {str(e)}"
    
    return render_template(
        'admin.html', 
        model_metrics=model_metrics,
        model_status=model_status,
        sample_data=sample_data
    )

@app.route('/upload_data', methods=['POST'])
def upload_data():
    try:
        if 'datafile' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'})
        
        file = request.files['datafile']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
        
        # Save uploaded file as new_data.csv
        # If an older file existed, it'll be overriden by the uploaded value
        os.makedirs(os.path.dirname(NEW_DATA), exist_ok=True)
        file.save(NEW_DATA)
        
        return jsonify({'success': True, 'message': 'File uploaded successfully'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/retrain_model', methods=['POST'])
def retrain():
    try:
        # Check if new data file exists
        if not os.path.exists(NEW_DATA):
            return jsonify({'success': False, 'error': 'No new data file found'})
        
        # Create historical data file if it doesn't exist (for first run)
        if not os.path.exists(HISTORICAL_DATA):
            os.makedirs(os.path.dirname(HISTORICAL_DATA), exist_ok=True)
            pd.DataFrame(columns=[
                'Invoice', 'StockCode', 'Description', 'Quantity', 
                'InvoiceDate', 'Price', 'Customer ID', 'Country'
            ]).to_csv(HISTORICAL_DATA, index=False)
        
        # Make directory for training data if it doesn't exist
        os.makedirs(os.path.dirname(TRAINING_DATA), exist_ok=True)
        
        retraining(HISTORICAL_DATA, NEW_DATA, MODEL_PATH)
        
        # Reload the model
        global model
        model = joblib.load(MODEL_PATH)
        
        # Store metrics if available (from recent evaluation)
        try:
            # This assumes the retrain_model function returns or saves metrics
            # If metrics are not directly accessible, this part would need adjustment
            if os.path.exists('model_metrics.pkl'):
                metrics = joblib.load('model_metrics.pkl')
                return jsonify({
                    'success': True, 
                    'message': 'Model retrained successfully',
                    'metrics': metrics
                })
            else:
                return jsonify({
                    'success': True, 
                    'message': 'Model retrained successfully, but metrics not available'
                })
        except Exception as e:
            return jsonify({
                'success': True, 
                'message': f'Model retrained but error retrieving metrics: {str(e)}'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check if model file exists.'}), 400
    
    try:
        # Get season value from form
        season = request.form.get('season', 'summer') 
        # 2nd argument, 'summer', is the default value that'll be used in case no value is provided.
        
        # Set all season flags to 0 initially
        is_summer = 0
        is_spring = 0
        is_fall = 0
        is_holiday_season = 0
        
        # Set the appropriate flag to 1 based on selected season
        if season == 'summer':
            is_summer = 1
            print("season is summer")
        elif season == 'spring':
            is_spring = 1
            print("season is spring")
        elif season == 'fall':
            is_fall = 1
            print("season is fall")
        elif season == 'holiday':
            is_holiday_season = 1
            print("season is holiday")
        
        # Get input features from the form
        features = {
            'ProductCategory': request.form.get('ProductCategory'),
            'Year': int(request.form.get('Year')),
            'Month': int(request.form.get('Month')),
            'WeekOfYear': int(request.form.get('WeekOfYear')),
            'Prev_Week_Revenue': float(request.form.get('Prev_Week_Revenue')),
            'Prev_2_Week_Revenue': float(request.form.get('Prev_2_Week_Revenue')),
            'Prev_3_Week_Revenue': float(request.form.get('Prev_3_Week_Revenue')),
            'IsSummer': is_summer,
            'IsHolidaySeason': is_holiday_season,
            'IsSpring': is_spring,
            'IsFall': is_fall
        } 
        print(features['ProductCategory'])
        # Create a dataframe with the input features
        input_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Return prediction
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'inputs': features
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)