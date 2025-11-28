from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and data
model = None
df = None

def load_model_and_data():
    """Load the trained model and dataset"""
    global model, df
    
    try:
        # Load and preprocess the dataset (same as your original script)
        df = pd.read_csv('DailyDelhiClimateTest.csv')
        
        # Convert date to day-of-year for numerical feature
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Fix outlier in meanpressure
        df.loc[df['meanpressure'] < 900, 'meanpressure'] = df['meanpressure'].mean()
        
        # Train the model (same as your original script)
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        
        # Features & Label
        X = df[['day_of_year', 'humidity', 'wind_speed', 'meanpressure']]
        y = df['meantemp']
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save model coefficients for reference
        print(f"Model trained successfully!")
        print(f"Coefficients: {model.coef_}")
        print(f"Intercept: {model.intercept_}")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def get_day_of_year(date_str):
    """Convert date string to day of year"""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        return date.timetuple().tm_yday
    except:
        return None

def validate_inputs(data):
    """Validate input parameters"""
    errors = []
    
    day_of_year = data.get('day_of_year')
    humidity = data.get('humidity')
    wind_speed = data.get('wind_speed')
    pressure = data.get('pressure')
    
    if day_of_year is None or not (1 <= day_of_year <= 365):
        errors.append("Day of year must be between 1 and 365")
    
    if humidity is None or not (0 <= humidity <= 100):
        errors.append("Humidity must be between 0 and 100")
    
    if wind_speed is None or not (0 <= wind_speed <= 50):
        errors.append("Wind speed must be between 0 and 50 km/h")
    
    if pressure is None or not (900 <= pressure <= 1050):
        errors.append("Pressure must be between 900 and 1050 hPa")
    
    return errors

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for temperature prediction"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract and validate inputs
        date = data.get('date')
        humidity = float(data.get('humidity'))
        wind_speed = float(data.get('wind_speed'))
        pressure = float(data.get('pressure'))
        
        # Convert date to day of year
        day_of_year = get_day_of_year(date)
        if day_of_year is None:
            return jsonify({'error': 'Invalid date format'}), 400
        
        # Validate inputs
        input_data = {
            'day_of_year': day_of_year,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': pressure
        }
        
        errors = validate_inputs(input_data)
        if errors:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400
        
        # Make prediction using the trained model
        input_df = pd.DataFrame([[day_of_year, humidity, wind_speed, pressure]], 
                               columns=['day_of_year', 'humidity', 'wind_speed', 'meanpressure'])
        
        predicted_temp = model.predict(input_df)[0]
        
        # Get weather condition based on temperature
        weather_condition = get_weather_condition(predicted_temp)
        
        # Return prediction result
        result = {
            'temperature': round(float(predicted_temp), 2),
            'weather_condition': weather_condition,
            'inputs': input_data,
            'model_info': {
                'algorithm': 'Linear Regression',
                'features_used': ['day_of_year', 'humidity', 'wind_speed', 'meanpressure']
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def get_weather_condition(temp):
    """Get weather condition based on temperature"""
    if temp < 0:
        return '❄️ Freezing Cold'
    elif temp < 10:
        return '🧥 Cold'
    elif temp < 20:
        return '🌤️ Cool'
    elif temp < 30:
        return '☀️ Warm'
    elif temp < 35:
        return '🔥 Hot'
    else:
        return '🌋 Very Hot'

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the trained model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Calculate model metrics
        from sklearn.metrics import mean_squared_error, r2_score
        
        X = df[['day_of_year', 'humidity', 'wind_speed', 'meanpressure']]
        y = df['meantemp']
        
        # Make predictions on the full dataset
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        info = {
            'algorithm': 'Linear Regression',
            'features': ['day_of_year', 'humidity', 'wind_speed', 'meanpressure'],
            'target': 'meantemp',
            'coefficients': {
                'day_of_year': float(model.coef_[0]),
                'humidity': float(model.coef_[1]),
                'wind_speed': float(model.coef_[2]),
                'meanpressure': float(model.coef_[3])
            },
            'intercept': float(model.intercept_),
            'metrics': {
                'mse': float(mse),
                'r2_score': float(r2),
                'accuracy_percent': round(float(r2 * 100), 1)
            },
            'dataset_info': {
                'total_samples': len(df),
                'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
                'temperature_range': f"{df['meantemp'].min():.1f}°C to {df['meantemp'].max():.1f}°C"
            }
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/api/historical-data', methods=['GET'])
def historical_data():
    """Get historical temperature data for chart"""
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        # Sample data for chart (every 7th day to reduce data size)
        chart_data = df.iloc[::7].copy()
        
        # Convert to JSON-serializable format
        data = []
        for _, row in chart_data.iterrows():
            data.append({
                'day_of_year': int(row['day_of_year']),
                'date': row['date'].strftime('%Y-%m-%d'),
                'actual_temp': float(row['meantemp']),
                'humidity': float(row['humidity']),
                'wind_speed': float(row['wind_speed']),
                'pressure': float(row['meanpressure'])
            })
        
        return jsonify({
            'data': data,
            'total_points': len(data)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get historical data: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Weather Prediction API Server...")
    
    # Load model and data
    if load_model_and_data():
        print("✅ Model and data loaded successfully!")
        print("🌤️  Weather Prediction API is ready!")
        print("📊 Available endpoints:")
        print("   GET  /api/model-info - Get model information")
        print("   POST /api/predict - Make temperature prediction")
        print("   GET  /api/historical-data - Get historical data")
        print("\n🚀 Starting server on http://localhost:5000")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model and data. Exiting...")
        exit(1)
