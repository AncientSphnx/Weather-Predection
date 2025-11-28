# Weather Prediction AI

An amazing weather prediction web application with machine learning backend and beautiful UI.

## Features

- **Real ML Predictions**: Uses your actual trained Linear Regression model from `wheater.py`
- **Beautiful Weather UI**: Animated weather backgrounds, glass morphism effects, and smooth transitions
- **Interactive Forms**: Real-time temperature predictions with validation
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Fallback Mode**: Works even without backend using simplified predictions

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Backend Server
```bash
python app.py
```

### 3. Open in Browser
Visit `http://localhost:5000` to see the amazing UI!

## Project Structure

```
├── app.py              # Flask backend with real ML model
├── wheater.py          # Your original ML training script
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Main UI template
├── static/
│   ├── styles.css     # Beautiful weather-themed styling
│   └── script.js      # Interactive frontend JavaScript
└── data/
    └── DailyDelhiClimateTest.csv
```

## API Endpoints

- `GET /` - Main UI
- `POST /api/predict` - Make temperature prediction
- `GET /api/model-info` - Get model statistics
- `GET /api/historical-data` - Get historical temperature data

## How It Works

1. **Backend**: Flask loads your trained Linear Regression model from `wheater.py`
2. **Frontend**: Beautiful UI with weather animations and interactive forms
3. **API**: Real-time communication between UI and ML model
4. **Fallback**: If backend is unavailable, uses simplified local predictions

## Original ML Features

The underlying ML model (`wheater.py`) provides:

1. **Data Preprocessing**  
   - Converts dates to numeric `day_of_year` for regression.  
   - Handles outliers in pressure data.  

2. **Machine Learning**  
   - Trains a **Linear Regression** model using:  
     - Day of year  
     - Humidity  
     - Wind speed  
     - Mean pressure  
   - Splits data into train and test sets for evaluation.  

3. **Prediction & Evaluation**  
   - **Mean Squared Error (MSE)** and **R² Score** metrics.  
   - Custom input predictions with validation.  

## Technologies Used

- **Backend**: Flask, scikit-learn, pandas
- **Frontend**: HTML5, CSS3, JavaScript
- **Design**: Glass morphism, weather animations, responsive grid
- **ML**: Linear Regression with Delhi Climate dataset

Enjoy your amazing weather prediction app! 🌤️
