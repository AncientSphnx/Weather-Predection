# Weather Prediction

A simple **Weather Prediction** project using **Linear Regression** to predict daily temperature based on climate data from Delhi. The project also includes visualization and a custom input feature for interactive predictions.

---

## 📂 Project Structure

- `DailyDelhiClimateTest.csv` – Dataset containing historical weather data (temperature, humidity, wind speed, pressure).  
- `wheater.py` – Python script that loads data, trains a Linear Regression model, predicts temperatures, and visualizes results.  
- `README.md` – Project documentation.

---

## 🛠 Features

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
   - Prints **Mean Squared Error (MSE)** and **R² Score**.  
   - Displays predictions for test data.  

4. **Custom Input Prediction**  
   - Users can input day, humidity, wind speed, and pressure to predict temperature.  
   - Input validation ensures realistic values.

5. **Visualization**  
   - Plots actual vs predicted temperatures across days of the year.

---

## ⚙️ How to Run

1. Make sure you have **Python 3.x** installed.  
2. Install dependencies:
```bash
pip install pandas scikit-learn matplotlib
```
3. Run the script:
```
python wheater.py
```
4. Follow the prompts to input custom weather data for predictions.

📊 Example Output
```
✅ Model Trained. Mean Squared Error: 2.45
R² Score: 0.87

Input: {'day_of_year': 123, 'humidity': 70, 'wind_speed': 5, 'meanpressure': 1015} => Predicted Temp: 27.35 °C

🌡️ Predicted Temperature: 27.35 °C
```

📈 Visualization

- The script generates a scatter plot comparing actual vs predicted temperatures.

📚 Dataset

The dataset DailyDelhiClimateTest.csv contains the following columns:
- date – Date of record
- meantemp – Mean temperature (°C)
- humidity – Humidity (%)
- wind_speed – Wind speed (km/h)
- meanpressure – Atmospheric pressure (hPa)

💡 Notes
- Linear Regression assumes a linear relationship; accuracy may vary with seasonal or extreme weather changes.
- Ensure all dependencies are installed before running the script.
