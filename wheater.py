import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and preprocess the dataset
df = pd.read_csv('DailyDelhiClimateTest.csv')

# Convert date to day-of-year for numerical feature
df['date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df['date'].dt.dayofyear

# Fix outlier in meanpressure
df.loc[df['meanpressure'] < 900, 'meanpressure'] = df['meanpressure'].mean()

# 2. Features & Label
X = df[['day_of_year', 'humidity', 'wind_speed', 'meanpressure']]
y = df['meantemp']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Model Trained. Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}\n")

# 6. Show predictions
for i in range(len(X_test)):
    print(f"Input: {X_test.iloc[i].to_dict()} => Predicted Temp: {y_pred[i]:.2f} °C")

# 7. Custom Input with validation
print("\n🔍 Predict temperature based on your own input:")
day_of_year = float(input("Enter day of year (1-365): "))
humidity = float(input("Enter humidity (%): "))
wind_speed = float(input("Enter wind speed (km/h): "))
pressure = float(input("Enter pressure (hPa): "))

# Validate inputs
if not (1 <= day_of_year <= 365):
    raise ValueError("Day of year must be between 1 and 365")
if not (0 <= humidity <= 100):
    raise ValueError("Humidity must be between 0 and 100")
if not (0 <= wind_speed <= 50):
    raise ValueError("Wind speed must be between 0 and 50 km/h")
if not (900 <= pressure <= 1050):
    raise ValueError("Pressure must be between 900 and 1050 hPa")

# Create a DataFrame for prediction
custom_input = pd.DataFrame([[day_of_year, humidity, wind_speed, pressure]], 
                            columns=['day_of_year', 'humidity', 'wind_speed', 'meanpressure'])
predicted_temp = model.predict(custom_input)
print(f"🌡️ Predicted Temperature: {predicted_temp[0]:.2f} °C")

# 8. Visualization
plt.scatter(df['day_of_year'], df['meantemp'], color='blue', label='Actual Temp')
plt.plot(df['day_of_year'], model.predict(df[['day_of_year', 'humidity', 'wind_speed', 'meanpressure']]), 
         color='red', linestyle='--', label='Predicted')
plt.xlabel('Day of Year')
plt.ylabel('Temperature (°C)')
plt.title('Temperature vs Day of Year Prediction')
plt.legend()
plt.grid(True)
plt.show()