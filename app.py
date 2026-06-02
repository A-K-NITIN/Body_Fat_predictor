from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import uvicorn

warnings.filterwarnings('ignore')

app = FastAPI(title="Body Fat Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Train model on startup ----------
df = pd.read_csv('bodyfat.csv')
df = df.dropna()

X = df.drop('BodyFat', axis=1)
y = df['BodyFat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = float(r2_score(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
mean_cv = float(cv_scores.mean())

# Retrain on full dataset for predictions
model.fit(X, y)

print(f"[OK] Model ready | R2: {r2:.3f} | RMSE: {rmse:.2f} | CV Accuracy: {mean_cv*100:.1f}%")


# ---------- Pydantic schema ----------
class BodyMeasurements(BaseModel):
    Density: float
    Age: float
    Weight: float
    Height: float
    Neck: float
    Chest: float
    Abdomen: float
    Hip: float
    Thigh: float
    Knee: float
    Ankle: float
    Biceps: float
    Forearm: float
    Wrist: float


# ---------- Routes ----------
@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/api/info")
def get_info():
    return {
        "r2_score": round(r2, 4),
        "rmse": round(rmse, 2),
        "mean_cv_accuracy": round(mean_cv * 100, 2),
        "dataset_size": len(df),
        "features": X.columns.tolist(),
    }


@app.post("/api/predict")
def predict(data: BodyMeasurements):
    features = np.array([[
        data.Density, data.Age, data.Weight, data.Height,
        data.Neck, data.Chest, data.Abdomen, data.Hip,
        data.Thigh, data.Knee, data.Ankle, data.Biceps,
        data.Forearm, data.Wrist,
    ]])

    raw = float(model.predict(features)[0])
    body_fat = round(max(2.0, min(50.0, raw)), 2)

    if body_fat < 6:
        category, desc = "Essential Fat", "Below the minimum healthy threshold"
    elif body_fat < 14:
        category, desc = "Athletic", "Typical range for athletes"
    elif body_fat < 18:
        category, desc = "Fit", "Good fitness level, above average"
    elif body_fat < 25:
        category, desc = "Average", "Acceptable range for most adults"
    else:
        category, desc = "Above Average", "Consider consulting a health professional"

    return {
        "body_fat_percentage": body_fat,
        "category": category,
        "description": desc,
    }


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
