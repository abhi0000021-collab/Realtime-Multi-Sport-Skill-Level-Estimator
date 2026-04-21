import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

data = pd.read_csv("datasets/skill_data.csv")

X = data.drop("score",axis=1)
y = data["score"]

model = RandomForestRegressor()
model.fit(X,y)

joblib.dump(model,"models/skill_model.pkl")

print("Model trained and saved")
