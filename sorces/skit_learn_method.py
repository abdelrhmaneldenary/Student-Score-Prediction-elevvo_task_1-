import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_dir, 'data', 'clean_data.csv')
model_path = os.path.join(base_dir, 'models', 'student_score_model.pkl')

def train_and_evaluate():
    df = pd.read_csv(data_path)
    x = df[['Hours_Studied']]
    y = df['Exam_Score']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    
    joblib.dump(model, model_path)
    
    return X_train, X_test, y_train, y_test, y_predict, mse, r2
