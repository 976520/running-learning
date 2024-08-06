import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import re

data = {
    '날짜': ['07월 30일', '07월 31일', '08월 01일', '08월 02일', '08월 03일', '08월 04일', '08월 05일', '08월 06일', '08월 07일'],
    '뛴 거리 (km)': [5.33, 5.12, 6.0, 5.36, 5.02, 4.55, 5.0, 5.0, 4.0],
    '평균 페이스': ["6'25''", "5'09''", "5'07''", "4'55''", "5'01''", "6'40''", "4'50''", "4'53''", "4'52''"],
    '뛴 시간': ["34분 09초", "26분 20초", "30분 41초", "26분 19초", "25분 09초", "30분 19초", "24분 10초", "24분 26초", "19분 28초"],
    '소모된 칼로리': [368, 383, 445, 404, 377, 313, 376, 376, 301]
}

df = pd.DataFrame(data)

def time_to_minutes(time_str):
    minutes, seconds = map(int, re.findall(r'\d+', time_str))
    return minutes + seconds / 60

df['평균 페이스'] = df['평균 페이스'].apply(time_to_minutes)
df['뛴 시간'] = df['뛴 시간'].apply(time_to_minutes)
df['날짜'] = pd.to_datetime(df['날짜'], format='%m월 %d일')

df.set_index('날짜', inplace=True)

features = df[['뛴 거리 (km)', '평균 페이스', '뛴 시간']]
targets = df[['평균 페이스', '소모된 칼로리']]

features_shifted = features.shift(-1).iloc[:-1]
targets_shifted = targets.shift(-1).iloc[:-1]

X_train, X_test, y_train, y_test = train_test_split(features_shifted, targets_shifted, test_size=0.2, shuffle=False)

model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

last_data = features.iloc[-1:].copy()

next_day_pred = model.predict(last_data)

def minutes_to_time_str(minutes_float):
    minutes = int(minutes_float)
    seconds = int((minutes_float - minutes) * 60)
    return f"{minutes}'{seconds:02d}''"

predicted_pace = minutes_to_time_str(next_day_pred[0][0])
predicted_calories = next_day_pred[0][1]

print("평균 페이스: {}".format(predicted_pace))
print("소모할 칼로리: {:.2f} ".format(predicted_calories))
