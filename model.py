import numpy as np
import pandas as pd
import pickle
car=pd.read_csv('bmw_pricing_challenge.csv')
car['registration_date']=pd.to_datetime(car['registration_date'])
car['sold_at']=pd.to_datetime(car['sold_at'])
car.at[2938, 'mileage']= 64
import datetime
from dateutil.relativedelta import relativedelta
from datetime import date
car['AgeOfCar'] = car['sold_at'].sub(car['registration_date'], axis=0)
car['AgeOfCar'] = car['AgeOfCar'] / np.timedelta64(1, 'Y')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
car["model_key"] = le.fit_transform(car['model_key'])
car["registration_date"] = le.fit_transform(car['registration_date'])
car["fuel"] = le.fit_transform(car['fuel'])
car["paint_color"] = le.fit_transform(car['paint_color'])
car["car_type"] = le.fit_transform(car['car_type'])
car["feature_1"] = le.fit_transform(car['feature_1'])
car["feature_2"] = le.fit_transform(car['feature_2'])
car["feature_3"] = le.fit_transform(car['feature_3'])
car["feature_4"] = le.fit_transform(car['feature_4'])
car["feature_5"] = le.fit_transform(car['feature_5'])
car["feature_6"] = le.fit_transform(car['feature_6'])
car["feature_7"] = le.fit_transform(car['feature_7'])
car["feature_8"] = le.fit_transform(car['feature_8'])
car["sold_at"] = le.fit_transform(car['sold_at'])
y=car['price']
x=car.drop(['price'],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)
x_train=x_train.drop(['maker_key','paint_color','registration_date','feature_7','sold_at'],axis=1)
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(x_train,y_train)
pickle.dump(forest_reg,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
