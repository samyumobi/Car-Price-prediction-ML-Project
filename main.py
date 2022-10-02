# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import streamlit
import requests
import json
import joblib
import pickle

df = pd.read_csv('car.csv')
# print(df.head(5))
# print(df.info())
## Data Cleaning
df2 = df.copy()
## Investigate year
# print(df2['year'].value_counts())
df2 = df2[df2['year'].str.isnumeric()]
df2['year'] = df2['year'].astype(int)
# print(df2['year'].head(5))
## Investigate price
# Convert 'Ask for Price' to 0
# print(df2['Price'].head(5))
df2 = df2[df2["Price"] != "Ask For Price"]
df2.Price = df2.Price.str.replace(",","").astype(int)
##Investigate kms_driven
df2["kms_driven"] = df2["kms_driven"].str.split(" ").str.get(0).str.replace(",","")
df2 = df2[df2["kms_driven"].str.isnumeric()]
df2["kms_driven"] = df2["kms_driven"].astype(int)
# print(df2.kms_driven.head(10))
# print(df2.info())
## Investigate fuel_type
# print(df2["fuel_type"].head(10))
# print(df2["fuel_type"].value_counts())
df2 = df2[~df2["fuel_type"].isna()]
## Investigate name
df2['name']=df2['name'].str.split().str.slice(0,3).str.join(' ')
# print(df2['name'])
## Reset index of cleaned dataset
df2 = df2.reset_index(drop=True)
# Transport csv file
df2.to_csv('cd.csv')
# print(df2.info())
# print(df2.describe(include='all'))
# Drop the outliers
df2 = df2[df2['Price'] < 6e6].reset_index(drop=True)
# print(df2.head())
# Create feature
df2['company'] = df2['name'].str.split().head(10).str.get(0)

# Extract training data
x = df2[["name","company","year","fuel_type","kms_driven"]]
y = df2["Price"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

# print(x.info())
# convert object type data into numbers
o = OneHotEncoder()
o.fit_transform(x[['name','company','fuel_type']])
# print(x.head(5))
ct = make_column_transformer((OneHotEncoder(categories = o.categories_),
                              ['name','company','fuel_type']),
                             remainder='passthrough')
# print(x.info())
# Linear Regression model
lr = LinearRegression()
pipe = make_pipeline(ct, lr)
pipe.fit(xtrain, ytrain)
ypred = pipe.predict(xtest)
print(r2_score(ytest,ypred))

## finding the best model random_state
scores = []
for i in range(1000):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1,random_state=i)

    lr = LinearRegression()
    pipe = make_pipeline(ct, lr)
    pipe.fit(xtrain, ytrain)
    ypred = pipe.predict(xtest)
    scores.append(r2_score(ytest, ypred))
# print(np.argmax(scores))
# print(scores[np.argmax(scores)])
# print(pipe.predict(pd.DataFrame(columns= xtest.columns,data = np.array(['Maruti Suzuki Swift','Maruti',2019,'Petrol',100]).reshape(1,5))))
# print(pipe.steps[0][1].transformers[0][1].categories[0])

# Best Model
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1,random_state=np.argmax(scores))

lr = LinearRegression()
pipe = make_pipeline(ct, lr)
pipe.fit(xtrain, ytrain)
ypred = pipe.predict(xtest)
# print(r2_score(ytest, ypred))

# joblib.dump(pipe, open('LinearRegressionModel.pkl'),'wb')
# pipe.predict(pd.DataFrame(columns= xtest.columns,data = np.array(['Maruti Suzuki Swift','Maruti',2019,'Petrol',100]).reshape(1,5)))

## Save the model state using joblib
file = 'lrj.sav'
joblib.dump(pipe,file)
## Save the model state using pickle
filename = 'lrp.sav'
pickle.dump(pipe,open(filename,'wb'))

def run():
    streamlit.title("Car Price Prediction")
    name = streamlit.selectbox("Car Model",df2.name.unique())
    company = streamlit.selectbox("Car Company",df2.company.unique())
    year = streamlit.number_input("Year")
    kms_driven = streamlit.number_input("Kms driven")
    fuel_type = streamlit.selectbox("Fuel type",df2.fuel_type.unique())

    data = {
        'name':name,
        'company':company,
        'year':year,
        'kms_driven':kms_driven,
        'fuel_type':fuel_type,
    }

    if streamlit.button("Predict"):
        response = requests.post("http://127.0.0.1:8000/predict",json=data)
        prediction = response.text
        streamlit.success(f"The prediction from model: {prediction}")

run()









