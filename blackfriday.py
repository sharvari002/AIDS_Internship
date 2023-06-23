import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dftrain = pd.read_csv("C:/CIDSA/Black Friday Sales/train.csv")
dftest = pd.read_csv("C:/CIDSA/Black Friday Sales/test.csv")

dftrain.head()
dftrain.info()

dftrain.shape
dftrain.describe()
dftrain.isnull().sum()

dftest.head()
dftest.info()
dftest.describe()
dftest.isnull().sum()
dftrain = dftrain.drop(['User_ID','Product_ID'],axis=1)
dftrain['Product_Category_2']=dftrain['Product_Category_2'].fillna(dftrain['Product_Category_2'].mean())
dftrain['Product_Category_3']=dftrain['Product_Category_3'].fillna(dftrain['Product_Category_3'].mean())
dftrain.isnull().sum()

dftest = dftest.drop(['User_ID','Product_ID'],axis=1)
dftest['Product_Category_2']=dftest['Product_Category_2'].fillna(dftest['Product_Category_2'].mean())
dftest['Product_Category_3']=dftrain['Product_Category_3'].fillna(dftest['Product_Category_3'].mean())

#Data Expolartion
plt.figure(figsize=(10,8))
sns.distplot(dftrain['Purchase'])
plt.title("purchase distribution")
plt.show()

dftrain['Gender'].value_counts()

plt.figure(figsize=(8,6))
sns.countplot(x='Gender',data=dftrain , palette='husl')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='Marital_Status',data=dftrain , palette ='husl')
plt.show()

dftrain['Stay_In_Current_City_Years']=dftrain['Stay_In_Current_City_Years'].str.replace('+','')

dftrain['Stay_In_Current_City_Years'].unique()

dftrain.groupby("Marital_Status")["Purchase"].mean()

plt.figure(figsize=(8,6))
sns.countplot(x ='Occupation', data = dftrain, palette ='husl')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x ='City_Category', data = dftrain, palette ='husl')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x ='Stay_In_Current_City_Years', data= dftrain, palette ='husl')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x ='Age', data = dftrain, palette ='husl')
plt.show()

dftrain

dftrain[['Product_Category_2']]=dftrain[['Product_Category_2']].astype(int)
dftrain[['Product_Category_3']]=dftrain[['Product_Category_3']].astype(int)
dftrain.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dftrain['Gender']=le.fit_transform(dftrain['Gender'])

dftrain['Age']=le.fit_transform(dftrain['Age'])

dftrain['City_Category']=le.fit_transform(dftrain['City_Category'])

dftrain


corr = dftrain.corr()
fig,ax=plt.subplots(figsize=(10,10))
sns.heatmap(corr , annot=True ,ax=ax)

x=dftrain.drop('Purchase',axis=1)
y=dftrain['Purchase']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
regressor  = LinearRegression()
regressor .fit(x_train_scaled,y_train)

prediction1 = regressor.predict(x_test_scaled)


from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
r2_score(y_test, prediction1)

mean_absolute_error(y_test, prediction1)

from sklearn.tree import DecisionTreeRegressor
dt= DecisionTreeRegressor(random_state = 0)
dt.fit(x_train_scaled, y_train)

prediction2 = dt.predict(x_test_scaled)
r2_score(y_test , prediction2)

mean_absolute_error(y_test, prediction2)

user_input=np.array([[0,0,10,0,2,0,1,6,14]])
prediction= dt.predict(user_input)
print(prediction)


