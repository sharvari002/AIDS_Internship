import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from collections import Counter


df=pd.read_csv("C:/CDSAI/Walmart.csv")



X = df.drop('Store', axis=1)
X = X.drop('Weekly_Sales',axis=1)
Y = df['Weekly_Sales']
print(X)
print(Y)

le = LabelEncoder()

le.fit(df['Date'])
df['Date'] = le.transform(df['Date'])


'''
#feature
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScore = pd.concat([dfcolumns, dfscores], axis=1)
featuresScore.columns=['Store','Weekly_Sales']

print(featuresScore)

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
'''


feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(4).plot(kind='pie')
plt.show()



#random forest
rf = RandomForestClassifier()
df['Weekly_Sales']=pd.cut(df['Weekly_Sales'],3,labels=['0','1','2'])
df['Holiday_Flag']=pd.cut(df['Holiday_Flag'],3,labels=['0','1','2'])
df['Temprature']=pd.cut(df['Temprature'],3,labels=['0','1','2'])
df['Fuel_Price']=pd.cut(df['Fuel_Price'],3,labels=['0','1','2'])

print(df)

X = df.drop('Store', axis=1)
X = X.drop('Weekly_Sales',axis=1)
Y = df['Weekly_Sales']
print(Y)

print(df.isnull().sum())
print(df.notnull().sum())
print(Counter(Y))
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X, Y)
print(Counter(Y))




#log reg
logr = LogisticRegression()
pca = PCA(n_components=2)
X = df.drop('Store', axis=1)
X = X.drop('Weekly_Sales',axis=1)
Y = df['Weekly_Sales']
pca.fit(X)
X = pca.transform(X)
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=2, test_size=0.4)
logr.fit(X_train, Y_train)
Y_pred = logr.predict(X_test)
print(accuracy_score(Y_test, Y_pred))




#Boxplot
sns.boxplot(df['sepal_length'])
plt.show()
print(df['sepal_length'])
Q1 = df['sepal_length'].quantile(0.25)
Q3 = df['sepal_length'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 + 1.5*IQR
print(upper)
print(lower)
out1 = df['Weekly_Sales'] < lower.values
out1 = df['Weekly_Sales'] < upper.values
df['Weekly_Sales'].replace(out1, lower, inplace = True)
df['Weekly_Sales'].replace(out2, upper, inplace = True)
print(df['Weekly_Sales'])




X = df.drop('Store', axis=1)
X = X.drop('Weekly_Sales',axis=1)
Y = df['Weekly_Sales']
print(X)
print(Y)

print(df.isnull().sum())
print(df.notnull().sum())
print(Counter(Y))
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X, Y)
print(Counter(Y))



logr = LogisticRegression()
pca = PCA(n_components=2)
X = df.drop('Store', axis=1)
X = X.drop('Weekly_Sales',axis=1)
Y = df['Weekly_Sales']
pca.fit(X)
X = pca.transform(X)
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=2, test_size=0.4)
logr.fit(X_train, Y_train)
Y_pred = logr.predict(X_test)
print(accuracy_score(Y_test, Y_pred))