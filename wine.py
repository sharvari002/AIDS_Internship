from typing import Any

import pandas as pd
from pandas import DataFrame
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
from sklearn.datasets import load_wine

df = pd.read_csv("C:\CDSAI\winequalityN.csv")
'''
print(df)
print(df.head(10))
print(df.tail)
print(df.columns.values)
print(df.describe())

'''

'''

X = df.drop('type', axis=1)
y = df['type']
print(y)


le=LabelEncoder()

le.fit(df['type'])
df['type'] = le.transform(df['type'])

bestfeatures = SelectKBest(k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScore = pd.concat([dfcolumns, dfscores], axis=1)
featuresScore.columns = ['features', 'Score']

print(featuresScore)

model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(4).plot(kind='pie')
plt.show()

'''
'''
rf = RandomForestClassifier()
df['fixed acidity']=pd.cut(df['fixed acidity'],3,labels=['0','1','2'])
df['volatile acidity']=pd.cut(df['volatile acidity'],3,labels=['0','1','2'])
df['citric acid']=pd.cut(df['citric acid'],3,labels=['0','1','2'])
df['sulphates']=pd.cut(df['sulphates'],3,labels=['0','1','2'])

print(df)

X = df.drop('type', axis=1)
Y = df['type']
print(Y)

print(df.isnull().sum())
print(df.notnull().sum())

print(Counter(Y))
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X, Y)
print(Counter(Y))


X = df.drop('type', axis=1)
Y = df['type']
print(X)
print(Y)

logr = LogisticRegression()
pca = PCA(n_components=2)
X = df.drop('quality', axis=1)
Y = df['quality']
pca.fit(X)
X = pca.transform(X)
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.4)
logr.fit(X_train, Y_train)
Y_pred = logr.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
'''
'''
sns.boxplot(df['quality'])
plt.show()

print(df['quality'])
Q1 = df['quality'].quantile(0.25)
Q3= df['quality'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper = Q3 + 0.5*IQR
lower = Q1 + 0.5*IQR

print(upper)
print(lower)

out1 = df[df['quality']< lower].values
out2 = df[df['quality']< upper].values

df['quality'].replace(out1,lower,inplace=True)
df['quality'].replace(out2,upper,inplace=True)

print(df['quality'])

'''
'''
X = df.drop('type', axis=1)
Y = df['type']
print(X)
print(Y)
le=LabelEncoder()

le.fit(df['density'])
df['density'] = le.transform(df['density'])


bestfeatures = SelectKBest(k=5)
fit = bestfeatures.fit(X, Y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featurescores = pd.concat([dfcolumns,dfscores],axis=1)

pd.set_option('display.width',100)
data.head(1)
print(data)

scatter_matrix(data)
pyplot.show()

print(featurescores.nlargest('2,score'))
'''
X = df.drop('type', axis=1)
Y = df['type']
print(X)
print(Y)


le = LabelEncoder()

le.fit(df['fixed acidity'])
df['fixed acidity'] = le.transform(df['fixed acidity'])

le.fit(df['volatile acidity'])
df['volatile acidity'] = le.transform(df['volatile acidity'])

le.fit(df['sulphates'])
df['sulphates'] = le.transform(df['sulphates'])

le.fit(df['citric acid'])
df['citric acid'] = le.transform(df['citric acid'])

le.fit(df['chlorides'])
df['chlorides'] = le.transform(df['chlorides'])

