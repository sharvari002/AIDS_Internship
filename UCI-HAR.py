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


df=pd.read_csv("C:/CDSAI/UCI.csv")

X = df.drop('link', axis=1)
X = X.drop('year', axis=1)
Y = df['year']
print(X)
print(Y)

le=LabelEncoder()

le.fit(df['link'])
df['link']=le.transform(df['link'])

le.fit(df['Data-Name'])
df['Data-Name']=le.transform(df['Data-Name'])

le.fit(df['data type'])
df['data type']=le.transform(df['data type'])

le.fit(df['default task'])
df['default task']=le.transform(df['default task'])

le.fit(df['attribute-type'])
df['attribute-type']=le.transform(df['attribute-type'])

le.fit(df['instances'])
df['instances']=le.transform(df['instances'])

le.fit(df['attributes'])
df['attributes']=le.transform(df['attributes'])

#feature
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScore = pd.concat([dfcolumns, dfscores], axis=1)
featuresScore.columns=['Link','Year']

print(featuresScore)

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(4).plot(kind='pie')
plt.show()



#random forest
rf = RandomForestClassifier()
df['link']=pd.cut(df['link'],3,labels=['0','1','2'])
df['attribute-type']=pd.cut(df['attribute-type'],3,labels=['0','1','2'])
df['instances']=pd.cut(df['instances'],3,labels=['0','1','2'])
df['attributes']=pd.cut(df['attributes'],3,labels=['0','1','2'])

print(df)


X = df.drop('link', axis=1)
X = X.drop('year', axis=1)
Y = df['year']
print(X)
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

X = df.drop('link', axis=1)
X = X.drop('year', axis=1)
Y = df['year']
pca.fit(X)
X = pca.transform(X)
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=2, test_size=0.4)
logr.fit(X_train, Y_train)
Y_pred = logr.predict(X_test)
print(accuracy_score(Y_test, Y_pred))




#Boxplot
sns.boxplot(df['link'])
plt.show()
print(df['link'])
Q1 = df['link'].quantile(0.25)
Q3 = df['link'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 + 1.5*IQR
print(upper)
print(lower)
out1 = df['link'] < lower.values
out1 = df['link'] < upper.values
df['link'].replace(out1, lower, inplace = True)
df['link'].replace(out2, upper, inplace = True)
print(df['link'])




#counter

X = df.drop('link', axis=1)
X = X.drop('year', axis=1)
Y = df['year']
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

X = df.drop('link', axis=1)
X = X.drop('year', axis=1)
Y = df['year']

pca.fit(X)
X = pca.transform(X)
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=2, test_size=0.4)
logr.fit(X_train, Y_train)
Y_pred = logr.predict(X_test)
print(accuracy_score(Y_test, Y_pred))