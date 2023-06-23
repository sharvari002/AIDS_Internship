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

df=pd.read_csv("C:/CDSAI/adult.csv")

X = df.drop('income', axis=1)
Y = df['income']
print(X)
print(Y)
'''
print(df)
print(df.head(10))
print(df.tail)
print(df.columns.values)
print(df.describe())




#boxplot
sns.boxplot(df['age'])
plt.show()

print(df['age'])
Q1 = df['age'].quantile(0.25)
Q3= df['age'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper = Q3 + 0.5*IQR
lower = Q1 + 0.5*IQR

print(upper)
print(lower)

out1 = df[df['age']< lower].values
out2 = df[df['age']< upper].values

df['age'].replace(out1,lower,inplace=True)
df['age'].replace(out2,upper,inplace=True)

print(df['age'])
'''
'''
#random forest
rf = RandomForestClassifier()
df['age']=pd.cut(df['age'],3,labels=['0','1','2'])
df['hours.per.week']=pd.cut(df['hours.per.week'],3,labels=['0','1','2'])
df['fnlwgt']=pd.cut(df['fnlwgt'],3,labels=['0','1','2'])
df['capital.loss']=pd.cut(df['capital.loss'],3,labels=['0','1','2'])

print(df)

X = df.drop('income', axis=1)
Y = df['income']
print(Y)

print(df.isnull().sum())
print(df.notnull().sum())
print(Counter(Y))
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X, Y)
print(Counter(Y))
'''
'''
logr = LogisticRegression()
pca = PCA(n_components=2)
X = df.drop('income', axis=1)
Y= df['income']
pca.fit(X)
X = pca.transform(X)
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=2, test_size=0.4)
logr.fit(X_train, Y_train)
Y_pred = logr.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
'''
'''
#counter

print(df.isnull().sum())
print(df.notnull().sum())

print(Counter(Y))
ros = RandomOverSampler(random_state=0)
X, Y =ros.fit_resample(X, Y)
print(Counter(Y))
'''
'''
#feature
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScore = pd.concat([dfcolumns, dfscores], axis=1)
featuresScore.columns=['feature','Score']

print(featuresScore)

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(4).plot(kind='pie')
plt.show()
'''
le = LabelEncoder()

le.fit(df['age'])
df['age'] = le.transform(df['age'])

le.fit(df['workclass'])
df['workclass'] = le.transform(df['workclass'])

le.fit(df['fnlwgt'])
df['fnlwgt'] = le.transform(df['fnlwgt'])

le.fit(df['capital.loss'])
df['capital.loss'] = le.transform(df['capital.loss'])

le.fit(df['hours.per.week'])
df['hours.per.week'] = le.transform(df['hours.per.week'])



