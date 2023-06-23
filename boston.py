import pandas as pd
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


df=pd.read_csv("C:\CDSAI\HousingData.csv")

X = df.drop('CRIM', axis=1)
Y = df['CRIM']
print(X)
print(Y)

'''
print(df)
print(df.head(10))
print(df.tail)
print(df.columns.values)
print(df.describe())

'''

'''

#random forest
rf = RandomForestClassifier()
df['ZN']=pd.cut(df['ZN'],3,labels=['0','1','2'])
df['RM']=pd.cut(df['RM'],3,labels=['0','1','2'])
df['DIS']=pd.cut(df['DIS'],3,labels=['0','1','2'])
df['NOX']=pd.cut(df['NOX'],3,labels=['0','1','2'])

print(df)



#counter
print(df.isnull().sum())
print(df.notnull().sum())
print(Counter(Y))
ros = RandomOverSampler(random_state=0)
X, Y= ros.fit_resample(X, Y)
print(Counter(Y))



#boxplot
sns.boxplot(df['TAX'])
plt.show()

print(df['TAX'])
Q1 = df['TAX'].quantile(0.25)
Q3= df['TAX'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper = Q3 + 0.5*IQR
lower = Q1 + 0.5*IQR

print(upper)
print(lower)

out1 = df[df['TAX']< lower].values
out2 = df[df['TAX']< upper].values

df['TAX'].replace(out1,lower,inplace=True)
df['TAX'].replace(out2,upper,inplace=True)

print(df['TAX'])
'''
'''
#feature
X = df.drop('CRIM', axis=1)
Y = df['CRIM']
print(X)
print(Y)


bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature','Score']

print(featureScores)

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

'''

le = LabelEncoder()

le.fit(df['ZN'])
df['ZN'] = le.transform(df['ZN'])

le.fit(df['CRIM'])
df['CRIM'] = le.transform(df['CRIM'])

le.fit(df['INDUS'])
df['INDUS'] = le.transform(df['INDUS'])

le.fit(df['B'])
df['B'] = le.transform(df['B'])

le.fit(df['LSTAT'])
df['LSTAT'] = le.transform(df['LSTAT'])

