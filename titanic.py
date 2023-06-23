import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


df=pd.read_csv("C:/CDSAI/tested.csv")

'''
#iris
irs = load_iris()
print(irs)
print(irs.keys())
print(irs.data)
print(irs.target)
print(irs.feature_names)
print(irs.target_names)
print(irs.DESCR)

print(df)
print(df.head(10))
print(df.tail)
print(df.columns.values)
print(df.describe())




#boxplot
sns.boxplot(df['PassengerId'])
plt.show()

print(df['PassengerId'])
Q1 = df['PassengerId'].quantile(0.25)
Q3= df['PassengerId'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper = Q3 + 0.5*IQR
lower = Q1 + 0.5*IQR

print(upper)
print(lower)

out1 = df[df['PassengerId']< lower].values
out2 = df[df['PassengerId']< upper].values

df['PassengerId'].replace(out1,lower,inplace=True)
df['PassengerId'].replace(out2,upper,inplace=True)

print(df['PassengerId'])




#counter
print(df.isnull().sum())
print(df.notnull().sum())

print(Counter(Y))
ros = RandomOverSampler(random_state=0)
X, Y =ros.fit_resample(X, Y)
print(Counter(Y))




#random forest
rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=0)
nb=MultinomialNB()
dt=DecisionTreeClassifier(random_state=0)
gtm=GradientBoostingClassifier(n_estimators=10)

rf = RandomForestClassifier()

df['Pclass']=pd.cut(df['Pclass'],3, labels=['0' , '1', '2'])
df['Parch']=pd.cut(df['Parch'],3, labels=['0' , '1' , '2'] )
df['Survived']=pd.cut(df['Survived'],3, labels=['0' , '1' , '2'] )
df['PassengerId']=pd.cut(df['PassengerId'],3, labels=['0' , '1' , '2'] )

print(df)
X= df.drop('Embarked',axis=1)
Y= df['Embarked']
print(Y)
le=LabelEncoder()

X= df.drop('Embarked',axis=1)
Y= df['Embarked']
print(X)
print(Y)
'''



#feature
X = df.drop('Survived', axis=1)
Y = df['Survived']
print(X)
print(Y)


bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs','Score']

print(featureScores)

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)







