#data processing, CSV
import pandas as pd 

#Visualization
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

#Machine learning
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

#Load Data
titanic_test = pd.read_csv('C:/Users/amendoza/Desktop/titanic/data/test.csv')
titanic_train = pd.read_csv('C:/Users/amendoza/Desktop/titanic/data/train.csv')

#Test Data
print(titanic_test)

#Train Data
print(titanic_train)

#Analysis
print(titanic_train.describe())

#Survived by Sex
titanic_train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)

titanic_train.groupby(['Sex','Survived']).count()['PassengerId'].unstack().plot.bar()

g = sns.FacetGrid(titanic_train,col='Survived')
g.map(plt.hist, 'Age', bins=20)

#Survived by class
titanic_train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False)

titanic_train.groupby(['Pclass','Survived']).count()['PassengerId'].unstack().plot.bar()

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
cols = ['blue', 'lightcoral']
titanic_train['Sex'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', shadow=True, colors=cols)
plt.title('Total Male/Female onboard')

plt.subplot(1,2,2)
sns.barplot(x='Sex', y='Survived', data=titanic_train, palette='plasma')
plt.title('Sex Vs Survived')
plt.ylabel('Survived Rate')
plt.show()

#Preprocessing
titanic_train.info()
titanic_train.isnull().sum()
sum(list(titanic_train.Survived))

titanic_train['Age'].describe()

#When Age is null set 0
age_mean = titanic_train['Age'].mean()
titanic_train['Age'] = titanic_train['Age'].fillna(age_mean).round(0).astype(int)

#When Embarked is null set empty object
embarked_mode = titanic_train['Embarked'].mode()[0]
titanic_train['Embarked'] = titanic_train['Embarked'].fillna(embarked_mode)

#Clean our data in titanic_train 
titanic_train = titanic_train.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1)
titanic_train

titanic_train.isnull().sum()

Sex = { 'male': 0, 'female': 1}
titanic_train['Sex'] = titanic_train['Sex'].map(Sex)

Embarked = {'S': 0, 'C': 1, 'Q': 2}
titanic_train['Embarked'] = titanic_train['Embarked'].map(Embarked)

#View data Types
titanic_train.dtypes

#Feature selection
x = titanic_train.drop(['Survived'], axis = 1)
y = titanic_train['Survived']

lasso = LassoCV()
lasso.fit(x, y)

print('Best Alpha:', lasso.alpha_)
print('Best Score:',lasso.score(x,y))

coef = pd.Series(lasso.coef_, index=x.columns)
print('Lasso picked:' + str(sum(coef != 0))+ ' variables and eliminated the orden ' + str(sum(coef == 0)) + ' variables')

imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind='barh')
plt.title('Feature Importance Using LASSO')

#Model training 
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.20)

#K-Nearest *Neighbors*
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {'n_neighbors':range(3,15),'weights':['uniform','distance'],'leaf_size':range(25,35)}

grid = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=param_grid, scoring='balanced_accuracy',cv=4,refit=True)

grid.fit(X_train,Y_train)

print(grid.best_score_)
print(grid.best_estimator_)

knc = KNeighborsClassifier(n_neighbors=9,leaf_size=25,weights='uniform')
knc.fit(X_train,Y_train)

Y_pred = knc.predict(X_test)
print(knc.score(X_train,Y_train))
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
