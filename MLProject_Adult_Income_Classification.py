#Objective: Prediction task is to determine whether a person makes over 50K a year.

'''
Adult Income DataSet Description

A). Number of Instances
   48842 instances, mix of continuous and discrete
=====================================================

B). Number of Attributes
   6 continuous, 8 nominal attributes.
====================================================

C). Attribute Information:

    1) age: continuous.
    2) workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    3) fnlwgt: continuous.
    4) education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    5) education-num: continuous.
    6) marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    7) occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    8) relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    9) race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    10)sex: Female, Male.
    11)capital-gain: continuous.
    12)capital-loss: continuous.
    13)hours-per-week: continuous.
    14)native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    15)class: >50K, <=50K

==========================================================
D. Missing Attribute Values:
   7% have missing values

'''

#Step-1 : Import the required libraries
import warnings
warnings.filterwarnings(action="ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


#Step-2: Read the dataset
data=pd.read_csv("adult_income_dataset.csv")


#Step-3: Exploratory Data Analytics


print("data.shape = ", data.shape)
print("\n\n")


print("Info about dataset\n", data.info() )
print("\n\n")


#print first 10 rows of dataset
print("First 10 rows : \n", data.head(10))
print("\n\n")


print("data.describe() = \n", data.describe())
print("\n\n")


print("data.describe(include='object') = \n", data.describe(include='object'))
print("\n\n")


print("data.income.unique() = \n", data.income.unique() )
print("\n\n")

## data.income=data.income.replace(['<=50K', '>50K'],[0,1])
sns.countplot(data=data, x="income")
plt.show()


#Feature Analysis and Vizualization
print("data.age.unique() = \n", data.age.unique())
print("\n\n")


#Vizualization-01: Income vs Age
#sns.boxplot(data.income, data.age)
sns.boxplot(x="income", y="age", data=data)
plt.show()


#Vizualization-02:  Income vs fnlwgt
#sns.boxplot(data.income,data['fnlwgt'])
sns.boxplot(x="income", y="fnlwgt", data=data)
plt.show()



#Vizualization-03:  Income vs educational-num
#sns.boxplot(data.income,data['educational-num'])
sns.boxplot(x="income", y="educational-num", data=data)
plt.show()



#Vizualization-04:  Income vs capital-gain
#sns.boxplot(data.income,data['capital-gain'])
sns.boxplot(x="income", y="capital-gain", data=data)
plt.show()



#Vizualization-05:  Income vs capital-loss
#sns.boxplot(data.income,data['capital-loss'])
sns.boxplot(x="income", y="capital-loss", data=data)
plt.show()



#Vizualization-06:  Income vs capital-loss
#sns.boxplot(data.income,data['capital-loss'])
sns.boxplot(x="income", y="capital-loss", data=data)
plt.show()



#Vizualization-07:  Income vs hours-per-week
#sns.boxplot(data.income,data['hours-per-week'])
sns.boxplot(x="income", y="hours-per-week", data=data)
plt.show()



#Print the details for missing values
print( data.isnull().sum()  )
print("\n\n")


#Remove the missing values if any
data=data.dropna()


#Replace the income class with numeric values
#a) Replace "<=50K" with  0
#b) Replace ">50K"  with  1
data.income=data.income.replace(['<=50K', '>50K'],[0,1])



#Vizualization-08: Print correlation matrix via heatmap
plt.figure(figsize=(7,6))
sns.heatmap(data.corr(),annot=True,cmap='Blues')
plt.show()



#print unique values in the column "workclass"
print("data.workclass.unique() = \n", data.workclass.unique() )
print("\n\n")



#Vizualization-09: workclass vs income
plt.figure(figsize=(15,7))
ax=sns.countplot(data=data, x="workclass", hue=data.income)
plt.legend(['<=50K', '>50K'])
plt.show()




#print unique values in the column "education"
print("data.education.unique() =\n",  data.education.unique() )
print("\n\n")


#Vizualization-10: education vs income
plt.figure(figsize=(15,7))
ax=sns.countplot(data=data, x="education", hue=data.income)
ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
plt.legend(['<=50K', '>50K'])
plt.show()



#print unique values in the column "marital-status"
print("data['marital-status'].unique() =\n",  data['marital-status'].unique() )
print("\n\n")

#Vizualization-11: marital-status vs income
plt.figure(figsize=(15,7))
ax=sns.countplot(data=data, x = data['marital-status'], hue=data.income)
plt.legend(['<=50K', '>50K'])
plt.show()




#print unique values in the column "occupation"
print("data.occupation.unique() =\n",  data.occupation.unique() )
print("\n\n")


#Vizualization-12: occupation vs income
plt.figure(figsize=(15,7))
ax=sns.countplot(data=data, x="occupation", hue=data.income)
ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
plt.legend(['<=50K', '>50K'])
plt.show()



#print unique values in the column "relationship"
print("data.relationship.unique() =\n",  data.relationship.unique() )
print("\n\n")


#Vizualization-13: relationship vs income
plt.figure(figsize=(15,7))
ax=sns.countplot(data=data, x="relationship", hue=data.income)
plt.legend(['<=50K', '>50K'])
plt.show()




#print unique values in the column "gender"
print("data.gender.unique() =\n",  data.gender.unique() )
print("\n\n")


#Vizualization-14: gender vs income
plt.figure(figsize=(15,7))
ax=sns.countplot(data=data, x="gender", hue=data.income)
plt.legend(['<=50K', '>50K'])
plt.show()




#print unique values in the column "native-country"
print("data['native-country'].unique() =\n",  data['native-country'].unique())
print("\n\n")


#Vizualization-15: native-country vs income
plt.figure(figsize=(15,7))
ax=sns.countplot(data=data, x='native-country', hue=data.income)
ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
plt.legend(['<=50K', '>50K'])
plt.show()



#Data Preprocessing
x=data.drop(['income'],axis=1)
y=data.income

print("x.head(10) = \n", x.head(10))
print("\n\n")


x=pd.get_dummies(x) #Categorical data will converted into OneHotEncoding format
print( "x.head(10) = \n", x.head(10))
print("\n\n")


#Remove the column "fnlwgt" because it do not have any significant differences
#data=data.drop(['fnlwgt'],axis=1)

#print the list of column names havin numerical values
numericalcols=list(data.select_dtypes(exclude='object').columns)
print( "numericalcols = ", numericalcols)
print("\n\n")


numericalcols.pop()  #To remove the last income column name from the list
print( "numericalcols = ", numericalcols)
print("\n\n")



#Standardized the values of  numerical columns
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x[numericalcols]=scaler.fit_transform(x[numericalcols])


print("After StandardScaler: x.head() = \n", x.head() )
print("\n\n")



#Data Splitting:
x_train,x_test,y_trian,y_test=train_test_split(x,y,random_state=7,test_size=0.3)



#Train the model-01: LogisticRegression Algoritmns

model=LogisticRegression()
model.fit(x_train,y_trian)

#First Method to print the accuracy of LogisticRegression Algorithm
accuracy_result =  model.score(x_test,y_test)
print("Accuracy of LogisticRegression : ",  accuracy_result )
print("\n\n")

#Second Method to print the accuracy of LogisticRegression Algorithm
y_predictL=model.predict(x_test)
accuracy_result = accuracy_score(y_test,y_predictL)
print("Accuracy of LogisticRegression : ",  accuracy_result )

#Vizualization-16: Performance of LogisticRegression
ax=sns.distplot(y_test,hist=False,label='Actual Values')
ax=sns.distplot(y_predictL,hist=False,label='Predicted Values')
ax.set_title('LogisticRegression')
plt.legend()
plt.show()

#print the confusion_matrix for prediction accuracy of LogicsticRegression
matrix = confusion_matrix(y_test,y_predictL)
print("confusion_matrix for prediction accuracy of LogicsticRegression = \n")
print(matrix)
print("\n\n")


#Vizualization-17: heatmap of confusion_matrix for LogicsticRegression
sns.heatmap(confusion_matrix(y_test,y_predictL), annot=True, cmap='Blues')
plt.show()



#Train the model-02: SupportVectorClassifier Algoritmns

model=SVC()
model.fit(x_train,y_trian)

#First Method to print the accuracy of SupportVectorClassifier Algorithm
accuracy_result =  model.score(x_test,y_test)
print("Accuracy of SupportVectorClassifier : ",  accuracy_result )
print("\n\n")

#Second Method to print the accuracy of SupportVectorClassifier Algorithm
y_predicS=model.predict(x_test)
accuracy_result = accuracy_score(y_test,y_predicS)
print("Accuracy of SupportVectorClassifier : ",  accuracy_result )
print("\n\n")

ax=sns.distplot(y_test,hist=False,label='Actual Values')
ax=sns.distplot(y_predicS,hist=False,label='Predicted Values')
ax.set_title('SupportVectorClassifier')
plt.legend()
plt.show()


#print the confusion_matrix for prediction accuracy of SupportVectorClassifier
matrix = confusion_matrix(y_test,y_predictL)
print("confusion_matrix for prediction accuracy of SupportVectorClassifier = \n")
print(matrix)
print("\n\n")
