

###########################################
############## Importing Packages #########
###########################################


import json
import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import  ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import catboost as ctb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import  recall_score
from sklearn.metrics import  precision_score
from sklearn.utils import resample



##############################################################
######################## Importing Dataset ##################
##############################################################


cust=pd.read_csv("E:/ADM/customer.csv")
cust=cust.drop(["Unnamed: 0"],axis=1)


#Droping the variables which is not important for output variable.
cust=cust.drop(["customerDevice","customerEmail","customerIPAddress","customerPhone","customerBillingAddress","orderId","orderShippingAddress","paymentMethodId","transactionId"],axis=1)
cust.columns

#converting into binary
lb=LabelEncoder()
cust["fraudulent"]=lb.fit_transform(cust["fraudulent"])
cust["orderState"]=lb.fit_transform(cust["orderState"])
cust["transactionFailed"]=lb.fit_transform(cust["transactionFailed"])
cust["paymentMethodIssuer"]=lb.fit_transform(cust["paymentMethodIssuer"])
cust["paymentMethodProvider"]=lb.fit_transform(cust["paymentMethodProvider"])
cust["paymentMethodRegistrationFailure"]=lb.fit_transform(cust["paymentMethodRegistrationFailure"])
cust["paymentMethodType"]=lb.fit_transform(cust["paymentMethodType"])


########################################################
########################### EDA ########################
########################################################


#Identify duplicates records in the data
dupes=cust.duplicated()
sum(dupes)

#Removing Duplicates
cust1=cust.drop_duplicates() ######

cust1['fraudulent'].unique()
cust1.isnull().sum()
cust1.columns
cust1.shape
describe= cust1.describe() #mean,Std. dev, range
cust1.describe().T
cust1.median()
cust1.var()
max(cust1.orderAmount)-min(cust1.orderAmount)#range
cust1.skew()

#Finding outliers with index
def detect_outliers(x):
   q1 = np.percentile(x,25)
   q3 = np.percentile(x,75)
   iqr = q3 - q1
   lower = q1-(1.5*iqr)
   upper = q3+(1.5*iqr)
   outlier_indices = list(x.index[(x<lower) | (x>upper)])
   outlier_value = list(x[outlier_indices])
   
   return outlier_indices, outlier_value
indics, values =  detect_outliers(cust1["orderAmount"])
print(indics)
print(values)
indics, values =  detect_outliers(cust1["transactionAmount"])
print(indics)
print(values)

#histograms for each variable in df
hist =cust1.hist(bins=20,figsize =(14,14))

sns.countplot(data = cust1, x = 'fraudulent')
sns.countplot(data = cust1, x = 'paymentMethodIssuer')
sns.countplot(data = cust1, x = 'paymentMethodProvider')
sns.countplot(data = cust1, x = 'paymentMethodRegistrationFailure')
sns.countplot(data = cust1, x = 'paymentMethodType')

#create a boxplot for every column in df
boxplot = cust1.boxplot(grid=True, vert=True,fontsize=13)

#create the correlation matrix heat map
plt.figure(figsize=(14,12))
sns.heatmap(cust1.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0)

#pair plots
g = sns.pairplot(cust1)

sb.factorplot(x='fraudulent' ,col='orderState' ,kind='count', data=cust1);
sb.factorplot(x='fraudulent' ,col='transactionFailed' ,kind='count', data=cust1);
sb.factorplot(x='fraudulent' ,col='paymentMethodProvider' ,kind='count', data=cust1);
sb.factorplot(x='fraudulent' ,col='paymentMethodRegistrationFailure' ,kind='count', data=cust1);
sb.factorplot(x='fraudulent' ,col='paymentMethodType' ,kind='count', data=cust1);

sns.lmplot(data= cust1, x='paymentMethodRegistrationFailure', y='paymentMethodType')

# Density Plot and Histogram of all arrival delays
sns.distplot(cust1['orderAmount'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4}) 

sns.barplot(x='transactionAmount',y='transactionFailed',data=cust1)

################## Standardization ##############

def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

df_norm = norm_func(cust1.iloc[:,1:])
df_norm.describe()

### New Dataset

cust2= pd.concat([df_norm, cust1.iloc[:,0]], axis = 1)



#########################################
############## Balancing ############## : oversampling performing better than smote and undersampling
#########################################


# Separate input features and target
x = cust2.iloc[:,:8]
y = cust2.iloc[:,8]

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=53)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
not_fraud = X[X.fraudulent==0]
fraud = X[X.fraudulent==1]

# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=53) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

result = upsampled.reset_index() 

######### Create New Dataframe 
r2=result.drop(["index"],axis=1)

# check new class counts
r2.fraudulent.value_counts()






###################################################################
########################## Feature Selection ######################
###################################################################


#Feature Selection using Tree Classifier
a = r2.iloc[:,:8]  #independent columns
b = r2.iloc[:,-1]    #target column

model = ExtraTreeClassifier()
model.fit(a,b)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=a.columns)
feat_importances.nlargest(8).plot(kind='barh')

#Almost all 8 variables are contributing towards output variable.


###############################################################
####################### Cross Validation ######################
###############################################################


colnames = list(r2.columns)
predictors = colnames[:8]
target = colnames[8]

Xx = cust2[predictors]
Yy = cust2[target]

#Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=300, random_state=24)
all_accuracies = cross_val_score(classifier,Xx,Yy,cv=10)
print(all_accuracies)
print(all_accuracies.mean()) #70.28

#Catboost
modell = ctb.CatBoostClassifier()
all_accuraciess = cross_val_score(modell,Xx,Yy,cv=10)
print(all_accuraciess)
print(all_accuraciess.mean())#71.47


#Adaboost Classifier
abc = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
all_accuraciesss = cross_val_score(abc,Xx,Yy,cv=10)
print(all_accuraciesss)
print(all_accuraciesss.mean())#75.75

#XGBClassifier
model = XGBClassifier()
all_accuracy = cross_val_score(model,Xx,Yy,cv=10)
print(all_accuracy)
print(all_accuracy.mean()) #71.13

#Adaboost classifier is giving highest accuracy so we will build final model on this algorithm.


#############################################################################
############################## Final Model  Bulding #########################
#############################################################################


#Adaboost classifier


X = r2[predictors]
Y = r2[target]

#Train Test split
train,test = train_test_split(r2,test_size = 0.4,stratify=r2.fraudulent,random_state=53)

###### Model building ######
rf =AdaBoostClassifier(n_estimators=50,learning_rate=0.1)

rf.fit(train[predictors],train[target])

rf.estimators_ # 
rf.classes_ # class labels (output)
rf.predict(X)


pred_train=rf.predict(train[predictors])
pred_test = rf.predict(test[predictors])
pd.Series(pred_test).value_counts()
pd.crosstab(test[target],pred_test)
pd.crosstab(train[target],pred_train)

#f1 Score 
f1_score(test[target], pred_test)#96.70

#recall
recall_score(test[target], pred_test, average='weighted') #96.51

#precision
precision_score(test[target], pred_test, average='weighted')#96.88

# Accuracy = train
np.mean(train.fraudulent == rf.predict(train[predictors]))#77.64

# Accuracy = Test
np.mean(pred_test==test.fraudulent) # 74.39

#just Good model.