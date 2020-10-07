#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler


# In[2]:


le = preprocessing.LabelEncoder()
def objColHandler(df):
    objectCols=[]
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):         
            objectCols.append(col)
            df[col]= le.fit_transform(df[col])     
    return objectCols,df


def nullHandler(df,thold):
    fullNullCols=[]
    nullCols=[]
    for col in df.columns:
        ratio=df[col].isna().sum()/len(df[col])
        if ratio>0:
            if ratio>thold:
                fullNullCols.append(col)
                df = df.drop(col, 1)
            else:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col].fillna(df[col].mean(),inplace=True)
                    nullCols.append(col)
                else:
                    df[col].fillna(df[col].value_counts().idxmax(),inplace=True)    
    return fullNullCols,nullCols,df


# In[3]:


train = pd.read_csv("hw08_training_data.csv",index_col = 0)
trainLabel = pd.read_csv("hw08_training_label.csv",index_col = 0)
test = pd.read_csv("hw08_test_data.csv",index_col = 0)
y=trainLabel
dfT=train
dfT = pd.concat([dfT, test], axis=0)


# In[4]:


"""print(train.shape)
print(train.dtypes.unique())
train.isna().sum().unique()
train.dtypes

wD.isna().sum().unique()

print(wD.shape)
wD.columns
y.isna().sum().unique()
y.iloc[0:2,0:3]"""


# In[5]:


fNC,nC,dfT=nullHandler(dfT,0.3)
oC,dfT=objColHandler(dfT)
dfT.shape


# In[6]:


dfT.shape


# In[7]:


dfT=StandardScaler().fit_transform(dfT)
pca=PCA()
dfTPCA=pca.fit_transform(dfT)
dfTPCA=pd.DataFrame(dfTPCA)


# In[8]:


#print(pca.components_)
#print(pca.explained_variance_)
#print(pca.explained_variance_ratio_.cumsum())
#print(np.cumsum(pca.get_covariance())
k=0
for i in pca.explained_variance_ratio_.cumsum():
    if pca.explained_variance_ratio_.cumsum()[k]<0.99:
        k=k+1
print(k)
pca.explained_variance_ratio_.cumsum()
dfTPCARed=dfTPCA.iloc[:,0:k]
print(dfTPCARed.shape)


# In[9]:


dTest=dfTPCARed.iloc[len(train):len(dfTPCARed),]
dfTPCARed=dfTPCARed.iloc[0:len(train),]


# In[10]:


wD1 = pd.concat([dfTPCARed, y.iloc[:,0:1]], axis=1)
wD2 = pd.concat([dfTPCARed, y.iloc[:,1:2]], axis=1)
wD3 = pd.concat([dfTPCARed, y.iloc[:,2:3]], axis=1)
wD4 = pd.concat([dfTPCARed, y.iloc[:,3:4]], axis=1)
wD5 = pd.concat([dfTPCARed, y.iloc[:,4:5]], axis=1)
wD6 = pd.concat([dfTPCARed, y.iloc[:,5:6]], axis=1)



# In[11]:


dTest=pd.concat([dfTPCARed, y.iloc[:,0:1]], axis=1)


# In[12]:


dTest=dTest.drop(columns=['TARGET_1'])
dTest.columns


# In[ ]:





# In[13]:


wD1N=wD1[pd.notnull(wD1['TARGET_1'])]
x=wD1N.iloc[:,0:k]
y1=wD1N.iloc[:,k:(k+1)]


# In[14]:


x.columns


# In[15]:


wD2N=wD2[pd.notnull(wD2['TARGET_2'])]
x2=wD2N.iloc[:,0:k]
y2=wD2N.iloc[:,k:(k+1)]


# In[16]:


wD3N=wD3[pd.notnull(wD3['TARGET_3'])]
x3=wD3N.iloc[:,0:k]
y3=wD3N.iloc[:,k:(k+1)]


# In[17]:


wD4N=wD4[pd.notnull(wD4['TARGET_4'])]
x4=wD4N.iloc[:,0:k]
y4=wD4N.iloc[:,k:(k+1)]


# In[18]:


wD5N=wD5[pd.notnull(wD5['TARGET_5'])]
x5=wD5N.iloc[:,0:k]
y5=wD5N.iloc[:,k:(k+1)]


# In[19]:


wD6N=wD6[pd.notnull(wD6['TARGET_6'])]
x6=wD6N.iloc[:,0:k]
y6=wD6N.iloc[:,k:(k+1)]


# In[ ]:





# In[20]:


trainX, valX, trainY, valY = train_test_split(x, y1, test_size=0.2,random_state=56)
trainX2, valX2, trainY2, valY2 = train_test_split(x2, y2, test_size=0.2,random_state=56)
trainX3, valX3, trainY3, valY3 = train_test_split(x3, y3, test_size=0.2,random_state=56)
trainX4, valX4, trainY4, valY4 = train_test_split(x4, y4, test_size=0.2,random_state=56)
trainX5, valX5, trainY5, valY5 = train_test_split(x5, y5, test_size=0.2,random_state=56)
trainX6, valX6, trainY6, valY6 = train_test_split(x6, y6, test_size=0.2,random_state=56)


# In[21]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(trainX, trainY)
valYpred=clf.predict(valX)
roc_auc_score(valY, valYpred)


# In[22]:


clfXG1=xgb.XGBClassifier(random_state=0,learning_rate=0.85)
clfXG1.fit(trainX,trainY)
valYpred = clfXG1.predict(valX)
roc_auc_score(valY, valYpred)


# In[23]:


clfXG2=xgb.XGBClassifier(random_state=0,learning_rate=0.85)
clfXG2.fit(trainX2,trainY2)
valYpred2 = clfXG2.predict(valX2)
roc_auc_score(valY2, valYpred2)


# In[24]:


clfXG3=xgb.XGBClassifier(random_state=0,learning_rate=0.9)
clfXG3.fit(trainX3,trainY3)
valYpred3 = clfXG3.predict(valX3)
roc_auc_score(valY3, valYpred3)


# In[25]:


clfXG4=xgb.XGBClassifier(random_state=0,learning_rate=0.9)
clfXG4.fit(trainX4,trainY4)
valYpred4 = clfXG4.predict(valX4)
roc_auc_score(valY4, valYpred4)


# In[26]:


clfXG5=xgb.XGBClassifier(random_state=0,learning_rate=0.85)
clfXG5.fit(trainX5,trainY5)
valYpred5 = clfXG5.predict(valX5)
roc_auc_score(valY5, valYpred5)


# In[27]:


clfXG6=xgb.XGBClassifier(random_state=0,learning_rate=0.9)
clfXG6.fit(trainX6,trainY6)
valYpred6 = clfXG6.predict(valX6)
roc_auc_score(valY6, valYpred6)


# In[28]:


testPred1 = pd.DataFrame(clfXG1.predict_proba(dTest))
testPred1 = testPred1.iloc[:,1]

testPred2 = pd.DataFrame(clfXG2.predict_proba(dTest))
testPred2 = testPred2.iloc[:,1]

testPred3 = pd.DataFrame(clfXG3.predict_proba(dTest))
testPred3 = testPred3.iloc[:,1]

testPred4 = pd.DataFrame(clfXG4.predict_proba(dTest))
testPred4 = testPred4.iloc[:,1]

testPred5 = pd.DataFrame(clfXG5.predict_proba(dTest))
testPred5 = testPred5.iloc[:,1]

testPred6 = pd.DataFrame(clfXG6.predict_proba(dTest))
testPred6 = testPred6.iloc[:,1]

"""testPred1=clfXG1.predict(dTest)
testPred2=clfXG2.predict(dTest)
testPred3=clfXG3.predict(dTest)
testPred4=clfXG4.predict(dTest)
testPred5=clfXG5.predict(dTest)
testPred6=clfXG6.predict(dTest)
"""
predictions = [testPred1,testPred2,testPred3,testPred4,testPred5,testPred6]


# In[ ]:





# In[29]:





# In[39]:


predictions = [testPred1,testPred2,testPred3,testPred4,testPred5,testPred6]
predictionsD=pd.DataFrame(predictions)
predictionsD=predictionsD.T
predictionsD.to_csv("hw08_test_predictions.csv")

