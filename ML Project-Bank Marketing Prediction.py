#!/usr/bin/env python
# coding: utf-8

# Task To Perform

# Read in the file and get basic information about the data, including numerical summaries.
# 
# Describe the pdays column, make note of the mean, median and minimum values. Anything fishy in the values?
# 
# Describe the pdays column again, this time limiting yourself to the relevant values of pdays. How different are the mean and the median values?
# 
# Plot a horizontal bar graph with the median values of balance for each education level value. Which group has the highest median?
# 
# Make a box plot for pdays. Do you see any outliers?

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


bank_mark=pd.read_csv('bank-marketing.csv')


# # Initial Details about the dataset to get the basic insights

# In[3]:


bank_mark.head()


# In[4]:


bank_mark.shape


# In[5]:


bank_mark.describe()


# In[6]:


bank_mark.info()


# Describe the pdays column
# 
# make note of the mean, median and minimum values. Anything fishy in the values?

# In[7]:


bank_mark['pdays'].describe()


# "pdays" is using -1 as indicator and not value. Hence treat these are missing value.
# 
# 
# Ignore these values in our average/median/state calculations.Keeping it NaN makes more sense so Wherever pdays is -1, replacing the same with NaN.

# In[8]:


df=bank_mark.copy()


# In[9]:


df.drop(bank_mark[bank_mark['pdays'] <0 ] .index, inplace=True)


# In[10]:


df['pdays'].describe()


# # Ploting a bar graph with the median values of balance for each education level value. Which group has the highest median?

# Removing objects and numerical data types to represent the graph accurately.

# In[11]:


df.groupby(['education'])['balance'].median().plot.barh()


# # Tertiary level of education having median value for balance as the plot implies
# 

# Make a box plot for pdays. Do you see any outliers?

# df.pdays.plot.box()
# plt.show()

# Clear Indication in the boxplot which confirms the presence of outliers( data point that differs significantly from other observations.).

# # The final goal is to make a predictive model to predict if the customer will respond positively to the campaign or not. The target variable is “response”.

# First, perform bi-variate analysis to identify the features that are directly associated with the target variable. You can refer to the notebook we used for the EDA discussion.
# 
# 
# _Convert the response variable to a convenient form
# _Make suitable plots for associations with numerical features and categorical features’
# 
# Are the features about the previous campaign data useful?
# 
# Are pdays and poutcome associated with the target?
# 
# If yes, and if you plan to use them – how do you handle the pdays column with a value of -1 where the previous campaign data is missing? Explain your approach and your decision.

# In[13]:


df['response'].value_counts()


# In[14]:


df.response.value_counts(normalize=True)


# In[16]:


df.replace({'response': {"yes":1, "no":0}}, inplace=True)


# In[17]:


df['response'].value_counts()


# Taking away objects and numerical datatypes

# In[18]:


obj_col = []
num_col = []
for col in df.columns:
    if df[col].dtype=='O':
        obj_col.append(col)
    else:
        num_col.append(col)


# In[19]:


print("Features of Object ",obj_col)
print(" Features of Numerical ",num_col)


# In[20]:


## violin plots is used as it gives proper distribution of data shows the data is Neg/Postive skewed or not at all.


from numpy import median
for col in obj_col[1:]:
    plt.figure(figsize=(8,6))
    sns.violinplot(df[col],df["response"])
    plt.title("Response vs "+col,fontsize=15)
    plt.xlabel(col,fontsize=10)
    plt.ylabel("Response",fontsize=10)
    plt.show()


# In[21]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,cmap='Greens')
plt.title("Correlation betweet all Numerical Feature")
plt.show()


# To summarize "duration" variable is correlated with response variable. Whereas "pdays" variable is not highly correlated with response variable.

# In[22]:


for col in num_col[:-1]:
    plt.figure(figsize=(10,8))
    sns.jointplot(x = df[col],y = df["response"],kind='reg',joint_kws={'color':'green'})
    plt.xlabel(col,fontsize = 12)
    plt.ylabel("Response",fontsize = 12)
    plt.grid()
    plt.show()


# In[23]:


from sklearn.preprocessing import LabelEncoder


# In[24]:


df2 = df[obj_col].apply(LabelEncoder().fit_transform)


# In[25]:


df2.head()


# In[27]:


df3 = df2.join(df[num_col])


# In[28]:


df3.corr()


# # Logistic Regression

# In[29]:


X = df3.drop("response", axis=1)
X.head()


# In[30]:


y= df3[['response']]
y.head()


# In[31]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
from sklearn.metrics import classification_report,confusion_matrix,f1_score
np.random.seed(42)
from sklearn.model_selection import cross_val_score
lr.fit(X_train,y_train)


# In[33]:


cv_score= cross_val_score(lr,X_train,y_train, cv=5)
np.mean(cv_score)


# In[34]:


y_pred = lr.predict(X_test)


# In[35]:


print(classification_report(y_test, y_pred))


# In[36]:


confusion_matrix(y_pred,y_test)


# In[37]:


f1_score(y_pred,y_test)


# In[38]:


confusion_matrix(y_pred,y_test)


#        Performing Recursive Feature Elimination

# In[39]:


from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[40]:


rfe = RFE(lr, 5)
rfe.fit(X_train,y_train)


# In[41]:


rfe.support_


# In[42]:


X_train.columns[rfe.support_]


# In[43]:


cols = X_train.columns[rfe.support_]


# In[44]:


lr.fit(X_train[cols],y_train)


# In[45]:


y_pred2 = lr.predict(X_test[cols])


# In[46]:


f1_score(y_pred2,y_test)


# In[47]:


confusion_matrix(y_pred2,y_test)


# Using Stats model

# In[48]:


import statsmodels.api as StatMod


# In[49]:


X_train.head()


# In[50]:


X_train_StatMod = StatMod.add_constant(X_train[cols])
X_train_StatMod.head()


# In[51]:


Logreg = StatMod.OLS(y_train, X_train_StatMod).fit()


# In[52]:


Logreg.summary()


# Variance inflation factor

# In[53]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[54]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Housing, loan, default, poutcome are important features from logistic regression model
# 

# Random Forest

# In[55]:


from sklearn.ensemble import RandomForestClassifier


# In[56]:


rfc = RandomForestClassifier(max_depth=5, random_state=42,max_leaf_nodes=50)


# In[57]:


rfc.fit(X_train,y_train)


# In[58]:


score= cross_val_score(rfc,X_train,y_train, cv=5)
np.mean(score)


# In[59]:


y_predict = rfc.predict(X_test)


# In[60]:


print(classification_report(y_test, y_predict))


# In[61]:


f1_score(y_test,y_predict)


# In[62]:


confusion_matrix(y_test,y_predict)


# In[63]:


from sklearn.metrics import roc_auc_score


# In[64]:


roc_auc_score(y_test,y_predict)


# Recursive Feature Elimination

# In[65]:


from sklearn.feature_selection import RFE


# In[66]:


rfe1 = RFE(rfc, 5)
rfe1.fit(X_train,y_train)


# In[67]:


rfe1.support_


# In[68]:


X_train.columns[rfe1.support_]


# In[69]:


cols = X_train.columns[rfe1.support_]


# In[70]:


rfc.fit(X_train[cols],y_train)


# In[71]:


y_pred3 = rfc.predict(X_test[cols])


# In[72]:


f1_score(y_pred3,y_test)


# In[74]:


confusion_matrix(y_pred3,y_test)


# # Housing, month, pdays, poutcome, duration are important features from Random forest

# In[ ]:




