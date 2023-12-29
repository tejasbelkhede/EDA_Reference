#!/usr/bin/env python
# coding: utf-8

# In[98]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
import warnings
warnings.filterwarnings('ignore')



# In[99]:


df=pd.read_csv("titanic-training-data.csv")


# In[100]:


df.head()


# In[101]:


df.describe()


# In[102]:


df.isnull().sum()


# In[103]:


df.shape


# In[104]:


df.dtypes


# In[105]:


median1=df["Age"].median()


# In[136]:


median1


# In[107]:


df["Age"]=df["Age"].replace(np.nan,median1)


# In[108]:


df.isnull().sum()


# In[109]:


mode1=df["Embarked"].mode()[0]
mode1


# In[110]:


mode1


# In[111]:


df["Embarked"]=df["Embarked"].replace(np.nan,mode1)


# In[112]:


df.isnull().sum()


# # Drop insignificant columns

# In[113]:


df=df.drop(["Cabin","PassengerId","Name","Ticket"],axis=1)


# In[114]:


df.isnull().sum()


# In[115]:


duplicate=df.duplicated()
print(duplicate.sum())


# In[116]:


sns.boxplot(x="Age",data=df)


# In[117]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3=(1.5 * IQR)
    return lower_range, upper_range
    


# In[118]:


lowAge,uppAge=remove_outlier(df['Age'])
df['age']=np.where(df['Age']>uppAge,uppAge,df['Age'])
df['age']=np.where(df['Age']<lowAge,lowAge,df['Age'])



# In[119]:


sns.boxplot(x="Age",data=df)


# In[120]:


sns.boxplot(x="Age",data=df)


# In[121]:


###incodind


# In[122]:


df.dtypes


# In[123]:


df.head()


# In[124]:


df=pd.get_dummies(df,columns=["Sex","Embarked"])


# In[125]:


df.head()


# In[126]:


###dataset 2 (pima dibetes datasey)


# In[127]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
import warnings
warnings.filterwarnings('ignore')



# In[128]:


df=pd.read_csv("pima-indians-diabetes-2.csv")


# In[129]:


df.head()


# In[130]:


df.describe()


# In[131]:


df.isnull().sum()


# In[132]:


df.shape


# In[133]:


df.dtypes


# In[135]:


median1=df["age"].median()


# In[137]:


median1


# In[138]:


df["age"]=df["age"].replace(np.nan,median1)


# In[139]:


df.isnull().sum()


# In[140]:


mode1=df["test"].mode()[0]
mode1


# In[142]:


df=df.drop(["mass","class","test","skin"],axis=1)


# In[143]:


df.isnull().sum()


# In[144]:


duplicate=df.duplicated()
print(duplicate.sum())


# In[146]:


sns.boxplot(x="age",data=df)


# In[ ]:




