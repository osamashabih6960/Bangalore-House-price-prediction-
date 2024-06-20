#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


data = pd.read_csv('Bengaluru_House_Data.csv')


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info


# In[7]:


for column in data.columns:
    print(data[column].value_counts())
    print("*"*20)


# In[8]:


data.isna().sum()


# In[9]:


data.drop(columns=['area_type','availability','society','balcony'],inplace = True)


# In[10]:


data.describe()


# In[11]:


data.info


# In[12]:


data['location'].value_counts()


# In[14]:


data['location'] = data['location'].fillna('Sarjapur Road')


# In[15]:


data['size'].value_counts()


# In[16]:


data['size'] = data['size'].fillna('2 BHK')


# In[17]:


data['bath'] = data['bath'].fillna(data['bath'].median())


# In[18]:


data.info()


# In[19]:


data['bhk'] = data['size'].str.split().str.get(0).astype(int)


# In[20]:


data[data.bhk > 20]


# In[21]:


data['total_sqft'].unique()


# In[23]:


def convertRange(x):
    temp = x.split('-')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
            return None


# In[24]:


data['total_sqft'] = data['total_sqft'].apply(convertRange)


# In[25]:


data.head()


# # price Per square feet

# In[26]:


data['price_per_sqft'] = data['price']*100000/data['total_sqft']


# In[27]:


data['price_per_sqft']


# In[28]:


data.describe()


# In[29]:


data['location'].value_counts()


# In[36]:


location_counts_less_10 = location_counts[location_counts <= 10]
location_counts_less_10


# In[37]:


data['location'] = data['location'].apply(lambda x:x.strip())
location_counts = data['location'].value_counts()


# In[38]:


location_counts


# In[39]:


data['location']=data['location'].apply(lambda x: 'other' if x in location_counts_less_10 else x)


# # outlier detection and removal

# In[40]:


data.describe()


# In[42]:


(data['total_sqft']/data['bhk']).describe()


# In[43]:


data = data[((data['total_sqft']/data['bhk']) >= 300)]


# In[44]:


data.describe()


# In[46]:


data.shape


# In[47]:


data.price_per_sqft.describe()


# In[48]:


def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output,gen_df], ignore_index = True)
        
    return df_output
data = remove_outliers_sqft(data)
data.describe()


# In[52]:


def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean':np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
            return df.drop(exclude_indices, axis = 'index')
                


# In[55]:


data=bhk_outlier_remover(data)


# In[56]:


data.shape


# In[57]:


data


# In[58]:


data.drop(columns=['size','price_per_sqft'],inplace=True)


# # clean Data

# In[59]:


data.head()


# In[60]:


data.to_csv("cleaned_data.csv")


# In[61]:


x = data.drop(columns=['price'])
y = data['price']


# In[ ]:


# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[67]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state=0)


# In[68]:


print(x_train.shape)
print(y_train.shape)


# # APPLY LINEAR REGRESSION

# In[71]:


column_trans = make_column_transformer((OneHotEncoder(sparse = False),['location']),remainder = 'passthrough')


# In[72]:


scaler = StandardScaler()


# In[78]:


lr = LinearRegression(normalize=True)


# In[80]:


pipe = make_pipeline(column_trans,scaler)


# In[81]:


pipe.fit(x_train,y_train)


# # apply lasso

# In[86]:


lasso = Lasso()


# In[87]:


pipe = make_pipeline(column_trans,scaler,lasso)


# In[88]:


pipe.fit(x_train,y_train)


# In[89]:


y_pred_lasso = pipe.predict(x_test)
r2_score(y_test,y_pred_lasso)


# # APPLY RIDGE

# In[105]:


ridge = Ridge


# In[106]:


pipe = make_pipeline(column_trans,scaler,ridge)


# In[107]:


import pickle


# In[108]:


pickle.dump(pipe, open('RidgeModel.pkl', 'wb'))


# In[ ]:





# In[ ]:




