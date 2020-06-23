#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# 1.โหลด csv เข้าไปใน Python Pandas

# In[2]:


df = pd.read_csv('../Desktop/DataCamp/Prostate_Cancer.csv')
df


# 2. เขียนโค้ดแสดง หัว10แถว ท้าย10แถว และสุ่ม10แถว

# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.sample(10)


# 3. เช็คว่ามีข้อมูลที่หายไปไหม สามารถจัดการได้ตามความเหมาะสม

# In[6]:


df.isnull().sum()


# 4. ใช้ info และ describe อธิบายข้อมูลเบื้องต้น

# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.replace('B', 1, inplace = True)
df.replace('M', 0, inplace = True)


# In[10]:


df


# 5. ใช้ pairplot ดูความสัมพันธ์เบื้องต้นของ features ที่สนใจ

# In[11]:


sns.pairplot(data = df)


# In[12]:


sns.pairplot(data = df , vars=['radius', 'area','compactness','perimeter','diagnosis_result'])


# 6. ใช้ displot เพื่อดูการกระจายของแต่ละคอลัมน์

# In[13]:


sns.distplot(df['texture'], kde = False, bins = 15)


# In[14]:


sns.distplot(df['perimeter'], kde = False, bins = 15)


# In[15]:


sns.distplot(df['smoothness'], kde = False, bins = 15)


# 7. ใช้ heatmap ดูความสัมพันธ์ของคอลัมน์ที่สนใจ

# In[16]:


plt.figure(figsize = (12,8))
sns.heatmap(df.corr(), annot = df.corr())


# 8. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation สูงสุด

# In[17]:


sns.scatterplot(data = df, x = 'area', y = 'perimeter')


# 9. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation ต่ำสุด

# In[18]:


sns.scatterplot(data = df, x = 'perimeter', y = 'diagnosis_result')


# 10. สร้าง histogram ของ feature ที่สนใจ

# In[19]:


plt.hist(df['radius'])


# In[20]:


plt.hist(df['symmetry'])


# 11. สร้าง box plot ของ features ที่สนใจ

# In[21]:


sns.boxplot(data = df, x = 'diagnosis_result', y = 'radius')


# In[22]:


sns.boxplot(data = df, x = 'diagnosis_result', y = 'symmetry')


# In[23]:


sns.boxplot(data = df, x = 'diagnosis_result', y = 'compactness')


# In[24]:


sns.boxplot(data = df, x = 'diagnosis_result', y = 'area')


# 13. ทำ Data Visualization อื่นๆ (แล้วแต่เลือก)

# In[25]:


sns.countplot(data = df, x= 'diagnosis_result')


# 14. พิจารณาว่าควรทำ Normalization หรือ Standardization หรือไม่ควรทั้งสองอย่าง พร้อมให้เหตุผล 

# ควรทำ Normalization เพราะ x ไม่เป็น normal distribution

# # Standardization

# In[26]:


from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score,accuracy_score
from sklearn.svm import SVC


# In[27]:


X = df.drop('diagnosis_result', axis = 1)
y = df['diagnosis_result']


# In[28]:


sc_X = StandardScaler()

X1 = sc_X.fit_transform(X)


# In[29]:


X_train,X_test,y_train,y_test = train_test_split(X1,y, test_size = 0.2, random_state = 100)


# In[30]:


svc1 = SVC()
svc1.fit(X_train,y_train)


# In[31]:


predicted1 = svc1.predict(X_test)
predicted1


# In[32]:


confusion_matrix(predicted1, y_test)


# In[33]:


print('accuracy score',accuracy_score(y_test,predicted1))
print('precision score',precision_score(y_test,predicted1))
print('recall_score',recall_score(y_test,predicted1))
print('f1 score',f1_score(y_test,predicted1))


# # Normalization

# In[34]:


min_max_scaler = MinMaxScaler()


# In[35]:


X_minmax = min_max_scaler.fit_transform(df.drop('diagnosis_result', axis = 1))
X_minmax


# In[36]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax,y, test_size = 0.2, random_state = 100)


# In[37]:


svc2 = SVC()
svc2.fit(X_train,y_train)


# In[38]:


predicted2 = svc2.predict(X_test)
predicted2


# In[39]:


confusion_matrix(predicted2, y_test)


# In[40]:


print('accuracy score',accuracy_score(y_test,predicted2))
print('precision score',precision_score(y_test,predicted2))
print('recall_score',recall_score(y_test,predicted2))
print('f1 score',f1_score(y_test,predicted2))


# 15. เลือกช้อยที่ดีที่สุดจากข้อ 14 (หรือจะทำทุกอันแล้วนำมาเปรียบเทียบก็ได้)

# ผลของ  Standardization ดีกว่าผลของ Normalization

# 16. วัดผลโมเดล โดยใช้ confusion matrix และ ประเมินผลด้วยคะแนน Accuracy, 
# F1 score, Recall, Precision

# In[41]:


#Standardization
print('accuracy score',accuracy_score(y_test,predicted1))
print('precision score',precision_score(y_test,predicted1))
print('recall_score',recall_score(y_test,predicted1))
print('f1 score',f1_score(y_test,predicted1))


# In[42]:


#Normalization
print('accuracy score',accuracy_score(y_test,predicted2))
print('precision score',precision_score(y_test,predicted2))
print('recall_score',recall_score(y_test,predicted2))
print('f1 score',f1_score(y_test,predicted2))


# 17. หาค่า parameter combination ที่ดีที่สุด สำหรับ Dataset นี้ โดยใช้ GridSearch (Hyperparameter Tuning)

# In[43]:


param_combination = {'C':[0.01,0.1,1,10,100,1000,10000], 'gamma':[0.00001,0.0001,0.001,0.01,0.1,1,10]}


# In[44]:


param_combination


# In[45]:


grid_search = GridSearchCV(SVC(),param_combination, verbose = 1)
grid_search


# In[46]:


grid_search.fit(X_train,y_train)


# In[47]:


grid_search.best_params_


# In[48]:


grid_search.best_estimator_


# In[49]:


grid_predicted = grid_search.predict(X_test)
grid_predicted


# In[50]:


confusion_matrix(grid_predicted, y_test)


# In[51]:


print('accuracy score', accuracy_score(y_test,grid_predicted))    
print('precision score', precision_score(y_test,grid_predicted))
print('recall score', recall_score(y_test,grid_predicted))
print('f1 score', f1_score(y_test,grid_predicted))


# 18. เลือกเฉพาะ features ที่สนใจมาเทรนโมเดล และวัดผลเปรียบเทียบกับแบบ all-features

# In[52]:


df.corr()


# In[53]:


X = df[['perimeter','compactness']]
y = df['diagnosis_result']


# In[54]:


min_max_scaler2 = MinMaxScaler()


# In[55]:


X_minmax = min_max_scaler2.fit_transform(X)
X_minmax


# In[56]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax,y, test_size = 0.2, random_state = 100)


# In[57]:


svc3 = SVC()
svc3.fit(X_train,y_train)


# In[58]:


predicted3 = svc3.predict(X_test)
predicted3


# In[59]:


confusion_matrix(predicted3, y_test)


# In[60]:


#Normalization 2 features: perimeter,compactness
print('accuracy score',accuracy_score(y_test,predicted3))
print('precision score',precision_score(y_test,predicted3))
print('recall_score',recall_score(y_test,predicted3))
print('f1 score',f1_score(y_test,predicted3))


# In[61]:


#Normalization all features
print('accuracy score',accuracy_score(y_test,predicted2))
print('precision score',precision_score(y_test,predicted2))
print('recall_score',recall_score(y_test,predicted2))
print('f1 score',f1_score(y_test,predicted2))


# In[62]:


#GridSearch + Normalization 
print('accuracy score', accuracy_score(y_test,grid_predicted))    
print('precision score', precision_score(y_test,grid_predicted))
print('recall score', recall_score(y_test,grid_predicted))
print('f1 score', f1_score(y_test,grid_predicted))


# # Default

# In[63]:


X = df.drop('diagnosis_result', axis = 1)
y = df['diagnosis_result']


# In[64]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax,y, test_size = 0.2, random_state = 100)


# In[65]:


svc4 = SVC()
svc4.fit(X_train,y_train)


# In[66]:


predicted4 = svc3.predict(X_test)
predicted4


# In[67]:


confusion_matrix(predicted4, y_test)


# In[68]:


print('accuracy score',accuracy_score(y_test,predicted4))
print('precision score',precision_score(y_test,predicted4))
print('recall_score',recall_score(y_test,predicted4))
print('f1 score',f1_score(y_test,predicted4))


# 19. ทำ Visualization ของค่า F1 Score ระหว่าง ผลลัพธ์ที่ได้จากค่า Default, ผลลัพธ์ที่ได้จากการใช้ Grid Search และ ผลลัพธ์ของ Normalization

# In[69]:


data = {'Defult' : f1_score(y_test,predicted4) , 'Grid Search': f1_score(y_test,grid_predicted),
        'Normalization' : f1_score(y_test,predicted2)}
data


# In[70]:


Series = pd.Series(data = data)
Series


# In[71]:


Series.index


# In[72]:


Series.values


# In[73]:


df2 = pd.DataFrame(Series)
df2


# In[74]:


sns.barplot(data = df2, x = df2.index, y = df2[0])
plt.ylabel('f1 score')


# 20. ทำ Visualization ของค่า Recall ระหว่าง ผลลัพธ์ที่ได้จากค่า Default, ผลลัพธ์ที่ได้จากการใช้ Grid Search และ ผลลัพธ์ของ Normalization

# In[75]:


data = {'Defult' : recall_score(y_test,predicted4) , 'Grid Search': recall_score(y_test,grid_predicted),
        'Normalization' : recall_score(y_test,predicted2)}
data


# In[76]:


Series2 = pd.Series(data = data)
Series2


# In[77]:


df3 = pd.DataFrame(Series2)
df3


# In[78]:


sns.barplot(data = df3, x = df3.index, y = df3[0])
plt.ylabel('recall score')


# 21. ทำ Visualization ของค่า Accuracy ระหว่าง ผลลัพธ์ที่ได้จากค่า Default, ผลลัพธ์ที่ได้จากการใช้ Grid Search และ ผลลัพธ์ของ Normalization

# In[79]:


data = {'Defult' : accuracy_score(y_test,predicted4) , 'Grid Search': accuracy_score(y_test,grid_predicted),
        'Normalization' : accuracy_score(y_test,predicted2)}
data


# In[80]:


Series3 = pd.Series(data = data)
Series3


# In[81]:


df4 = pd.DataFrame(Series3)
df4


# In[82]:


sns.barplot(data = df4, x = df4.index, y = df4[0])
plt.ylabel('accuracy score')


# 22. สามารถใช้เทคนิคใดก็ได้ตามที่สอนมา ใช้ SVM Algorithm แล้วให้ผลลัพธ์ที่ดีที่สุดที่เป็นไปได้ (อาจจะรวม Grid Search กับ Normalization ?)

# # GridSearch + Standardization

# In[83]:


X = df.drop('diagnosis_result', axis = 1)
y = df['diagnosis_result']


# In[84]:


sc_X2 =  StandardScaler()
X2 = sc_X2.fit_transform(X)


# In[85]:


X_train,X_test,y_train,y_test = train_test_split(X2,y,test_size =0.3, random_state = 20)


# In[86]:


param_combination = {'C':[0.01,0.1,1,10,100,1000,10000], 'gamma':[0.00001,0.0001,0.001,0.01,0.1,1,10]}


# In[87]:


grid_search = GridSearchCV(SVC(),param_combination, verbose = 1)
grid_search


# In[88]:


grid_search.fit(X_train,y_train)


# In[89]:


grid_search.best_params_


# In[90]:


grid_search.best_estimator_


# In[91]:


grid_predicted2 = grid_search.predict(X_test)
grid_predicted2


# In[92]:


confusion_matrix(grid_predicted2, y_test)


# In[93]:


print('accuracy score', accuracy_score(y_test,grid_predicted2))    
print('precision score', precision_score(y_test,grid_predicted2))
print('recall score', recall_score(y_test,grid_predicted2))
print('f1 score', f1_score(y_test,grid_predicted2))


# # GridSearch

# In[94]:


X = df.drop('diagnosis_result', axis = 1)
y = df['diagnosis_result']


# In[95]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3, random_state = 20)


# In[96]:


grid_search = GridSearchCV(SVC(),param_combination, verbose = 1)
grid_search


# In[97]:


grid_search.fit(X_train,y_train)


# In[98]:


grid_search.best_params_


# In[99]:


grid_search.best_estimator_


# In[100]:


grid_predicted3 = grid_search.predict(X_test)
grid_predicted3


# In[101]:


confusion_matrix(grid_predicted3, y_test)


# In[102]:


print('accuracy score', accuracy_score(y_test,grid_predicted3))    
print('precision score', precision_score(y_test,grid_predicted3))
print('recall score', recall_score(y_test,grid_predicted3))
print('f1 score', f1_score(y_test,grid_predicted3))


# # Standardization is the best accuracy score = 0.9
