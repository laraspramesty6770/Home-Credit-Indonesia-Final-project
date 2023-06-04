#!/usr/bin/env python
# coding: utf-8

# ###### By Laras Puji Pramesty (4 Juni 2023)

# # Load Dataset

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import gc

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from scipy.stats import norm
from sklearn import tree
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.utils import resample #re-sampling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #normalize features
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.naive_bayes import GaussianNB #gaussian naive bayes
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier #decision tree
from sklearn.ensemble import RandomForestClassifier #random forest
from sklearn.neighbors import KNeighborsClassifier #k-nearest neighbor
from sklearn.neural_network import MLPClassifier #neural network
from sklearn.metrics import confusion_matrix, classification_report 


# In[2]:


#Setting Output
pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


test=pd.read_csv("../Final Project Rakamin/application_test.csv")
credit_card_balance=pd.read_csv("../Final Project Rakamin/credit_card_balance.csv")
installments_payments=pd.read_csv("../Final Project Rakamin/installments_payments.csv")
bureau=pd.read_csv("../Final Project Rakamin/bureau.csv")
bureau_balance=pd.read_csv("../Final Project Rakamin/bureau_balance.csv")
pos_cash=pd.read_csv("../Final Project Rakamin/POS_CASH_balance.csv")
prev_applications=pd.read_csv("../Final Project Rakamin/previous_application.csv")


# In[4]:


print("Application test:",test.shape)
print("Credit card balance:",credit_card_balance.shape)
print("Installements Payements:",installments_payments.shape)
print("Bureau:",bureau.shape)
print("Bureau balance:",bureau_balance.shape)
print("POS_cash Balance:",pos_cash.shape)
print("Previous Applications:",prev_applications.shape)


# In[5]:


#Data Application Train
df_train = pd.read_csv("../Final Project Rakamin/application_train.csv", sep=',')
print('This dataset has %d rows dan %d columns.\n' % df_train.shape)
df_train.head()


# In[21]:


df_train.columns.values


# ## Distribution of Target variable
# From the description 0 means that loan is repayed and 1 means loan is not repayed.
# As you can see that the percentage of y=1 is very less and the data set is skewed. So the best metric to test the performance is ROC

# In[6]:


sns.countplot(data =df_train,x='TARGET')
plt.title("Distribution of target variable ")
plt.show()


# In[23]:


df_train.describe()


# # Check Dataset

# In[7]:


#Check Data Types
print('Tipe Data: \n')
df_train.info(verbose=True)


# In[8]:


# Check Data Shape
df_train.shape


# # Data Preprocessing
# 

# In[28]:


def missing_data(data):
    na=pd.DataFrame()
    na['number']=data.isnull().sum().sort_values(ascending=False)
    na['Percent']=data.isnull().sum()/data.shape[0]*100
    na.drop(index=na.loc[na['number']==0].index,inplace=True)
    return na


# In[29]:


print(missing_data(df_train).shape[0])
missing_data(df_train).head(10)


# In[85]:


print(missing_data(test).shape[0])
missing_data(test).head(10)


# ## EXPLORATION THE DATASET

# ### Binary Feature
# NAME_CONTRACT_TYPE: Identification if loan is cash or revolving
# 
# FLAG_OWN_REALTY: Flag if client owns a house or flat

# In[9]:


f =plt.figure(figsize=(14,6))
ax= f.add_subplot(221)
sns.countplot(data = df_train, x='NAME_CONTRACT_TYPE')
ax=f.add_subplot(222)
sns.countplot(df_train, x='CODE_GENDER')
ax=f.add_subplot(223)
sns.countplot(df_train, x='FLAG_OWN_CAR')
ax=f.add_subplot(224)
sns.countplot(df_train, x='FLAG_OWN_REALTY')
plt.tight_layout()


# In[31]:


# convert to categorical type
df_train[['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']] =   df_train[
    ['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']].astype('object')
test[['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']] =   test[
    ['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']].astype('object')


# In[ ]:





# # Visualization

# In[10]:


round(df_train['AMT_CREDIT'].describe())


# In[11]:


sns.histplot(df_train['AMT_CREDIT'])


# In[12]:


round(df_train['AMT_INCOME_TOTAL'].describe())


# In[13]:


sns.distplot(df_train['AMT_INCOME_TOTAL'])


# In[14]:


f,ax=plt.subplots(figsize=(8,6))
fig=sns.boxplot(x=df_train['TARGET'],y=df_train['AMT_INCOME_TOTAL'])


# In[15]:


f,ax=plt.subplots(figsize=(8,6))
fig=sns.boxplot(x=df_train['TARGET'],y=df_train['CNT_CHILDREN'])


# In[38]:


correlation=df_train.corr()
f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(correlation,vmax=.8,square=True)


# In[16]:


corr = df_train.corr().abs()

sort = corr.unstack()
so = sort.sort_values(kind="quicksort")
print(so.head())



# In[40]:


df_train=df_train.drop(['WEEKDAY_APPR_PROCESS_START','FLAG_EMP_PHONE','REG_CITY_NOT_WORK_CITY','REGION_RATING_CLIENT','REG_REGION_NOT_WORK_REGION'],axis=1)


# In[17]:


sns.distplot(df_train['AMT_INCOME_TOTAL'], kde=True)
fig =plt.figure()
prob =stats.probplot(df_train['AMT_INCOME_TOTAL'],plot=plt)


# In[41]:


sns.distplot(df_train['AMT_INCOME_TOTAL'], fit = norm)
fig =plt.figure()
prob =stats.probplot(df_train['AMT_INCOME_TOTAL'],plot=plt)


# In[18]:


sns.distplot(df_train['AMT_CREDIT'],fit=norm)
fig=plt.figure()
res=stats.probplot(df_train['AMT_CREDIT'],plot=plt)


# In[43]:


x=df_train[['DAYS_EMPLOYED','AMT_CREDIT','AMT_INCOME_TOTAL','CNT_CHILDREN','REGION_POPULATION_RELATIVE']]
y=df_train['TARGET']


# In[44]:


clf=tree.DecisionTreeClassifier(min_samples_split=20)
clf=clf.fit(x,y)
y_pred=clf.predict(x)
print (pd.crosstab(y_pred,df_train['TARGET']))


# In[180]:


print(metrics.classification_report(y_pred,df_train['TARGET']))


# In[45]:


print("0: Membayar Tepat Waktu/Tidak ada kesulitan pembayaran")
print("1: Memiliki Kesulitan Pembayaran")

class_dist = df_train['TARGET'].value_counts()

plt.figure(figsize=(12,3))
plt.title('Class Distribution')
plt.barh(class_dist.index, class_dist.values)
plt.yticks([0, 1])

for i, value in enumerate(class_dist.values):
    plt.text(value-2000, i, str(value), fontsize=12, color='white',
             horizontalalignment='right', verticalalignment='center')

plt.show()


# In[46]:


def convert_age(age_days_negative):
    age_days_positive = -age_days_negative
    age_years = age_days_positive/365
    return age_years


# In[47]:


df_train['DAYS_BIRTH'] = df_train['DAYS_BIRTH'].apply(convert_age)
df_train['DAYS_EMPLOYED'] = df_train['DAYS_EMPLOYED'].apply(convert_age)


# In[48]:


plt.figure(figsize=(10,8))
plt.title('Age Distribution')
plt.xlabel('Age')
sns.kdeplot(df_train[df_train['TARGET']==1]['DAYS_BIRTH'], label='Target=1')
sns.kdeplot(df_train[df_train['TARGET']==0]['DAYS_BIRTH'], label='Target=0')
plt.grid()
plt.show()


# In[19]:


train1 = df_train.copy()


# In[20]:


decode_map = {0: "Tidak ada Kesulitan Pembayaran", 1: "Kesulitan Pembayaran"}
def decode_sentiment(label):
    return decode_map[int(label)]

train1['TARGET'] = train1['TARGET'].apply(lambda x: decode_sentiment(x))


# In[21]:


target_grp = (train1[['TARGET']]
                .groupby("TARGET")
                .agg(COUNT=("TARGET","count"))
                .sort_values(by=["COUNT"],ascending=False)
                .reset_index()
                )

target_grp.style.background_gradient(cmap='Blues')


# In[22]:


target = train1['TARGET'].value_counts(normalize=True)
target.reset_index().style.background_gradient(cmap='Oranges')


# In[23]:


figure = plt.figure(figsize = (12,7))
target.plot(kind='bar', color= ['green','Red'], alpha = 0.9, rot=0)
plt.title('Distribusi Kemampuan Pembayaran Customer\n', fontsize=14)
plt.show()


# Berdasarkan Diagram Batang diatas terdapat 91% pinjaman atau setara dengan sekitar 282 Ribu dengan TARGET = 0, dimana menunjukkan bahwa klien Tidak memiliki kesulitan dalam pembayaran. Sedangkan terdapat 8% dari total Pinjaman atau sekitar 24 Ribu dalam dataset ini dimana klien yang memiliki kesulitan pembayaran bermasalah dalam mengembalikan pinjaman

# # Bivariate : Categorical Features With Target
# Visualization of The Relationship Betweenn 2 Features
# Based On Contract Type, Gender, Car Ownership State and Realty Ownership State

# #### Contract Type, Gender, Car Ownership Status, and Realty Ownership Status

# In[24]:


# Visualization 1
sns.set_style('darkgrid')
fig, ax = plt.subplots(2,2, figsize=(25,25))
sns.set_context('paper', font_scale=1)

ax[0][0].set_title('Kemampuan Pembayaran berdasarkan Tipe Kontrak\n', fontweight='bold', fontsize=14)
sns.countplot(x='NAME_CONTRACT_TYPE', data = train1, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r', ax=ax[0][0])

ax[0][1].set_title('Kemampuan Pembayaran berdasarkan Jenis Kelamin\n', fontweight='bold', fontsize=14)
sns.countplot(x='CODE_GENDER', data = train1, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r', ax=ax[0][1])

ax[1][0].set_title('\n Kemampuan Pembayaran berdasarkan Kepemilikan Mobil\n', fontweight='bold', fontsize=14)
sns.countplot(x='FLAG_OWN_CAR', data = train1, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r', ax=ax[1][0])

ax[1][1].set_title('\n Kemampuan Pembayaran berdasarkan Status Kepemilikian Realty\n', fontweight='bold', fontsize=14)
sns.countplot(x='FLAG_OWN_REALTY', data = train1, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r', ax=ax[1][1])


# ## Contract Type With Target

# In[25]:


Contract_Type = train1.groupby(by=['NAME_CONTRACT_TYPE','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran klien berdasarkan Contract_Type')
Contract_Type.sort_values(by='SK_ID_CURR', ascending=False)
Contract_Type.style.background_gradient(cmap='Blues')


# Pinjaman tunai dengan sekitar 278 ribu pinjaman merupakan mayoritas dari total pinjaman dalam kumpulan data ini. Pinjaman bergulir memiliki jumlah yang jauh lebih rendah sekitar 29 ribu dibandingkan dengan pinjaman tunai.

# # Gender With Target

# In[26]:


train1['CODE_GENDER'] = train1['CODE_GENDER'].replace(['F','M'],['Female','Male'])


# In[27]:


gender = train1.groupby(by=['CODE_GENDER','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran Klien Berdasarkan Jenis Kelamin (gender)')
gender.sort_values(by='SK_ID_CURR', ascending=False)
gender.style.background_gradient(cmap='Blues')


# In[28]:


df_train[['CODE_GENDER','TARGET']].groupby(['CODE_GENDER'],as_index=False).mean().sort_values(by=['TARGET'], ascending=False)


# Dapat dilihat bahwa perempuan telah mengajukan sebagian besar pinjaman. Secara total, ada sekitar 202.448 aplikasi pinjaman yang diajukan oleh perempuan, dan sekitar 105.059 aplikasi diajukan oleh laki-laki.
# Namun, persentase yang lebih besar (sekitar 10% dari total) laki-laki memiliki masalah dalam membayar pinjaman dibandingkan dengan nasabah perempuan (sekitar 7%).

# # Car Ownership with Target

# In[29]:


train1['FLAG_OWN_CAR'] = train1['FLAG_OWN_CAR'].replace(['Y','N'],['Yes','No'])


# In[30]:


ownership_type = train1.groupby(by=['FLAG_OWN_CAR','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran Klien Berdasarkan Kepemilikan Mobil')
ownership_type.sort_values(by='SK_ID_CURR', ascending=False)
ownership_type.style.background_gradient(cmap='YlOrRd')


# In[31]:


df_train[['FLAG_OWN_CAR','TARGET']].groupby(['FLAG_OWN_CAR'],as_index=False).mean().sort_values(by=['TARGET'], ascending=False)


# Sebagian besar klien tidak memiliki mobil.
# Nasabah yang memiliki mobil (sekitar 8%) mengalami kesulitan dalam pengembalian pinjaman dibandingkan dengan nasabah yang tidak memiliki mobil (sekitar 7%). Namun, perbedaannya tidak terlalu signifikan.

# # Realty Ownership Status VS Target

# In[32]:


train1['FLAG_OWN_REALTY'] = train1['FLAG_OWN_REALTY'].replace(['Y','N'],['Yes','No'])


# In[33]:


ros = train1.groupby(by=['FLAG_OWN_REALTY','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran Klien Berdasarkan Kepemilikan Realti')
ros.sort_values(by='SK_ID_CURR', ascending=False)
ros.style.background_gradient(cmap='RdPu')


# In[34]:


df_train[['FLAG_OWN_REALTY','TARGET']].groupby(['FLAG_OWN_REALTY'],as_index=False).mean().sort_values(by=['TARGET'], ascending=False)


# Sebagian besar klien memiliki flat / rumah.
# Nasabah yang memiliki rumah/flat (sekitar 8%) mengalami kesulitan dalam pengembalian pinjaman dibandingkan nasabah yang tidak memiliki rumah/flat (sekitar 7%). Namun, perbedaannya tidak terlalu signifikan.

# # Analysis Based On Suite Type, Income Type, Education Type, and Family Status

# In[35]:


# visualization pt. 2
sns.set_style('whitegrid')
fig, ax = plt.subplots(2,2, figsize=(25,35))
sns.set_context('paper', font_scale=1)

ax[0][0].set_title('Clients Repayment Abilities By Suite Type\n', fontweight='bold', fontsize=14)
sns.countplot(x='NAME_TYPE_SUITE', data = train1, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r', ax=ax[0][0])

ax[0][1].set_title('Clients Repayment Abilities By Income Type\n', fontweight='bold', fontsize=14)
sns.countplot(x='NAME_INCOME_TYPE', data = train1, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r', ax=ax[0][1])

ax[1][0].set_title('\nClients Repayment Abilities By Education Type)\n', fontweight='bold', fontsize=14)
sns.countplot(x='NAME_EDUCATION_TYPE', data = train1, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r', ax=ax[1][0])

ax[1][1].set_title('\nClients Repayment Abilities By Family Status\n', fontweight='bold', fontsize=14)
sns.countplot(x='NAME_FAMILY_STATUS', data = train1, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r', ax=ax[1][1])


# ## Type Suite With Target

# In[36]:


st = train1.groupby(by=['NAME_TYPE_SUITE','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran Klien Berdasarkan Tipe Suite')
st.sort_values(by='SK_ID_CURR', ascending=False)
st.style.background_gradient(cmap='Blues')


# In[37]:


df_train[['NAME_TYPE_SUITE','TARGET']].groupby(['NAME_TYPE_SUITE'],as_index=False).mean().sort_values(by=['TARGET'], ascending=False)


# Klien yang didampingi oleh Other_B saat mengajukan pinjaman memiliki persentase kesulitan pengembalian pinjaman yang lebih tinggi (sekitar 10%).

# # Income Type With Target

# In[38]:


income_type = train1.groupby(by=['NAME_INCOME_TYPE','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran Klien Berdasarkan Tipe Pendapatan')
income_type.sort_values(by='SK_ID_CURR', ascending=False)
income_type.style.background_gradient(cmap='Purples')


# In[39]:


df_train[['NAME_INCOME_TYPE','TARGET']].groupby(['NAME_INCOME_TYPE'],as_index=False).mean().sort_values(by=['TARGET'], ascending=False)


# Dapat dilihat bahwa klien yang memiliki penghasilan bekerja telah mengajukan sebagian besar pinjaman, ada sekitar 158.774 aplikasi pinjaman.
# Nasabah dengan jenis penghasilan pengusaha dan mahasiswa tidak mengalami kesulitan dalam mengembalikan pinjamannya.
# Sedangkan klien dengan jenis penghasilan cuti hamil dan menganggur memiliki persentase tertinggi (sekitar 40% dan 36%) dengan TARGET = 1 yaitu. mengalami kendala dalam pengembalian pinjaman.

# ## Type Education with Target

# In[40]:


education_type = train1.groupby(by=['NAME_EDUCATION_TYPE','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran Klien Berdasarkan Jenis Pendidikan')
education_type.sort_values(by='SK_ID_CURR')
education_type.style.background_gradient(cmap='Reds')


# In[41]:


df_train[['NAME_EDUCATION_TYPE','TARGET']].groupby(['NAME_EDUCATION_TYPE'],as_index=False).mean().sort_values(by=['TARGET'], ascending=False)


# Sejumlah besar aplikasi (218K) diajukan oleh klien yang memiliki pendidikan menengah diikuti oleh orang dengan pendidikan tinggi dengan aplikasi 75K.
# Sedangkan klien dengan jenis pendidikan SMP memiliki persentase tertinggi (sekitar 10%) dari TARGET = 1 yaitu. mengalami Kesulitan dalam Pembayaran

# # Family Status With Target

# In[42]:


family = train1.groupby(by=['NAME_FAMILY_STATUS','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran Klien Berdasarkan Status Keluarga')
family.sort_values(by='SK_ID_CURR', ascending=False)
family.style.background_gradient(cmap='Purples')


# In[43]:


df_train[['NAME_FAMILY_STATUS','TARGET']].groupby(['NAME_FAMILY_STATUS'],as_index=False).mean().sort_values(by=['TARGET'], ascending=False)


# Dapat dilihat bahwa klien yang sudah menikah mengajukan sebagian besar pinjaman, ada sekitar 196.432 aplikasi pinjaman.
# Klien dengan status keluarga perkawinan sipil dan lajang memiliki persentase tertinggi (sekitar 9%) dari klien yang bermasalah dalam mengembalikan pinjaman.

# # Housing Type With Target

# In[44]:


housing = train1.groupby(by=['NAME_HOUSING_TYPE','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran Klien Berdasarkan Tipe Perumahan ')
housing.sort_values(by='SK_ID_CURR', ascending=False)
housing.style.background_gradient(cmap='Greens')


# In[45]:


df_train[['NAME_HOUSING_TYPE','TARGET']].groupby(['NAME_HOUSING_TYPE'],as_index=False).mean().sort_values(by=['TARGET'], ascending=False)


# Terlihat bahwa nasabah dengan tipe rumah apartemen sewa memiliki persentase kesulitan tertinggi (sekitar 12%) dalam kesulitan dalam melunasi pinjamannya.

# In[46]:


plt.figure(figsize=(15,8))
fig = sns.countplot(x='NAME_HOUSING_TYPE', data = train1, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r')
plt.title('Kemampuan Pembayaran Klien Berdasarkan Tipe Perumahan\n', fontweight='bold', fontsize=14)


# Klien yang tinggal di House/Apartment memiliki jumlah tertinggi pada aplikasi pinjaman sekitar  250ribu keatas.

# ## Occupation Type With Target

# In[47]:


occ = train1.groupby(by=['OCCUPATION_TYPE','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran Klien Berdasarkan Tipe Pekerjaan')
occ.sort_values(by='SK_ID_CURR', ascending=False)
occ.style.background_gradient(cmap='Blues')


# In[48]:


df_train[['OCCUPATION_TYPE','TARGET']].groupby(['OCCUPATION_TYPE'],as_index=False).mean().sort_values(by=['TARGET'], ascending=False)


# Sejumlah besar aplikasi (55K) diajukan oleh klien yang bekerja sebagai Buruh.
# Terlihat bahwa klien dengan jenis pekerjaan Low-skill Laborers memiliki persentase tertinggi (sekitar 17%) dengan TARGET = 1 yaitu. mengalami kendala dalam pengembalian pinjaman.

# ## Process Day With Target

# In[49]:


process_day = train1.groupby(by=['WEEKDAY_APPR_PROCESS_START','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran Klien Berdasarkan Proses Hari')
process_day.sort_values(by='SK_ID_CURR', ascending=False)
process_day.style.background_gradient(cmap='Blues')


# In[50]:


df_train[['WEEKDAY_APPR_PROCESS_START','TARGET']].groupby(['WEEKDAY_APPR_PROCESS_START'],as_index=False).mean().sort_values(by=['TARGET'], ascending=False)


# In[51]:


plt.figure(figsize=(15,8))
fig = sns.countplot(x='WEEKDAY_APPR_PROCESS_START', data = train1, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r')
plt.title('Kemampuan Pembayaran Klien Berdasarkan Proses hari\n', fontweight='bold', fontsize=14)
plt.xlabel('\nProcess Day', fontsize=12)


# Terlihat bahwa setiap hari pengerjaan aplikasi memiliki persentase yang hampir sama (sekitar 7% - 8%) pada TARGET = 1 yaitu yang memiliki kesulitan pembyaran atau mengalami kendala dalam pengembalian pinjaman.

# ## Organization Type With Target

# In[52]:


ot = train1.groupby(by=['ORGANIZATION_TYPE','TARGET'], as_index=False)['SK_ID_CURR'].count()
print('Kemampuan Pembayaran Klien berdasarkan Suite Type')
ot.sort_values(by='SK_ID_CURR', ascending=False)
ot.style.background_gradient(cmap='Blues')


# In[53]:


df_train[['ORGANIZATION_TYPE','TARGET']].groupby(['ORGANIZATION_TYPE'],as_index=False).mean().sort_values(by=['TARGET'], ascending=False)


# Dapat terlihat bahwa klien dengan tipe organisasi transportasi: type 3 memiliki persentase tertinggi (sekitar 15%) dari TARGET = 1 yaitu. mengalami kendala dalam pengembalian pinjaman.

# # Bivariate : Numerical Features Vs Target

# ## AMT CREDIT WITH TARGET

# In[54]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x='TARGET', y='AMT_CREDIT', hue="TARGET",data=train1, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x='TARGET', y='AMT_CREDIT', hue="TARGET",data=train1, palette="PRGn",showfliers=False)
plt.show();


# Terlihat bahwa nilai median jumlah kredit klien yang tidak mengalami kesulitan pembayaran sedikit lebih besar dibandingkan dengan nilai median nasabah yang mengalami kesulitan pembayaran. Artinya, nasabah dengan jumlah kredit yang lebih tinggi memiliki peluang yang sedikit lebih tinggi untuk mampu membayar kembali pinjamannya dibandingkan dengan nasabah dengan jumlah kredit yang lebih rendah.

# # AMT INCOME WITH TARGET

# In[55]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x='TARGET', y='AMT_INCOME_TOTAL', hue="TARGET",data=train1, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x='TARGET', y='AMT_INCOME_TOTAL', hue="TARGET",data=train1, palette="PRGn",showfliers=False)
plt.show();


# Terlihat bahwa baik klien yang mengalami kesulitan pembayaran maupun klien yang tidak mengalami kesulitan pembayaran, sebagian besar memiliki nilai yang sama. Namun, dengan meningkatnya pendapatan klien, kemungkinan bahwa klien tidak akan mengalami kesulitan membayar kembali pinjaman juga meningkat.

# # Age With Target

# In[56]:


N_age = df_train[df_train['TARGET']==0]['DAYS_BIRTH'].values/-365
Y_age = df_train[df_train['TARGET']==1]['DAYS_BIRTH'].values/-365


# In[57]:


plt.figure(figsize=(10,3))
plt.hist(df_train['DAYS_BIRTH'].values/-365, bins=10, edgecolor='black', color='Pink')
plt.title('Usia Klien  (Dalam tahun) Pada Saat Permohonanan\n')
plt.xlabel('Age Bucket')
plt.ylabel('Number of Clients')
plt.show()

plt.figure(figsize=(10,3))
plt.hist(N_age, bins=10, edgecolor='black', color='green')
plt.title('Usia Klien (Dalam Tahun) yang Memiliki Kesulitan Pembayaran\n')
plt.xlabel('Age')
plt.ylabel('Number of Clients')
plt.show()

plt.figure(figsize=(10,3))
plt.hist(Y_age, bins=10, edgecolor='black', color='orange')
plt.title('Usia Klien (Dalam Tahun) yang Tidak Memiliki Kesulitan Pembayaran\n')
plt.xlabel('Age')
plt.ylabel('Number of Clients')
plt.show()


# Sebagian besar nasabah yang mengajukan pinjaman berada pada rentang usia 35-40 tahun, diikuti nasabah pada rentang usia 40-45 tahun. Sementara itu, jumlah pelamar untuk klien berusia <25 tahun atau usia >65 tahun sangat rendah.
# Klien yang tidak mengalami kesulitan pembayaran adalah klien dengan rentang usia 35-45 tahun. Sedangkan nasabah yang mengalami kesulitan pembayaran adalah nasabah dengan rentang usia 25-35 tahun.

# # MULTIVARIAT VISUALIZATION

# ### A. Car Ownership Status, The Number of Children, Target, and House/Flat Ownership Status

# In[58]:


sns.catplot(x = 'FLAG_OWN_CAR',
            y = 'CNT_CHILDREN', 
            hue = 'TARGET', 
            col = 'FLAG_OWN_REALTY', 
            kind = 'bar', palette = 'ch:start=0.2,rot=-.3_r',
            data = train1)
plt.show()


# Klien yang memiliki mobil dan rumah/flat memiliki masalah dalam mengembalikan pinjaman untuk jumlah anak yang tinggi dibandingkan dengan klien yang tidak memiliki rumah/flat.

# ###  B. Income Type, Amount of Annuaty, Target, and House/Flat Ownership Status

# In[59]:


fig = sns.catplot(x = 'NAME_INCOME_TYPE',
            y = 'AMT_GOODS_PRICE', 
            hue = 'TARGET', 
            col = 'FLAG_OWN_REALTY', 
            kind = 'bar', palette = 'ch:start=0.2,rot=-.3_r',
            data = train1)

fig.set_xticklabels(rotation=45, horizontalalignment='right')
plt.show()


# Klien dengan jenis penghasilan cuti hamil (Maternity Leave) di FLAG_OWN_REALTY = Yes (yaitu memiliki rumah/flat) memiliki masalah dalam membayar pinjaman dibandingkan ketika FLAG_OWN_REALTY = No (yaitu tidak memiliki rumah/flat).

# ###  C. Family Status, Amount of Income, Target, and House/Flat Ownership Status

# In[60]:


fig = sns.catplot(x = 'NAME_FAMILY_STATUS', y = 'AMT_INCOME_TOTAL', 
                  hue = 'TARGET', 
                  col = 'FLAG_OWN_REALTY', 
                  kind = 'bar', palette = 'ch:start=0.2,rot=-.3_r',
                  data = train1)

fig.set_xticklabels(rotation=45, horizontalalignment='right')
plt.show()


# Klien yang sudah menikah dan memiliki rumah/flat (FLAG_OWN_REALTY = Yes) memiliki masalah dalam membayar kembali pinjaman dengan pendapatan menengah dibandingkan dengan saat klien tidak memiliki rumah/flat (FLAG_OWN_REALTY = No).

# #### Kelompok Berdasarkan Ownership Status

# ### A.Contract Type, The Number of Children, Target, and Car Ownership Status

# In[61]:


sns.catplot(x = 'NAME_CONTRACT_TYPE',
            y = 'CNT_CHILDREN', 
            hue = 'TARGET', 
            col = 'FLAG_OWN_CAR', 
            kind = 'bar', palette = 'ch:start=0.2,rot=-.3_r',
            data = train1)
plt.show()


# Untuk pinjaman bergulir di FLAG_OWN_CAR = Tidak (yaitu tidak memiliki mobil) memiliki masalah dalam membayar kembali pinjaman dibandingkan ketika FLAG_OWN_CAR = Ya (yaitu memiliki mobil).

# #### Kelompok Berdasarkan Contract Type

# ### A.Income Type, Amount of Credit, Target, and Contract Type

# In[62]:


fig = sns.catplot(x = 'NAME_INCOME_TYPE',
            y = 'AMT_CREDIT', 
            hue = 'TARGET', 
            col = 'NAME_CONTRACT_TYPE', 
            kind = 'bar', palette = 'ch:start=0.2,rot=-.3_r',
            data = train1)

fig.set_xticklabels(rotation=45, horizontalalignment='right')
plt.gcf().set_size_inches(15, 8)
plt.show()


# Untuk jenis pendapatan cuti hamil dengan pinjaman tunai, semua klien memiliki masalah dalam membayar kembali pinjaman dengan jumlah pinjaman sedang. Sementara semua klien dengan cuti hamil dan pinjaman bergulir tidak mengalami kesulitan untuk mengembalikan pinjaman.
# Untuk klien yang menganggur dengan pinjaman tunai, lebih dari 50% klien memiliki masalah dalam membayar kembali pinjaman dengan jumlah kredit menengah dari pinjaman tersebut. Sementara semua klien yang menganggur dengan pinjaman bergulir tidak mengalami kesulitan untuk mengembalikan pinjaman.
# Semua klien siswa tidak mengalami kesulitan untuk membayar kembali pinjaman baik dengan pinjaman tunai atau pinjaman bergulir dengan jumlah pinjaman kredit rendah hingga menengah.

# #### Kelompok berdasarkan Rating of Region where Client Lives

# ### A. Housing Type, Amount Credit of Loan, Target, and Rating of Region where Client Lives

# In[63]:


fig = sns.catplot(x = 'NAME_HOUSING_TYPE',
            y = 'AMT_CREDIT', 
            hue = 'TARGET', 
            col = 'REGION_RATING_CLIENT', 
            kind = 'bar', palette = 'ch:start=0.2,rot=-.3_r',
            data = train1)

fig.set_xticklabels(rotation=45, horizontalalignment='right')
plt.show()


# Klien yang tinggal di apartemen sewaan dan apartemen kantor dan wilayahnya memiliki peringkat 1, memiliki masalah dalam membayar kembali pinjaman dibandingkan dengan klien di wilayah dengan peringkat 2 untuk jumlah pinjaman kredit sedang.

# ### B. Education Type, Amount Credit of Loan, Target, and Rating of Region where Client Lives

# In[64]:


fig = sns.catplot(x = 'NAME_EDUCATION_TYPE',
            y = 'AMT_CREDIT', 
            hue = 'TARGET', 
            col = 'REGION_RATING_CLIENT', 
            kind = 'bar', palette = 'ch:start=0.2,rot=-.3_r',
            data = train1)

fig.set_xticklabels(rotation=45, horizontalalignment='right')
plt.show()


# Untuk klien yang memiliki gelar akademik dan tinggal di wilayah dengan peringkat 2, memiliki masalah dalam membayar kembali pinjaman untuk jumlah kredit pinjaman yang lebih tinggi. Dan, klien dengan gelar yang sama tetapi tinggal di wilayah dengan peringkat 3 memiliki masalah dalam membayar pinjaman untuk kredit pinjaman dalam jumlah sedang.

# ### C.  Family Status, Amount Credit of Loan, Target, and Rating of Region where Client Lives

# In[65]:


fig = sns.catplot(x = 'NAME_FAMILY_STATUS',
            y = 'AMT_CREDIT', 
            hue = 'TARGET', 
            col = 'REGION_RATING_CLIENT', 
            kind = 'bar', palette = 'ch:start=0.2,rot=-.3_r',
            data = train1)

fig.set_xticklabels(rotation=45, horizontalalignment='right')
plt.show()


# Klien yang berstatus keluarga janda, baik yang tinggal di daerah dengan rating 1, 2, atau 3, mengalami kesulitan dalam mengembalikan pinjaman untuk kredit pinjaman dalam jumlah sedang hingga tinggi.
# Klien yang memiliki status keluarga terpisah, dan tinggal di daerah dengan peringkat 3, memiliki masalah dalam membayar pinjaman dengan jumlah kredit pinjaman yang moderat dibandingkan dengan klien yang tinggal di daerah dengan peringkat 1 atau 2.

# # Data Cleaning

# ### Detecting Data Duplication

# In[66]:


print('Banyaknya Duplikasi adalah:', df_train.duplicated().sum())
# there is no duplication


# ### Detecting Missing Values

# In[67]:


# checking for empty elements
print('Missing values status:', df_train.isnull().values.any())
nvc = pd.DataFrame(df_train.isnull().sum(), columns=['Total Null Values'])
nvc['Percentage'] = (nvc['Total Null Values']/df_train.shape[0])*100
nvc.sort_values(by=['Percentage'], ascending=False).reset_index()


# In[68]:


# Drop features that have large number of missing values (± 50%)
df_train.drop(df_train.iloc[:, 44:91], inplace=True, axis=1)
df_train.drop(['OWN_CAR_AGE','EXT_SOURCE_1'], inplace=True, axis=1)


# In[69]:


# after drop some features
print('Status Missing Value:', df_train.isnull().values.any())
nvc = pd.DataFrame(df_train.isnull().sum().sort_values(), columns=['Total Null Values'])
nvc['Percentage'] = (nvc['Total Null Values']/df_train.shape[0])*100
nvc.sort_values(by=['Percentage'], ascending=False).reset_index()


# In[70]:


# distribution of numerical features that have missing values part.1
sns.set_style('whitegrid')
fig, ax = plt.subplots(2,2, figsize=(10,10))
sns.set_context('paper', font_scale=1)

sns.distplot(df_train['AMT_REQ_CREDIT_BUREAU_YEAR'], ax=ax[0][0])

sns.distplot(df_train['AMT_REQ_CREDIT_BUREAU_MON'], ax=ax[0][1])

sns.distplot(df_train['AMT_REQ_CREDIT_BUREAU_WEEK'], ax=ax[1][0])

sns.distplot(df_train['AMT_REQ_CREDIT_BUREAU_DAY'], ax=ax[1][1])


# In[71]:


# distribution of numerical features that have missing values part.2
sns.set_style('whitegrid')
fig, ax = plt.subplots(2,2, figsize=(10,10))
sns.set_context('paper', font_scale=1)

sns.distplot(df_train['AMT_REQ_CREDIT_BUREAU_HOUR'], ax=ax[0][0])

sns.distplot(df_train['AMT_REQ_CREDIT_BUREAU_QRT'], ax=ax[0][1])

sns.distplot(df_train['AMT_GOODS_PRICE'], ax=ax[1][0])

sns.distplot(df_train['AMT_ANNUITY'], ax=ax[1][1])


# In[72]:


# distribution of numerical features that have missing values part 3
sns.set_style('whitegrid')
fig, ax = plt.subplots(2,2, figsize=(10,10))
sns.set_context('paper', font_scale=1)

sns.distplot(df_train['DEF_60_CNT_SOCIAL_CIRCLE'], ax=ax[0][0])

sns.distplot(df_train['OBS_60_CNT_SOCIAL_CIRCLE'], ax=ax[0][1])

sns.distplot(df_train['DEF_30_CNT_SOCIAL_CIRCLE'], ax=ax[1][0])

sns.distplot(df_train['OBS_30_CNT_SOCIAL_CIRCLE'], ax=ax[1][1])


# In[73]:


# distribution of numerical features that have missing values part.4
sns.set_style('whitegrid')
fig, ax = plt.subplots(2, figsize=(10,10))
sns.set_context('paper', font_scale=1)

sns.distplot(df_train['CNT_FAM_MEMBERS'], ax=ax[0])

sns.distplot(df_train['DAYS_LAST_PHONE_CHANGE'], ax=ax[1])


# Terlihat bahwa sebaran fitur numerik yang disebutkan di atas miring, sehingga nilai yang hilang pada fitur tersebut akan diperhitungkan dengan median.

# In[74]:


# impute missing values with median because the data is skewed for numerical features
# impute missing values with mode for categorical features

category_columns = df_train.select_dtypes(include=['object']).columns.tolist()
integer_columns = df_train.select_dtypes(include=['int64','float64']).columns.tolist()

for column in df_train:
    if df_train[column].isnull().any():
        if(column in category_columns):
            df_train[column]=df_train[column].fillna(df_train[column].mode()[0])
        else:
            df_train[column]=df_train[column].fillna(df_train[column].median())


# In[75]:


# after imputation
print('Status Missing Value:', df_train.isnull().values.any())
print('\nThe number of missing values for each columns (after imputation): \n')
nvc = pd.DataFrame(df_train.isnull().sum().sort_values(), columns=['Total Null Values'])
nvc['Percentage'] = (nvc['Total Null Values']/df_train.shape[0])*100
nvc.sort_values(by=['Percentage'], ascending=False).reset_index()


# ## Detecting Outliers

# In[76]:


# dataset that only consist numerical features part.1
int_features = df_train[["CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]]


# In[77]:


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in int_features.items():
    sns.boxplot(y = k, data = int_features, ax=axs[index])
    index += 1
    if index == 5:
      break;
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[78]:


# dataset that only consist numerical features part.2
int_features = df_train[["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "CNT_FAM_MEMBERS"]]


# In[79]:


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in int_features.items():
    sns.boxplot(y = k, data = int_features, ax=axs[index])
    index += 1
    if index == 5:
      break;
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[80]:


# dataset that only consist numerical features part.3
int_features = df_train[["AMT_REQ_CREDIT_BUREAU_YEAR", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_HOUR"]]


# In[81]:


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in int_features.items():
    sns.boxplot(y = k, data = int_features, ax=axs[index])
    index += 1
    if index == 5:
      break;
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[82]:


# dataset that only consist numerical features part.4
int_features = df_train[["OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "REGION_POPULATION_RELATIVE"]]


# In[83]:


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in int_features.items():
    sns.boxplot(y = k, data = int_features, ax=axs[index])
    index += 1
    if index == 5:
      break;
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[84]:


# dataset that only consist numerical features pt.5
int_features = df_train[["REGION_RATING_CLIENT", "HOUR_APPR_PROCESS_START", "DAYS_LAST_PHONE_CHANGE", "FLAG_DOCUMENT_2", "AMT_REQ_CREDIT_BUREAU_QRT"]]


# In[85]:


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in int_features.items():
    sns.boxplot(y = k, data = int_features, ax=axs[index])
    index += 1
    if index == 5:
      break;
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# # Step Machine Learning Models

# ### - Label Encoding

# In[86]:


# label encoder for object features
df_train[["CODE_GENDER", "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", 
          "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
         "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE"]] = df_train[["CODE_GENDER", "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", 
          "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
         "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE"]].apply(LabelEncoder().fit_transform)


# In[87]:


df_train.head()


# ### - Feature Selection

# Drop Unnecessary Feature

# In[88]:


df_train.drop(['SK_ID_CURR'], inplace=True, axis=1)


# In[89]:


df_train.head()


# ###### Convert Negative Values to Positive Values

# The features that have negative values are DAYS_BIRTH, DAYS_EMPLOYED, DAYS_ID_PUBLISH, DAYS_REGISTRATION, and DAYS_LAST_PHONE_CHANGE

# In[90]:


df_train.iloc[:,16:20] = df_train.iloc[:,16:20].abs()
df_train.iloc[:,45] = df_train.iloc[:,45].abs()


# ##### Feature Selection

# In[91]:


x = df_train.drop(['TARGET'], axis=1)
y = df_train['TARGET']


# In[92]:


# feature selection
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis = 1)
featureScores.columns = ['Features', 'Score']
print('The features that correlate well with target feature:\n')
featureScores.sort_values(by=['Score'], ascending=False)


# Dapat terlihat bahwa fitur yang berkorelasi baik dengan kesulitan pembayaran.
# 1. Best features: DAYS_EMPLOYED, AMT_GOODS_PRICE, and AMT_CREDIT
# 
# 2. Worst features: FLAG_MOBIL, FLAG_CONT_MOBILE, and AMT_REQ_CREDIT_BUREAU_HOUR

# #### Handling Data Imbalance¶

# In[93]:


# create two different dataframe of majority and minority class 
df_majority = df_train[(df_train['TARGET']==0)] 
df_minority = df_train[(df_train['TARGET']==1)] 

# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 282686, # to match majority class
                                 random_state=42)  # reproducible results

# combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])


# In[94]:


fig = plt.figure(figsize = (12,7))
df_upsampled['TARGET'].value_counts(normalize=True).plot(kind='bar', color= ['midnightblue','lightgrey'], alpha = 0.9, rot=0)
plt.title('The Distribution of Clients Repayment Abilities\n', fontsize=14)
plt.ylabel('Percentage of the Customers\n', fontsize=12)
plt.xlabel('\nPayment Difficulty Status', fontsize=12)
plt.show()


# ### Data Splitting

# In[95]:


# define x and y features (top 20 features)
x_balanced = df_upsampled[['DAYS_EMPLOYED', 'AMT_GOODS_PRICE', 'AMT_CREDIT', 
                           'DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'DAYS_REGISTRATION', 
                           'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH', 'AMT_ANNUITY', 
                           'ORGANIZATION_TYPE', 'NAME_INCOME_TYPE', 'REG_CITY_NOT_WORK_CITY',
                          'CODE_GENDER', 'EXT_SOURCE_2', 'REG_CITY_NOT_LIVE_CITY', 'NAME_EDUCATION_TYPE',
                          'DEF_30_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_3', 'DEF_60_CNT_SOCIAL_CIRCLE', 'LIVE_CITY_NOT_WORK_CITY']]
y_balanced = df_upsampled['TARGET']


# In[96]:


# splitting tha data
X_train, X_test, y_train, y_test = train_test_split(x_balanced, y_balanced, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)


# In[97]:


# normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# ### Model Algorithms
# Beberapa algoritma yang akan digunakan untuk melakukan tes yaitu:
# 
# 1.Logistic Regression
# 
# 2.Gaussian Naive Bayes
# 
# 3.Decision Tree
# 
# 4.Random Forest
# 
# 5.K-Nearest Neighbor
# 
# 6.Neural Network

# #### 1. LOGISTIC REGRESSION

# In[98]:


# train the model
log_model = LogisticRegression().fit(X_train, y_train)
print(log_model)


# In[99]:


# predict data train
y_train_pred_log = log_model.predict(X_train)

# print classification report
print('Classification Report Training Model (Logistic Regression):')
print(classification_report(y_train, y_train_pred_log))


# In[100]:


# form confusion matrix as a dataFrame
confusion_matrix_log = pd.DataFrame((confusion_matrix(y_train, y_train_pred_log)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_log, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Training Model\n(Logistic Regression)\n', fontsize=18, color='blue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[101]:


# predict data test
y_test_pred_log = log_model.predict(X_test)

# print classification report
print('Classification Report Testing Model (Logistic Regression):')
print(classification_report(y_test, y_test_pred_log))


# In[102]:


# form confusion matrix as a dataFrame
confusion_matrix_log = pd.DataFrame((confusion_matrix(y_test, y_test_pred_log)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_log, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Testing Model\n(Logistic Regression)\n', fontsize=18, color='blue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[103]:


acc_log_train=round(log_model.score(X_train,y_train)*100,2)
acc_log_test=round(log_model.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Test Accuracy: % {}".format(acc_log_test))


# In[104]:


# ROC scores
roc_auc_log = round(roc_auc_score(y_test, y_test_pred_log),4)
print('ROC AUC:', roc_auc_log)


# #### 2. GAUSSIAN NAIVE BAYES

# In[105]:


# train the model
gnb_model = GaussianNB().fit(X_train, y_train)
print(gnb_model)


# In[106]:


# predict data train
y_train_pred_gnb = gnb_model.predict(X_train)

# print classification report
print('Classification Report Training Model (Gaussian Naive Bayes):')
print(classification_report(y_train, y_train_pred_gnb))


# In[107]:


# form confusion matrix as a dataFrame
confusion_matrix_gnb = pd.DataFrame((confusion_matrix(y_train, y_train_pred_gnb)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_gnb, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Training Model\n(Gaussian Naive Bayes)\n', fontsize=18, color='red')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[108]:


# predict data test
y_test_pred_gnb = gnb_model.predict(X_test)

# print classification report
print('Classification Report Testing Model (Gaussian Naive Bayes):')
print(classification_report(y_test, y_test_pred_gnb))


# In[109]:


# form confusion matrix as a dataFrame
confusion_matrix_gnb = pd.DataFrame((confusion_matrix(y_test, y_test_pred_gnb)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_gnb, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Testing Model\n(Gaussian Naive Bayes)\n', fontsize=18, color='green')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[110]:


accuracy_nb_train=round(gnb_model.score(X_train,y_train)*100,2)
accuracy_nb_test=round(gnb_model.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(accuracy_nb_train))
print("Test Accuracy: % {}".format(accuracy_nb_test))


# In[111]:


# ROC scores
roc_auc_gnb = round(roc_auc_score(y_test, y_test_pred_gnb),4)
print('ROC AUC:', roc_auc_gnb)


# ### 3. DECISION TREE

# In[112]:


# train the model
dt_model = DecisionTreeClassifier().fit(X_train,y_train)
print(dt_model)


# In[113]:


# predict data train
y_train_pred_dt = dt_model.predict(X_train)

# print classification report
print('Classification Report Training Model (Decision Tree Classifier):')
print(classification_report(y_train, y_train_pred_dt))


# In[114]:


# form confusion matrix as a dataFrame
confusion_matrix_dt = pd.DataFrame((confusion_matrix(y_train, y_train_pred_dt)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_dt, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Training Model\n(Decision Tree Classifier)\n', fontsize=18, color='blue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[115]:


# predict data test
y_test_pred_dt = dt_model.predict(X_test)

# print classification report
print('Classification Report Testing Model (Decision Tree Classifier):')
print(classification_report(y_test, y_test_pred_dt))


# In[116]:


# form confusion matrix as a dataFrame
confusion_matrix_dt = pd.DataFrame((confusion_matrix(y_test, y_test_pred_dt)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_dt, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Testing Model\n(Decision Tree Classifier)\n', fontsize=18, color='orange')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[117]:


accuracy_dt_train=round(dt_model.score(X_train,y_train)*100,2)
accuracy_dt_test=round(dt_model.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(accuracy_dt_train))
print("Test Accuracy: % {}".format(accuracy_dt_test))


# In[118]:


# ROC scores
roc_auc_dt = round(roc_auc_score(y_test, y_test_pred_dt),4)
print('ROC AUC:', roc_auc_dt)


# ### 4. RANDOM FOREST

# In[120]:


# train the model
rf_model = RandomForestClassifier().fit(X_train, y_train)
print(rf_model)


# In[121]:


# predict data train
y_train_pred_dt = rf_model.predict(X_train)

# print classification report
print('Classification Report Training Model (Decision Tree Classifier):')
print(classification_report(y_train, y_train_pred_dt))


# In[679]:


# form confusion matrix as a dataFrame
confusion_matrix_rf = pd.DataFrame((confusion_matrix(y_train, y_train_pred_dt)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_rf, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Training Model\n(Random Forest)\n', fontsize=18, color='green')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[680]:


# predict data test
y_test_pred_rf = rf_model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

# print classification report
print('Classification Report Testing Model (Random Forest Classifier):')
print(classification_report(y_test, y_test_pred_rf))


# In[681]:


# form confusion matrix as a dataFrame
confusion_matrix_rf = pd.DataFrame((confusion_matrix(y_test, y_test_pred_rf)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_rf, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Testing Model\n(Random Forest)\n', fontsize=18, color='red')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[682]:


accuracy_rf_train=round(rf_model.score(X_train,y_train)*100,2)
accuracy_rf_test=round(rf_model.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(accuracy_rf_train))
print("Test Accuracy: % {}".format(accuracy_rf_test))


# In[683]:


# ROC scores
roc_auc_rf = round(roc_auc_score(y_test, y_test_pred_rf),4)
print('ROC AUC:', roc_auc_rf)


# In[690]:


# important features
importances_rf = pd.Series(rf_model.feature_importances_, index=x_balanced.columns).sort_values(ascending=True)

plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(10,10))
fig = importances_rf.plot(kind ='barh', color ='coral')
plt.title('Features Importance Plot\n', fontsize=14)
plt.show()

fig.figure.tight_layout()
fig.figure.savefig('top feature.png')


# Plot di atas membuktikan bahwa 5 fitur terpenting adalah EXT_SOURCE_2, EXT_SOURCE_3, DAYS_BIRTH, DAYS_ID_PUBLISH, dan DAYS_REGISTRATION

# ### 5. K-NEAREST NEIGHBORS

# In[122]:


# train the model
knn_model = KNeighborsClassifier().fit(X_train,y_train)
print(knn_model)


# In[ ]:


# predit data train
y_train_pred_knn = knn_model.predict(X_train)

# print classification report
print('Classification Report Training Model (K-Nearest Neighbors):')
print(classification_report(y_train, y_train_pred_knn))


# In[ ]:


# form confusion matrix as a dataFrame
confusion_matrix_knn = pd.DataFrame((confusion_matrix(y_train, y_train_pred_knn)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_knn, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Training Model\n(K-Nearest Neighbors)\n', fontsize=18, color='black')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[ ]:


# predit data test
y_test_pred_knn = knn_model.predict(X_test)

# print classification report
print('Classification Report Testing Model (K-Nearest Neighbors):')
print(classification_report(y_test, y_test_pred_knn))


# In[ ]:


# form confusion matrix as a dataFrame
confusion_matrix_knn = pd.DataFrame((confusion_matrix(y_test, y_test_pred_knn)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_knn, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Testing Model\n(K-Nearest Neighbors)\n', fontsize=18, color='black')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[ ]:


acc_knn_train=round(knn_model.score(X_train,y_train)*100,2)
acc_knn_test=round(knn_model.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(acc_knn_train))
print("Test Accuracy: % {}".format(acc_knn_test))


# In[ ]:


# ROC scores
roc_auc_knn = round(roc_auc_score(y_test, y_test_pred_knn),4)
print('ROC AUC:', roc_auc_knn)


# ### 6. Neural Network

# In[ ]:


# train the model
nn_model = MLPClassifier().fit(X_train, y_train)


# In[ ]:


# predit data train
y_train_pred_nn = nn_model.predict(X_train)

# print classification report
print('Classification Report Training Model (Neural Network):')
print(classification_report(y_train, y_train_pred_nn))


# In[ ]:


# form confusion matrix as a dataFrame
confusion_matrix_nn = pd.DataFrame((confusion_matrix(y_train, y_train_pred_nn)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_nn, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Training Model\n(Neural Network)\n', fontsize=18, color='black')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[ ]:


# predit data test
y_test_pred_nn = nn_model.predict(X_test)

# print classification report
print('Classification Report Testing Model (Neural Network):')
print(classification_report(y_test, y_test_pred_nn))


# In[ ]:


# form confusion matrix as a dataFrame
confusion_matrix_nn = pd.DataFrame((confusion_matrix(y_test, y_test_pred_nn)), ('No Payment Difficulties', 'Payment Difficulties'), ('No Payment Difficulties', 'Payment Difficulties'))

# plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_nn, annot=True, annot_kws={'size': 14}, fmt='d', cmap='bone_r')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Testing Model\n(Neural Network)\n', fontsize=18, color='black')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()


# In[ ]:


acc_nn_train=round(nn_model.score(X_train,y_train)*100,2)
acc_nn_test=round(nn_model.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(acc_nn_train))
print("Test Accuracy: % {}".format(acc_nn_test))


# In[ ]:


# ROC scores
roc_auc_nn = round(roc_auc_score(y_test, y_test_pred_nn),4)
print('ROC AUC:', roc_auc_nn)


# ###### Model Selection

# In[ ]:


results = pd.DataFrame([["Logistic Regression", 0.6716, 0.6729, 0.6729],["Gaussian Naive Bayes", 0.6024, 0.6039, 0.604],
                       ["Decision Tree", 1, 0.8826, 0.8826],["Random Forest", 1, 0.9965, 0.9965],
                       ["K-Nearest Neighbor", 0.9156, 0.8807, 0.8806], ["Neural Network", 0.7001, 0.6948, 0.6948]],
                        columns = ["Models", "Training Accuracy Score", "Testing Accuracy Score", "ROC Score"])

results.sort_values(by=['Training Accuracy Score', 'Testing Accuracy Score'], ascending=False).style.background_gradient(cmap='Blues')


# In[157]:


plt.figure(figsize = (14,6))
plt.title('AMT CREDIT')
sns.set_color_codes("pastel")
sns.histplot(df_train['AMT_CREDIT'],kde=True,bins=200, color="blue")
plt.show()


# # Predict Using Test Data

# In[169]:


df_test = pd.read_csv("../Final Project Rakamin/application_test.csv")
print('This dataset has %d rows dan %d columns.\n' % df_test.shape)
df_test.head()


# ##### Data Preprocessing

# Detecting Duplication

# In[493]:


print('The number of duplication is:', df_test.duplicated().sum())
# there is no duplication


# Detecting Missing Values

# In[499]:


# drop features that have large number of missing values (± 50%)
df_test.drop(df_test.iloc[:, 43:90], inplace=True, axis=1)
df_test.drop(['OWN_CAR_AGE','EXT_SOURCE_1'], inplace=True, axis=1)


# In[500]:


# after drop some features
print('Missing values status:', df_test.isnull().values.any())
tvc = pd.DataFrame(df_test.isnull().sum(), columns=['Total Null Values'])
tvc['Percentage'] = (tvc['Total Null Values']/df_test.shape[0])*100
tvc.sort_values(by=['Percentage'], ascending=False).reset_index()


# In[ ]:


# impute missing values with median because the data is skewed for numerical features
# impute missing values with mode for categorical features

category_columns = df_test.select_dtypes(include=['object']).columns.tolist()
integer_columns = df_test.select_dtypes(include=['int64','float64']).columns.tolist()

for column in df_test:
    if df_test[column].isnull().any():
        if(column in category_columns):
            df_test[column]=df_test[column].fillna(df_test[column].mode()[0])
        else:
            df_test[column]=df_test[column].fillna(df_test[column].median())


# In[ ]:


# after imputation
print('Missing values status:', df_test.isnull().values.any())
tvc = pd.DataFrame(df_test.isnull().sum(), columns=['Total Null Values'])
tvc['Percentage'] = (tvc['Total Null Values']/df_test.shape[0])*100
tvc.sort_values(by=['Percentage'], ascending=False).reset_index()


# ###### Label Encoding

# In[ ]:


# label encoder for object features
df_test[["CODE_GENDER", "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", 
          "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
         "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE"]] = df_test[["CODE_GENDER", "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", 
          "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
         "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE"]].apply(LabelEncoder().fit_transform)


# In[ ]:


df_test.head()


# In[ ]:


df_test.iloc[:,16:20] = df_test.iloc[:,16:20].abs()
df_test.iloc[:,45] = df_test.iloc[:,45].abs()


# #### Prediction

# In[ ]:


pred_test = df_test[['DAYS_EMPLOYED', 'AMT_GOODS_PRICE', 'AMT_CREDIT', 
                           'DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'DAYS_REGISTRATION', 
                           'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH', 'AMT_ANNUITY', 
                           'ORGANIZATION_TYPE', 'NAME_INCOME_TYPE', 'REG_CITY_NOT_WORK_CITY',
                          'CODE_GENDER', 'EXT_SOURCE_2', 'REG_CITY_NOT_LIVE_CITY', 'NAME_EDUCATION_TYPE',
                          'DEF_30_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_3', 'DEF_60_CNT_SOCIAL_CIRCLE', 'LIVE_CITY_NOT_WORK_CITY']]
pred_test.head()


# In[ ]:


# lets predict!
predict = pd.Series(rf_model.predict(pred_test), name = "TARGET").astype(int)
results = pd.concat([df_test['SK_ID_CURR'], predict],axis = 1)
results.to_csv("predict application.csv", index = False)
results.head()

