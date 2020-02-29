#!/usr/bin/env python
# coding: utf-8

# In[295]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy import stats as ss

#import datasets
deforestationArea = pd.read_csv("C:\\Users\\Gabe\\Documents\\Bellevue University\\Data Exploration and Analysis\\Final Project\\def_area_2004_2019.csv")
weatherPhenomenon = pd.read_csv("C:\\Users\\Gabe\\Documents\\Bellevue University\\Data Exploration and Analysis\\Final Project\\el_nino_la_nina_1999_2019.csv")
amazonFireOutbreaks = pd.read_csv("C:\\Users\\Gabe\\Documents\\Bellevue University\\Data Exploration and Analysis\\Final Project\inpe_brazilian_amazon_fires_1999_2019.csv")

print(deforestationArea.head())
print(weatherPhenomenon.head())
print(amazonFireOutbreaks.head())


# In[200]:


# Descripive Analysis and Histogram
deforestationArea['PA'].hist()
plt.xlabel("State of PA", fontsize=15)
plt.ylabel("Frequency",fontsize=15)

print("State of PA Descriptive Characteristics")
print(deforestationArea['PA'].describe())
print("\n" + "Median: " + str(np.median(deforestationArea['PA'])))
print("Mode: " + str(stats.mode(deforestationArea['PA'])))
print("Spread: " + str(np.ptp(deforestationArea['PA'])))


# In[201]:


# Descripive Analysis and Histogram
deforestationArea['AMZ LEGAL'].hist()
plt.xlabel("Sum of Deforestation", fontsize=15)
plt.ylabel("Frequency",fontsize=15)

print("Sum of Deforestation Descriptive Characteristics")
print(deforestationArea['AMZ LEGAL'].describe())
print("\n" + "Median: " + str(np.median(deforestationArea['AMZ LEGAL'])))
print("Mode: " + str(stats.mode(deforestationArea['AMZ LEGAL'])))
print("Spread: " + str(np.ptp(deforestationArea['AMZ LEGAL'])))


# In[202]:


# Descripive Analysis and Histogram
amazonFireOutbreaks['firespots'].hist()
plt.xlabel("Fire Spots", fontsize=15)
plt.ylabel("Frequency",fontsize=15)

print("Fire Spots Descriptive Characteristics")
print(amazonFireOutbreaks['firespots'].describe())
print("\n" + "Median: " + str(np.median(amazonFireOutbreaks['firespots'])))
print("Mode: " + str(stats.mode(amazonFireOutbreaks['firespots'])))
print("Spread: " + str(np.ptp(amazonFireOutbreaks['firespots'])))


# In[203]:


# Descripive Analysis and Histogram
amazonFireOutbreaks['latitude'].hist()
plt.xlabel("Latitude of Fire Outbreaks", fontsize=15)
plt.ylabel("Frequency",fontsize=15)

print("Latitude of Fire Outbreaks Descriptive Characteristics")
print(amazonFireOutbreaks['latitude'].describe())
print("\n" + "Median: " + str(np.median(amazonFireOutbreaks['latitude'])))
print("Mode: " + str(stats.mode(amazonFireOutbreaks['latitude'])))
print("Spread: " + str(np.ptp(amazonFireOutbreaks['latitude'])))


# In[204]:


# Descripive Analysis and Histogram
amazonFireOutbreaks['longitude'].hist()
plt.xlabel("Longitude of Fire Outbreaks", fontsize=15)
plt.ylabel("Frequency",fontsize=15)

print("Longitude of Fire Outbreaks Descriptive Characteristics")
print(amazonFireOutbreaks['longitude'].describe())
print("\n" + "Median: " + str(np.median(amazonFireOutbreaks['longitude'])))
print("Mode: " + str(stats.mode(amazonFireOutbreaks['longitude'])))
print("Spread: " + str(np.ptp(amazonFireOutbreaks['longitude'])))


# In[205]:


# Probability Mass Function
df1=weatherPhenomenon['phenomenon'].value_counts()
sum1=len(weatherPhenomenon)
df2=pd.DataFrame(df1)
df2=df2.rename(columns={df2.columns[0]:'item_counts'})
df2['item']=df2.index
df2['probability']=df2['item_counts']/sum1
print(df2)
plt.bar(df2['item'],df2['probability'])
plt.show()


# In[206]:


# Probability Mass Function
df3=weatherPhenomenon['severity'].value_counts()
sum1=len(weatherPhenomenon)
df4=pd.DataFrame(df3)
df4=df4.rename(columns={df4.columns[0]:'item_counts'})
df4['item']=df4.index
df4['probability']=df4['item_counts']/sum1
print(df4)
plt.bar(df4['item'],df4['probability'])
plt.show()


# In[208]:


# Probability Mass Function
df5=amazonFireOutbreaks['month'].value_counts()
sum1=len(weatherPhenomenon)
df6=pd.DataFrame(df5)
df6=df6.rename(columns={df6.columns[0]:'item_counts'})
df6['item']=df6.index
df6['probability']=df6['item_counts']/sum1
print(df6)


# In[209]:


# Cumulative Mass Function
def ecdf(data):
    """Computer ECDF"""
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

#x,y = ecdf(amazonFireOutbreaks['month'])
a,b = ecdf(weatherPhenomenon['severity'])
#_ = plt.plot(x,y)
_ = plt.plot(a,b)
_ = plt.xlabel('Severity of Weather Phenomenon')
_ = plt.ylabel('ECDF')
plt.show()


# In[210]:


# Exponential Distribution
def successive_poisson(tau1, size=1):
    t1 = np.random.exponential(tau1, size=size)
    return t1


# (mean of firespots, 10,000,000 samples)    
waiting_times = successive_poisson(1167, size=10000000)
# Make the histogram
_ = plt.hist(waiting_times, bins=100, histtype='step',
             density=True)

# Label axes
_ = plt.xlabel('total waiting time (fires)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


# In[338]:


laNinaYear = weatherPhenomenon['phenomenon']=="La Nina"
laNina = weatherPhenomenon[elNinoYear]
print(laNina)

# Fire outbreaks the year El Nino started
fireLaNinaStart = amazonFireOutbreaks.query('year == 2000 or year == 2005 or year == 2008 or year == 2016 or year == 2017 or year == 2011 or year == 1999 or year == 2007 or year == 2010')

# Fire outbreaks the year El Nino ended
fireLaNinaEnd = amazonFireOutbreaks.query('year == 2001 or year == 2006 or year == 2009 or year == 2017 or year == 2018 or year == 2012 or year == 2000 or year == 2008 or year == 2011')
#sumFirebyYear = fireElNino.groupby('year')['firespots'].sum()

print(np.cov(fireLaNinaStart['year'],fireLaNinaStart['firespots']))
print(np.corrcoef(fireLaNinaStart['year'],fireLaNinaStart['firespots']))


print(np.cov(fireLaNinaEnd['year'],fireLaNinaEnd['firespots']))
print(np.corrcoef(fireLaNinaEnd['year'],fireLaNinaEnd['firespots']))

# Scatter Plot
startYear= plt.plot(fireLaNinaStart['year'],fireLaNinaStart['firespots'],'o', c = 'red')
red_patch = mpatches.Patch(color='red', label = 'Start Year')
endYear = plt.plot(fireLaNinaEnd['year'],fireLaNinaEnd['firespots'],'o', c = 'yellow')
yellow_patch = mpatches.Patch(color='yellow', label = 'End Year')
plt.xticks(np.arange(2001,2020,2))
plt.legend(handles=[red_patch,yellow_patch])
plt.title("La Nina Fire Outbreaks")
plt.xlabel("Years")
plt.ylabel("Fire Outbreaks")
plt.show()


# In[339]:


#Difference of Means (t-Statistic)
N = 20

var_StartYear = fireLaNinaStart['firespots'].var(ddof=1)
var_EndYear = fireLaNinaEnd['firespots'].var(ddof=1)

s = np.sqrt((var_StartYear + var_EndYear)/2)

t = (fireLaNinaStart['firespots'].mean()-fireLaNinaEnd['firespots'].mean())/(s*np.sqrt(2/N))

df = 2*N - 2

p = 1 - ss.t.cdf(t,df=df)

print("t = " + str(t))
print("p = " + str(2*p))


# In[340]:


# Linear Regression Analysis
from sklearn.linear_model import LinearRegression
y_array = np.array(fireLaNinaStart['firespots'])
x_array = np.array(fireLaNinaStart['year']).reshape((-1,1))
model = LinearRegression().fit(x_array,y_array)
r_sq = model.score(x_array, y_array)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred = model.predict(x_array)
print('predicted response:', y_pred, sep='\n')

# Prediction of 2020-2025
x_new = np.arange(2020,2026).reshape((-1, 1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

