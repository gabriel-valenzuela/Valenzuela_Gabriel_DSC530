# Weather Phenomenons and Fire Outbreaks

## Data Exploration and Analysis Project 

### Objective

A data exploration and analysis to determine if the severity of the weather phenomenons La Nina and El Nino increases the number of fire outbreaks that occur within the rainforests of Brazil or does it not play any factor on the recent increases of fire outbreaks in the Amazon. 

![Explanation of the Weather Storms](https://github.com/gabriel-valenzuela/Valenzuela_Gabriel_DSC530/blob/master/Final%20Project/Images/Nina_winterandsummer_620_from_climate.gov_.jpg)

<img src="Final%20Project/Images/Nina_winterandsummer_620_from_climate.gov_.jpg?raw=true"/>

### Environment

Python was utilized within a jupyter notebook to complete the analysis

#### Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy import stats as ss
from sklearn.linear_model import LinearRegression
```

### Data Sets

LINK

### Data Exploration Methods

Descriptive Analysis and Histograms were used to determine any outliers as well as a summary of the characteristcs

```python
deforestationArea['PA'].hist()
plt.xlabel("State of PA", fontsize=15)
plt.ylabel("Frequency",fontsize=15)

print("State of PA Descriptive Characteristics")
print(deforestationArea['PA'].describe())
print("\n" + "Median: " + str(np.median(deforestationArea['PA'])))
print("Mode: " + str(stats.mode(deforestationArea['PA'])))
print("Spread: " + str(np.ptp(deforestationArea['PA'])))
```

Probabilty mass function was utilized to determine the number of occurences that either La Nina or El Nino would appear in the Pacific Ocean. 

```python
df3=weatherPhenomenon['severity'].value_counts()
sum1=len(weatherPhenomenon)
df4=pd.DataFrame(df3)
df4=df4.rename(columns={df4.columns[0]:'item_counts'})
df4['item']=df4.index
df4['probability']=df4['item_counts']/sum1
print(df4)
plt.bar(df4['item'],df4['probability'])
plt.show()
```
![Probability Mass Function Severity](https://github.com/gabriel-valenzuela/Valenzuela_Gabriel_DSC530/blob/master/Final%20Project/Images/PhenomenonSeverityPMF.png)

![Probability Mass Function Type](https://github.com/gabriel-valenzuela/Valenzuela_Gabriel_DSC530/blob/master/Final%20Project/Images/PhenomenonTypePMF.png)

Cumulative mass function to examine the severity of the weather phenomenon and the likelihood that they would occur. 

```python
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
```

Exponential Distribution was used to understand the time it would take for another fire outbreak to be reported one after another. 

```python
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
```

![Exponential Distribution](https://github.com/gabriel-valenzuela/Valenzuela_Gabriel_DSC530/blob/master/Final%20Project/Images/FireSpots.Exponential.png)

### Analysis

Difference of Means to detemine how reproduceable these weather events are

```python
N = 20

var_StartYear = fireLaNinaStart['firespots'].var(ddof=1)
var_EndYear = fireLaNinaEnd['firespots'].var(ddof=1)

s = np.sqrt((var_StartYear + var_EndYear)/2)

t = (fireLaNinaStart['firespots'].mean()-fireLaNinaEnd['firespots'].mean())/(s*np.sqrt(2/N))

df = 2*N - 2

p = 1 - ss.t.cdf(t,df=df)

print("t = " + str(t))
print("p = " + str(2*p))
```


Linear Regression Analysis for Prediction of fire outbreaks from 2020-2025

```python
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

```


### Visualization

A scatter plot was used to show the number of outbreaks at the beginning and at the end of the year that La Nina storms were spotted. 
```python
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
```
![La Nina Fire Oubreaks](https://github.com/gabriel-valenzuela/Valenzuela_Gabriel_DSC530/blob/master/Final%20Project/Images/LaNinaFireOutbreaks.png)
