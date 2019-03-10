
___
This notebook tries to address two questions for rental lease:
1. is there a seasonality effect for certain months and areas?
1. how long should the lease be? 12, or 24?


```python
%matplotlib inline
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
import time
import glob
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn import datasets, linear_model
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
import sklearn.preprocessing as pp
from fbprophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
from datetime import timedelta,datetime
import warnings
warnings.filterwarnings("ignore")
```

    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



<style>.container { width:90% !important; }</style>


## Scope of analysis
- Data pull
- Pre-process data
- Forecast rental amount of each month for each district*
- Find out which district in each month is cheapest (and most expensive)
- Find out the amount of savings for each district
- Predict rental prices for each district for the next 24 months
- Find out the amount of savings for each district from prediction of next 24 months
- Get total savings
- Plot out districts that are top 10 in savings to see predict fit

*1 BR is only looked into here, although data for 2BR and 3BR are pulled too

## Summary
- Total savings of lease can be up to `$`7,854 (avg total savings: `$`2538.05)
    - Savings from extending lease to 24 months, can be up to `$`2070 (avg total savings: `$`644)
    - Savings from leasing on the lowest month, can be up to `$`7854 (avg total savings: `$`2254)
- The districts that have the most savings are `7, 1, 21 ,2, 3`

## Data pull
- due to the nature of the website and how the data can be pulled from the website
- a web scrap 'script' needs to be in placed so that we will not have to manually go through the steps and obtain the data
- Selenium is used to pull all the data required

### Initialize selenium browser


```python
driver = webdriver.Chrome()
driver.get("https://www.ura.gov.sg/realEstateIIWeb/resiRental/changeDisplayUnit.action")
```

### Loop through all condos to obtain data


```python
# for i in range(0,600):
#     print(i)
#     from_date_element = driver.find_element_by_name("from_Date_Prj")
#     to_date_element = driver.find_element_by_name("to_Date_Prj")
#     Select(from_date_element).select_by_value("JAN-2016")
#     Select(to_date_element).select_by_value("DEC-2018")
#     for j in range(0,5):
#         text = "addToProject_" + str(i*5+j)
#         project_element = driver.find_element_by_id(text)
#         project_element.click()
#         time.sleep(0.5)
#     driver.find_element_by_xpath('//*[@id="searchForm_0"]').click()
#     driver.find_element_by_xpath('//*[@id="SubmitSortForm"]/div[1]/div[3]').click()
#     time.sleep(1)
#     driver.find_element_by_xpath('//*[@id="SubmitSortForm"]/div[1]/div[4]').click()
#     print("finish")

driver.stop_client()
```

### Read csv files, and aggregate them


```python
file = "../data/aggregate_raw_data.csv"
if not os.path.exists(file):
    aggregate_df = None
    for files in glob.glob("../data/*.csv"):
        processed_raw_df = (
            pd.read_csv(files)
            .dropna()
            .reset_index()
        )
        processed_raw_df.columns = processed_raw_df.iloc[0]
        processed_raw_df = processed_raw_df.reindex(processed_raw_df.index.drop(0))

        if aggregate_df is None:
            aggregate_df = processed_raw_df
        else:
            aggregate_df = (
                aggregate_df
                .append(processed_raw_df)
            )
    aggregate_df.to_csv('../data/aggregate_raw_data.csv',index=False)
else:
    aggregate_df = pd.read_csv("../data/aggregate_raw_data.csv")
```

## Pre process data
- minor pre-process data
- pre-process data
- decision if mean or median of monthly rental should be used for predicted variable

### Minor pre-process data
- change column names and data type of columns


```python
# change column names and drop first column and change data type of columns
aggregate_df_processed = (
    aggregate_df
    .drop(['S/N'],1)
    .rename(columns={"Building/Project Name":"building_name","Street Name":"street_name","Postal District":"district",
                     "No. of Bedroom(for Non-Landed Only)":"num_bedrooms","Monthly Gross Rent($)":"monthly_rent",
                     "Floor Area (sq ft)":"sq_ft","Lease Commencement Date":"lease_month","Type":"type"})
    .pipe(lambda x:x.assign(lease_month = pd.to_datetime(x.lease_month)))
    .pipe(lambda x:x.assign(monthly_rent = x.monthly_rent.astype(float)))
    .query("(type == 'Non-landed Properties' | type == 'Executive Condominium')")
)

aggregate_df_processed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>building_name</th>
      <th>street_name</th>
      <th>district</th>
      <th>type</th>
      <th>num_bedrooms</th>
      <th>monthly_rent</th>
      <th>sq_ft</th>
      <th>lease_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DAKOTA RESIDENCES</td>
      <td>DAKOTA CRESCENT</td>
      <td>14</td>
      <td>Non-landed Properties</td>
      <td>2</td>
      <td>3600.0</td>
      <td>1000 to 1100</td>
      <td>2018-12-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DAKOTA RESIDENCES</td>
      <td>DAKOTA CRESCENT</td>
      <td>14</td>
      <td>Non-landed Properties</td>
      <td>2</td>
      <td>3700.0</td>
      <td>1000 to 1100</td>
      <td>2018-12-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DAKOTA RESIDENCES</td>
      <td>DAKOTA CRESCENT</td>
      <td>14</td>
      <td>Non-landed Properties</td>
      <td>4</td>
      <td>6300.0</td>
      <td>1800 to 1900</td>
      <td>2018-12-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DAKOTA RESIDENCES</td>
      <td>DAKOTA CRESCENT</td>
      <td>14</td>
      <td>Non-landed Properties</td>
      <td>2</td>
      <td>3600.0</td>
      <td>1000 to 1100</td>
      <td>2018-12-01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DAKOTA RESIDENCES</td>
      <td>DAKOTA CRESCENT</td>
      <td>14</td>
      <td>Non-landed Properties</td>
      <td>3</td>
      <td>5000.0</td>
      <td>1200 to 1300</td>
      <td>2018-12-01</td>
    </tr>
  </tbody>
</table>
</div>



### Pre-process data
- fill in missing data of number of bedrooms
- set the size of the property
- one hot encode which month the rental lease is signed
- introduce trend variables (up to third order)


```python
aggregate_df_processed_final = (
aggregate_df_processed
    .query("(type == 'Non-landed Properties' | type == 'Executive Condominium')")
    .pipe(lambda x:x.assign(sq_ft = np.where(
                                            x.sq_ft == '>3000',"3100 to 3100",
                                            np.where(x.sq_ft == '>8000',"8100 to 8100",x.sq_ft)
    )))
    .pipe(lambda x:x.assign(sq_ft_altered = x.sq_ft.str.split(' to ').str[1].astype(float)))
    .pipe(lambda x:x.assign(num_bedrooms_altered = np.where(
        x.sq_ft_altered <= 646.085859, 1, np.where(x.sq_ft_altered <= 1037.376865, 2,
                                                  np.where(x.sq_ft_altered <= 1521,3,
                                                          np.where(x.sq_ft_altered <= 2346,4,
                                                                  np.where(x.sq_ft_altered <= 3034,5,
                                                                          np.where(x.sq_ft_altered <= 3100,6,7)))))
    )))
    .pipe(lambda x:x.assign(monthly_rent = x.monthly_rent.astype(float)))
)
```


```python
model_data = (
    aggregate_df_processed_final
    .pipe(lambda x:x.assign(m1 = np.where(pd.to_datetime(x.lease_month).dt.month == 1,1,0)))
    .pipe(lambda x:x.assign(m2 = np.where(pd.to_datetime(x.lease_month).dt.month == 2,1,0)))
    .pipe(lambda x:x.assign(m3 = np.where(pd.to_datetime(x.lease_month).dt.month == 3,1,0)))
    .pipe(lambda x:x.assign(m4 = np.where(pd.to_datetime(x.lease_month).dt.month == 4,1,0)))
    .pipe(lambda x:x.assign(m5 = np.where(pd.to_datetime(x.lease_month).dt.month == 5,1,0)))
    .pipe(lambda x:x.assign(m6 = np.where(pd.to_datetime(x.lease_month).dt.month == 6,1,0)))
    .pipe(lambda x:x.assign(m7 = np.where(pd.to_datetime(x.lease_month).dt.month == 7,1,0)))
    .pipe(lambda x:x.assign(m8 = np.where(pd.to_datetime(x.lease_month).dt.month == 8,1,0)))
    .pipe(lambda x:x.assign(m9 = np.where(pd.to_datetime(x.lease_month).dt.month == 9,1,0)))
    .pipe(lambda x:x.assign(m10 = np.where(pd.to_datetime(x.lease_month).dt.month == 10,1,0)))
    .pipe(lambda x:x.assign(m11 = np.where(pd.to_datetime(x.lease_month).dt.month == 11,1,0)))
    .pipe(lambda x:x.assign(m12 = np.where(pd.to_datetime(x.lease_month).dt.month == 12,1,0)))
    .pipe(lambda x:x.assign(min_year = pd.to_datetime(x.lease_month).dt.year.min()))
    .pipe(lambda x:x.assign(time = 12*(pd.to_datetime(x.lease_month).dt.year-x.min_year) + pd.to_datetime(x.lease_month).dt.month))
    .pipe(lambda x:x.assign(time_time = x.time * x.time))
    .pipe(lambda x:x.assign(time_time_time = x.time * x.time * x.time))
)
```

### Decision if mean or median of monthly rental should be used for predicted variable
- predicted variable = dependent variable = variable we are trying to predict = monthly rent
- limiting to 3 bedrooms or lesser


```python
decision_mean_median = (
    model_data
    .pipe(lambda x:x.assign(monthly_rent_copy = x.monthly_rent))
    [['lease_month','district','num_bedrooms_altered','monthly_rent','monthly_rent_copy']]
    .groupby(['lease_month','district','num_bedrooms_altered'])
    .agg({"monthly_rent":"mean","monthly_rent_copy":"median"})
    .reset_index()
    .pipe(lambda x:x.assign(difference_between_mean_median = x.monthly_rent_copy - x.monthly_rent))
    .query("num_bedrooms_altered <= 3")
)

decision_mean_median.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lease_month</th>
      <th>district</th>
      <th>num_bedrooms_altered</th>
      <th>monthly_rent</th>
      <th>monthly_rent_copy</th>
      <th>difference_between_mean_median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-01</td>
      <td>1</td>
      <td>1</td>
      <td>2975.000000</td>
      <td>3250.0</td>
      <td>275.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-01</td>
      <td>1</td>
      <td>2</td>
      <td>4463.888889</td>
      <td>4500.0</td>
      <td>36.111111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-01</td>
      <td>1</td>
      <td>3</td>
      <td>5838.000000</td>
      <td>5950.0</td>
      <td>112.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-01-01</td>
      <td>2</td>
      <td>1</td>
      <td>3277.500000</td>
      <td>3325.0</td>
      <td>47.500000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016-01-01</td>
      <td>2</td>
      <td>2</td>
      <td>4193.000000</td>
      <td>4200.0</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
## sub plot of difference between mean and median for all districts
for i in decision_mean_median.district.unique():
    
    onebr = decision_mean_median.query("num_bedrooms_altered == 1 & district == " + str(i))
    twobr = decision_mean_median.query("num_bedrooms_altered == 2 & district == " + str(i))
    threebr = decision_mean_median.query("num_bedrooms_altered == 3 & district == " + str(i))
    
    plt.figure(figsize=(20,5))
    plt.subplot(1, 3, 1)
    plt.plot(onebr.lease_month,onebr.monthly_rent,label='mean')
    plt.plot(onebr.lease_month,onebr.monthly_rent_copy,label='median')
    plt.xticks(rotation=45)
    plt.title("District " + str(i) + ": Mean and median monthly rent, 1BR")
    plt.xlabel("date")
    plt.ylabel("Amount (SGD)")
    plt.legend(loc='best')
    
    plt.subplot(1, 3, 2)
    plt.plot(twobr.lease_month,twobr.monthly_rent,label='mean')
    plt.plot(twobr.lease_month,twobr.monthly_rent_copy,label='median')
    plt.xticks(rotation=45)
    plt.title("District " + str(i) + ": Mean and median monthly rent, 2BR")
    plt.xlabel("date")
    plt.ylabel("Amount (SGD)")
    plt.legend(loc='best')
    
    plt.subplot(1, 3, 3)
    plt.plot(threebr.lease_month,threebr.monthly_rent,label='mean')
    plt.plot(threebr.lease_month,threebr.monthly_rent_copy,label='median')
    plt.xticks(rotation=45)
    plt.title("District " + str(i) + ": Mean and median monthly rent, 3BR")
    plt.xlabel("date")
    plt.ylabel("Amount (SGD)")
    plt.legend(loc='best')
    
```


![png](property-rental-price_files/property-rental-price_19_0.png)



![png](property-rental-price_files/property-rental-price_19_1.png)



![png](property-rental-price_files/property-rental-price_19_2.png)



![png](property-rental-price_files/property-rental-price_19_3.png)



![png](property-rental-price_files/property-rental-price_19_4.png)



![png](property-rental-price_files/property-rental-price_19_5.png)



![png](property-rental-price_files/property-rental-price_19_6.png)



![png](property-rental-price_files/property-rental-price_19_7.png)



![png](property-rental-price_files/property-rental-price_19_8.png)



![png](property-rental-price_files/property-rental-price_19_9.png)



![png](property-rental-price_files/property-rental-price_19_10.png)



![png](property-rental-price_files/property-rental-price_19_11.png)



![png](property-rental-price_files/property-rental-price_19_12.png)



![png](property-rental-price_files/property-rental-price_19_13.png)



![png](property-rental-price_files/property-rental-price_19_14.png)



![png](property-rental-price_files/property-rental-price_19_15.png)



![png](property-rental-price_files/property-rental-price_19_16.png)



![png](property-rental-price_files/property-rental-price_19_17.png)



![png](property-rental-price_files/property-rental-price_19_18.png)



![png](property-rental-price_files/property-rental-price_19_19.png)



![png](property-rental-price_files/property-rental-price_19_20.png)



![png](property-rental-price_files/property-rental-price_19_21.png)



![png](property-rental-price_files/property-rental-price_19_22.png)



![png](property-rental-price_files/property-rental-price_19_23.png)



![png](property-rental-price_files/property-rental-price_19_24.png)



![png](property-rental-price_files/property-rental-price_19_25.png)



![png](property-rental-price_files/property-rental-price_19_26.png)


#### The difference between mean and median isnt that much, will be going with mean


```python
average_model_data = (
    model_data
    .groupby(['lease_month','district','num_bedrooms_altered','time','time_time','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12'])
    .agg({"monthly_rent":"mean","sq_ft_altered":"count"})
    .reset_index()
    .rename(columns={"sq_ft_altered":"num_data_points"})
)
```


```python
average_model_data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lease_month</th>
      <th>district</th>
      <th>num_bedrooms_altered</th>
      <th>time</th>
      <th>time_time</th>
      <th>m1</th>
      <th>m2</th>
      <th>m3</th>
      <th>m4</th>
      <th>m5</th>
      <th>m6</th>
      <th>m7</th>
      <th>m8</th>
      <th>m9</th>
      <th>m10</th>
      <th>m11</th>
      <th>m12</th>
      <th>monthly_rent</th>
      <th>num_data_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4921</th>
      <td>2018-12-01</td>
      <td>27</td>
      <td>4</td>
      <td>36</td>
      <td>1296</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2975.000000</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4922</th>
      <td>2018-12-01</td>
      <td>28</td>
      <td>1</td>
      <td>36</td>
      <td>1296</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1744.444444</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4923</th>
      <td>2018-12-01</td>
      <td>28</td>
      <td>2</td>
      <td>36</td>
      <td>1296</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1939.562500</td>
      <td>32</td>
    </tr>
    <tr>
      <th>4924</th>
      <td>2018-12-01</td>
      <td>28</td>
      <td>3</td>
      <td>36</td>
      <td>1296</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2206.000000</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4925</th>
      <td>2018-12-01</td>
      <td>28</td>
      <td>4</td>
      <td>36</td>
      <td>1296</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2733.333333</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## Forecast rental amount of each month for each district

### Forecast each month, the rental lease for each district
- functionalize for each bedroom type
- districts that have insufficient data will be ignored


```python
def get_each_district_cheapest_month(df, num_BR):
    unique_num_district = df.district.unique()
    summary_model_district_df = None
    aggregate_prediction_df = None
    for i in unique_num_district:
#         X = df.query("district == " + str(i) + " & num_bedrooms_altered == " + str(num_BR))[['time','time_time','time_time_time','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']]
        X = df.query("district == " + str(i) + " & num_bedrooms_altered == " + str(num_BR))[['time','time_time','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']]
        X = sm.add_constant(X)
        y = df.query("district == " + str(i) + " & num_bedrooms_altered == " + str(num_BR))[['monthly_rent']]
        model_rent = sm.OLS(y,X).fit()
        
        testing = pd.DataFrame(model_rent.pvalues).rename(columns={0:"district_"+str(i)+"_pvalues"}).T
        testing_ = pd.DataFrame(model_rent.params).rename(columns={0:"district_"+str(i)+"_params"}).T
        testing_2 = (
            testing
            .append(testing_)
            .pipe(lambda x:x.assign(r_adj = model_rent.rsquared_adj))
            .pipe(lambda x:x.assign(r_square = model_rent.rsquared))
        )
        if summary_model_district_df is None:
            summary_model_district_df = testing_2
        else:
            summary_model_district_df = summary_model_district_df.append(testing_2)
        try:
            
            prediction_df = (
                pd.DataFrame(model_rent.predict(X))
                .reset_index()
                .merge(y.reset_index(),how='inner',on=['index'])
                .rename(columns={0:"prediction"})
                .merge(average_model_data.reset_index()[['lease_month','index','time']],how='inner',on='index')
                .pipe(lambda x:x.assign(trend = model_rent.params.const +model_rent.params.time * x.time +model_rent.params.time_time * x.time*x.time))
#                 .pipe(lambda x:x.assign(seasonal = x.prediction - model_rent.params.time * x.time - model_rent.params.time_time*x.time*x.time - model_rent.params.time_time_time*x.time*x.time*x.time))
                .pipe(lambda x:x.assign(seasonal = x.prediction - model_rent.params.time * x.time - model_rent.params.time_time*x.time*x.time))
                .pipe(lambda x:x.assign(month = pd.to_datetime(x.lease_month).dt.month))    
                .pipe(lambda x:x.assign(district = i))
            )
            
            
            if aggregate_prediction_df is None:
                aggregate_prediction_df = prediction_df
            elif len(prediction_df) < 35:
                pass
            else:
                aggregate_prediction_df = aggregate_prediction_df.append(prediction_df)
        except:
            pass
        
        # check if seasonality exists
        
    
    return summary_model_district_df.dropna(), aggregate_prediction_df
```


```python
one_bedroom_figures = get_each_district_cheapest_month(average_model_data,1)
two_bedroom_figures = get_each_district_cheapest_month(average_model_data,2)
three_bedroom_figures = get_each_district_cheapest_month(average_model_data,3)
```

### Fit of linear regression models


```python
# fit of models
(
    one_bedroom_figures[1]
    .pipe(lambda x:x.assign(error_value = x.prediction-x.monthly_rent))
    .pipe(lambda x:x.assign(abs_error = abs(x.error_value)))
    .pipe(lambda x:x.assign(error_square = x.error_value * x.error_value))
    .groupby(['district'])
    .agg({"monthly_rent":"mean","month":"count","abs_error":"sum","error_square":"sum"})
    .reset_index()
    .pipe(lambda x:x.assign(mae = x.abs_error/x.month))
    .pipe(lambda x:x.assign(rmse = np.sqrt(x.error_square/x.month)))
    .pipe(lambda x:x.assign(mae_percentage_error = x.mae/x.monthly_rent))
    .pipe(lambda x:x.assign(rmse_percentage_error = x.rmse/x.monthly_rent))
    [['district','monthly_rent','mae','rmse','mae_percentage_error','rmse_percentage_error']]
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>district</th>
      <th>monthly_rent</th>
      <th>mae</th>
      <th>rmse</th>
      <th>mae_percentage_error</th>
      <th>rmse_percentage_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2964.733474</td>
      <td>81.035068</td>
      <td>98.002619</td>
      <td>0.027333</td>
      <td>0.033056</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2972.427251</td>
      <td>71.236593</td>
      <td>95.322122</td>
      <td>0.023966</td>
      <td>0.032069</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2487.578950</td>
      <td>53.276117</td>
      <td>71.748285</td>
      <td>0.021417</td>
      <td>0.028843</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2592.398396</td>
      <td>63.330893</td>
      <td>76.957875</td>
      <td>0.024429</td>
      <td>0.029686</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2272.416375</td>
      <td>34.529162</td>
      <td>44.288542</td>
      <td>0.015195</td>
      <td>0.019490</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>2834.889814</td>
      <td>184.729447</td>
      <td>219.394078</td>
      <td>0.065163</td>
      <td>0.077391</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>2240.632143</td>
      <td>41.094756</td>
      <td>50.292277</td>
      <td>0.018341</td>
      <td>0.022446</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>2840.559893</td>
      <td>28.227508</td>
      <td>39.426857</td>
      <td>0.009937</td>
      <td>0.013880</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>2501.706644</td>
      <td>28.338143</td>
      <td>36.794664</td>
      <td>0.011328</td>
      <td>0.014708</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>2525.229745</td>
      <td>54.736614</td>
      <td>64.814195</td>
      <td>0.021676</td>
      <td>0.025667</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12</td>
      <td>2020.552452</td>
      <td>19.205315</td>
      <td>23.806076</td>
      <td>0.009505</td>
      <td>0.011782</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13</td>
      <td>2038.372341</td>
      <td>35.863539</td>
      <td>42.003045</td>
      <td>0.017594</td>
      <td>0.020606</td>
    </tr>
    <tr>
      <th>12</th>
      <td>14</td>
      <td>1953.023056</td>
      <td>15.953913</td>
      <td>19.200223</td>
      <td>0.008169</td>
      <td>0.009831</td>
    </tr>
    <tr>
      <th>13</th>
      <td>15</td>
      <td>1989.043066</td>
      <td>32.071578</td>
      <td>38.418785</td>
      <td>0.016124</td>
      <td>0.019315</td>
    </tr>
    <tr>
      <th>14</th>
      <td>16</td>
      <td>2062.597470</td>
      <td>28.878133</td>
      <td>34.983980</td>
      <td>0.014001</td>
      <td>0.016961</td>
    </tr>
    <tr>
      <th>15</th>
      <td>17</td>
      <td>1649.086793</td>
      <td>18.500662</td>
      <td>25.057818</td>
      <td>0.011219</td>
      <td>0.015195</td>
    </tr>
    <tr>
      <th>16</th>
      <td>18</td>
      <td>1818.609337</td>
      <td>26.542009</td>
      <td>33.858950</td>
      <td>0.014595</td>
      <td>0.018618</td>
    </tr>
    <tr>
      <th>17</th>
      <td>19</td>
      <td>1871.928770</td>
      <td>19.193934</td>
      <td>22.835074</td>
      <td>0.010254</td>
      <td>0.012199</td>
    </tr>
    <tr>
      <th>18</th>
      <td>20</td>
      <td>2073.834896</td>
      <td>40.418322</td>
      <td>52.443098</td>
      <td>0.019490</td>
      <td>0.025288</td>
    </tr>
    <tr>
      <th>19</th>
      <td>21</td>
      <td>1934.097368</td>
      <td>55.965134</td>
      <td>65.949053</td>
      <td>0.028936</td>
      <td>0.034098</td>
    </tr>
    <tr>
      <th>20</th>
      <td>22</td>
      <td>2348.065641</td>
      <td>73.015941</td>
      <td>84.140054</td>
      <td>0.031096</td>
      <td>0.035834</td>
    </tr>
    <tr>
      <th>21</th>
      <td>23</td>
      <td>1873.124877</td>
      <td>19.658881</td>
      <td>25.197711</td>
      <td>0.010495</td>
      <td>0.013452</td>
    </tr>
    <tr>
      <th>22</th>
      <td>25</td>
      <td>1634.666331</td>
      <td>23.767915</td>
      <td>32.989465</td>
      <td>0.014540</td>
      <td>0.020181</td>
    </tr>
    <tr>
      <th>23</th>
      <td>27</td>
      <td>1682.525041</td>
      <td>17.960520</td>
      <td>21.780297</td>
      <td>0.010675</td>
      <td>0.012945</td>
    </tr>
    <tr>
      <th>24</th>
      <td>28</td>
      <td>1786.214175</td>
      <td>46.266333</td>
      <td>55.142541</td>
      <td>0.025902</td>
      <td>0.030871</td>
    </tr>
  </tbody>
</table>
</div>



### Based on the forecast, find out which is the lowest (and highest) month. And the savings from each district
- get top 5 saving districts


```python
def get_savings_for_each_district(df):
    unique_districts = df.district.unique()
    aggregate_savings_df = None
    for i in unique_districts:
        current_district_df = (
            df.query("district == " + str(i))
            .groupby(['month'])
            .agg({"seasonal":"mean"})
            .reset_index()
        )
        
        highest_month = current_district_df.seasonal.idxmax() + 1
        highest_month_rental_df = (
            current_district_df
            .query("month == " + str(highest_month))
            .pipe(lambda x:x.assign(district = i))
            .rename(columns={"month":"highest_month","seasonal":"highest_month_rent"})
        )
        lowest_month = (current_district_df.seasonal.idxmin()) + 1
        lowest_month_rental_df = (
            current_district_df
            .query("month == " + str(lowest_month))
            .pipe(lambda x:x.assign(district = i))
            .rename(columns={"month":"lowest_month","seasonal":"lowest_month_rent"})
        )
        savings_district_df = (
            highest_month_rental_df
            .merge(lowest_month_rental_df,how='left',on=['district'])
            .pipe(lambda x:x.assign(savings_per_year = round(12 * (x.highest_month_rent - x.lowest_month_rent),2)))
            [['district','savings_per_year','highest_month','highest_month_rent','lowest_month','lowest_month_rent']]
        )
        
        if aggregate_savings_df is None:
            aggregate_savings_df = savings_district_df
        else:
            aggregate_savings_df = aggregate_savings_df.append(savings_district_df)
    return aggregate_savings_df
        
```

### To change to 2BR or 3BR change the second line of code to:
- `get_savings_for_each_district(two_bedroom_figures[1])`
- `get_savings_for_each_district(three_bedroom_figures[1])`


```python
savings_for_each_district_df = (
    get_savings_for_each_district(one_bedroom_figures[1])
    .sort_values(['savings_per_year'],ascending=False)
)
```

### Show which areas is in each district


```python
district_areas = pd.read_csv('../data/district_area.csv')
savings_for_each_district_with_location = (
    savings_for_each_district_df
    .merge(district_areas,how='left',on=['district'])
)

savings_for_each_district_with_location.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>district</th>
      <th>savings_per_year</th>
      <th>highest_month</th>
      <th>highest_month_rent</th>
      <th>lowest_month</th>
      <th>lowest_month_rent</th>
      <th>areas</th>
      <th>location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>9196.77</td>
      <td>2</td>
      <td>3093.789321</td>
      <td>11</td>
      <td>2327.391896</td>
      <td>Beach Road</td>
      <td>Beach Road, Bencoolen Road, Bugis, Rochor</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4816.43</td>
      <td>4</td>
      <td>3476.205617</td>
      <td>6</td>
      <td>3074.836509</td>
      <td>Marina Area</td>
      <td>Boat Quay, Chinatown, Havelock Road, Marina Sq...</td>
    </tr>
  </tbody>
</table>
</div>



### Predict the next 24 months of rental for each district

### Ensemble of both methods, holt winter and fb-prophet


```python
def predict_next_24_months_rental(df):
    unique_districts_available = df.district.unique()
    aggregate_predict_next_24months = None
    for i in unique_districts_available:
        current_district_df = (
            df
            .query("district == " + str(i))
            [['lease_month','monthly_rent']]
            .rename(columns={"lease_month":"ds",'monthly_rent':'y'})
        )
        
        # prophet forecast
        m = Prophet()
        m.fit(current_district_df)
        future = m.make_future_dataframe(periods=24,freq='m')
        forecast = (
            m.predict(future)
            [['ds','yhat']]
            .pipe(lambda x:x.assign(ds = np.where(x.ds > '2018-12-01',x.ds+timedelta(days=1),x.ds)))
        )
        
        #holt winters forecast
        fit1 = ExponentialSmoothing(np.asarray(current_district_df['y']) ,seasonal_periods=12,trend='add', seasonal='add',).fit()
        hw_forecast = (
            pd.DataFrame(fit1.forecast(24))
            .pipe(lambda x:x.assign(ds = forecast.query("ds > '2018-12-01'").ds.values))
        )
        hw_fit = (
            pd.DataFrame(fit1.fittedvalues)
            .pipe(lambda x:x.assign(ds = forecast.query("ds <= '2018-12-01'").ds.values))
        )
        
        hw_forecast_fit = (
            hw_fit
            .append(hw_forecast)
            .rename(columns={0:"holtwinter"})
        )
        
        prophet_hw_combine = (
            forecast
            .merge(hw_forecast_fit,how='left',on=['ds'])
            .merge(current_district_df,how='left',on=['ds'])
            .fillna(0)
            .rename(columns={"yhat":"fbprophet","y":"actual_monthly_rent"})
        )
        
        X = prophet_hw_combine.query("ds <= '2018-12-01'")[['fbprophet','holtwinter']]
        y = prophet_hw_combine.query("ds <= '2018-12-01'")[['actual_monthly_rent']]
        weight_model = sm.OLS(y,X).fit()
        
        prophet_hw_combine = (
            prophet_hw_combine
            .pipe(lambda x:x.assign(ensemble_model_output = weight_model.params.fbprophet * x.fbprophet + weight_model.params.holtwinter * x.holtwinter))
            .pipe(lambda x:x.assign(district = i.astype(int)))
        )
        
        if aggregate_predict_next_24months is None:
            aggregate_predict_next_24months = prophet_hw_combine
        else:
            aggregate_predict_next_24months = aggregate_predict_next_24months.append(prophet_hw_combine)
        
        
#         fig2 = m.plot_components(forecast)
    return aggregate_predict_next_24months
        
```


```python
predict_future_values = predict_next_24_months_rental(one_bedroom_figures[1])
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.



```python
predict_future_values.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>fbprophet</th>
      <th>holtwinter</th>
      <th>actual_monthly_rent</th>
      <th>ensemble_model_output</th>
      <th>district</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-01</td>
      <td>3030.146058</td>
      <td>3070.507644</td>
      <td>2975.000000</td>
      <td>3029.157251</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-02-01</td>
      <td>3151.967873</td>
      <td>3191.192423</td>
      <td>3121.428571</td>
      <td>3151.013532</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-01</td>
      <td>3204.835136</td>
      <td>3382.326106</td>
      <td>3357.777778</td>
      <td>3200.163935</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-04-01</td>
      <td>3357.631160</td>
      <td>3466.171233</td>
      <td>3414.062500</td>
      <td>3354.819204</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-05-01</td>
      <td>3173.818509</td>
      <td>3272.966622</td>
      <td>3244.117647</td>
      <td>3171.253277</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Fit of time-series models


```python
(
    predict_future_values
    .query("actual_monthly_rent > 0")
    .pipe(lambda x:x.assign(error_value = x.holtwinter-x.actual_monthly_rent))
    .pipe(lambda x:x.assign(abs_error = abs(x.error_value)))
    .pipe(lambda x:x.assign(error_square = x.error_value * x.error_value))
    .groupby(['district'])
    .agg({"actual_monthly_rent":"mean","ds":"count","abs_error":"sum","error_square":"sum"})
    .reset_index()
    .pipe(lambda x:x.assign(mae = x.abs_error/x.ds))
    .pipe(lambda x:x.assign(rmse = np.sqrt(x.error_square/x.ds)))
    .pipe(lambda x:x.assign(mae_percentage_error = x.mae/x.actual_monthly_rent))
    .pipe(lambda x:x.assign(rmse_percentage_error = x.rmse/x.actual_monthly_rent))
    [['district','actual_monthly_rent','mae','rmse','mae_percentage_error','rmse_percentage_error']]
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>district</th>
      <th>actual_monthly_rent</th>
      <th>mae</th>
      <th>rmse</th>
      <th>mae_percentage_error</th>
      <th>rmse_percentage_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2964.733474</td>
      <td>105.656901</td>
      <td>142.207175</td>
      <td>0.035638</td>
      <td>0.047966</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2972.427251</td>
      <td>84.583398</td>
      <td>104.071358</td>
      <td>0.028456</td>
      <td>0.035012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2487.578950</td>
      <td>66.044601</td>
      <td>103.426482</td>
      <td>0.026550</td>
      <td>0.041577</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2592.398396</td>
      <td>68.562218</td>
      <td>105.906033</td>
      <td>0.026447</td>
      <td>0.040853</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2272.416375</td>
      <td>51.523411</td>
      <td>79.893517</td>
      <td>0.022673</td>
      <td>0.035158</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>2834.889814</td>
      <td>182.835988</td>
      <td>276.429089</td>
      <td>0.064495</td>
      <td>0.097510</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>2240.632143</td>
      <td>62.462414</td>
      <td>89.218392</td>
      <td>0.027877</td>
      <td>0.039818</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>2840.559893</td>
      <td>39.634566</td>
      <td>53.364981</td>
      <td>0.013953</td>
      <td>0.018787</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>2501.706644</td>
      <td>41.889356</td>
      <td>65.392707</td>
      <td>0.016744</td>
      <td>0.026139</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>2525.229745</td>
      <td>75.576729</td>
      <td>103.811234</td>
      <td>0.029929</td>
      <td>0.041110</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12</td>
      <td>2020.552452</td>
      <td>38.517213</td>
      <td>61.356149</td>
      <td>0.019063</td>
      <td>0.030366</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13</td>
      <td>2038.372341</td>
      <td>38.328170</td>
      <td>46.355652</td>
      <td>0.018803</td>
      <td>0.022742</td>
    </tr>
    <tr>
      <th>12</th>
      <td>14</td>
      <td>1953.023056</td>
      <td>23.157154</td>
      <td>31.563201</td>
      <td>0.011857</td>
      <td>0.016161</td>
    </tr>
    <tr>
      <th>13</th>
      <td>15</td>
      <td>1989.043066</td>
      <td>46.067313</td>
      <td>76.218856</td>
      <td>0.023161</td>
      <td>0.038319</td>
    </tr>
    <tr>
      <th>14</th>
      <td>16</td>
      <td>2062.597470</td>
      <td>34.232017</td>
      <td>52.638489</td>
      <td>0.016597</td>
      <td>0.025520</td>
    </tr>
    <tr>
      <th>15</th>
      <td>17</td>
      <td>1649.086793</td>
      <td>30.387994</td>
      <td>48.041040</td>
      <td>0.018427</td>
      <td>0.029132</td>
    </tr>
    <tr>
      <th>16</th>
      <td>18</td>
      <td>1818.609337</td>
      <td>35.655841</td>
      <td>55.569676</td>
      <td>0.019606</td>
      <td>0.030556</td>
    </tr>
    <tr>
      <th>17</th>
      <td>19</td>
      <td>1871.928770</td>
      <td>33.866030</td>
      <td>58.789820</td>
      <td>0.018092</td>
      <td>0.031406</td>
    </tr>
    <tr>
      <th>18</th>
      <td>20</td>
      <td>2073.834896</td>
      <td>48.125850</td>
      <td>81.128471</td>
      <td>0.023206</td>
      <td>0.039120</td>
    </tr>
    <tr>
      <th>19</th>
      <td>21</td>
      <td>1934.097368</td>
      <td>65.007626</td>
      <td>95.845440</td>
      <td>0.033611</td>
      <td>0.049556</td>
    </tr>
    <tr>
      <th>20</th>
      <td>22</td>
      <td>2348.065641</td>
      <td>53.534707</td>
      <td>95.675112</td>
      <td>0.022799</td>
      <td>0.040746</td>
    </tr>
    <tr>
      <th>21</th>
      <td>23</td>
      <td>1873.124877</td>
      <td>23.045726</td>
      <td>27.414015</td>
      <td>0.012303</td>
      <td>0.014635</td>
    </tr>
    <tr>
      <th>22</th>
      <td>25</td>
      <td>1634.666331</td>
      <td>31.866991</td>
      <td>52.742639</td>
      <td>0.019494</td>
      <td>0.032265</td>
    </tr>
    <tr>
      <th>23</th>
      <td>27</td>
      <td>1682.525041</td>
      <td>18.417836</td>
      <td>28.866445</td>
      <td>0.010947</td>
      <td>0.017157</td>
    </tr>
    <tr>
      <th>24</th>
      <td>28</td>
      <td>1786.214175</td>
      <td>51.339639</td>
      <td>82.280405</td>
      <td>0.028742</td>
      <td>0.046064</td>
    </tr>
  </tbody>
</table>
</div>



### combine both lowest month and duration
- assuming you rented in the lowest month, looking ahead in the next year, how much will you save?


```python
intermediate_df = (
    predict_future_values
    .pipe(lambda x:x.assign(previous_year = x.ds + timedelta(days=-365)))
    .pipe(lambda x:x.assign(previous_year = np.where(x.previous_year.dt.day==2,x.previous_year+timedelta(days=-1),x.previous_year)))
    .merge(savings_for_each_district_with_location[['district','lowest_month']],how='left',on=['district'])
)

savings_for_longer_lease = (
    intermediate_df
    .merge(intermediate_df[['ds','actual_monthly_rent','district']],how='inner',left_on=['previous_year','district'],right_on=['ds','district'])
    .pipe(lambda x:x.assign(lowest_month_date = pd.to_datetime(str("2018-")+(x.lowest_month.astype(str))+"-01")))
    .query("ds_y == lowest_month_date")
    .drop(['ds_y'],1)
    .rename(columns={"ds_x":"ds","actual_monthly_rent_x":"actual_monthly_rent","actual_monthly_rent_y":"previous_year_rental"})
    .pipe(lambda x:x.assign(savings_24months_lease = 12 * (x.holtwinter-x.previous_year_rental)))
    .pipe(lambda x:x.assign(should_extend_lease = np.where(x.savings_24months_lease>0,1,0)))
)

total_savings_for_longer_lease_lowest_months = (
    savings_for_longer_lease
    .merge(savings_for_each_district_with_location[['savings_per_year','district','location','highest_month']],how='left',on=['district'])
    .pipe(lambda x:x.assign(total_savings_per_year = x.should_extend_lease * x.savings_24months_lease + x.savings_per_year))
    .drop(['actual_monthly_rent'],1)
    .sort_values(['total_savings_per_year'],ascending=False)
)
```


```python
total_savings_for_longer_lease_lowest_months[['district','location','lowest_month','highest_month','savings_per_year','savings_24months_lease','should_extend_lease','total_savings_per_year']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>district</th>
      <th>location</th>
      <th>lowest_month</th>
      <th>highest_month</th>
      <th>savings_per_year</th>
      <th>savings_24months_lease</th>
      <th>should_extend_lease</th>
      <th>total_savings_per_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>Beach Road, Bencoolen Road, Bugis, Rochor</td>
      <td>11</td>
      <td>2</td>
      <td>9196.77</td>
      <td>3373.822052</td>
      <td>1</td>
      <td>12570.592052</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Boat Quay, Chinatown, Havelock Road, Marina Sq...</td>
      <td>6</td>
      <td>4</td>
      <td>4816.43</td>
      <td>-472.783183</td>
      <td>0</td>
      <td>4816.430000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Anson Road, Chinatown, Neil Road, Raffles Plac...</td>
      <td>3</td>
      <td>11</td>
      <td>4324.38</td>
      <td>-2700.657228</td>
      <td>0</td>
      <td>4324.380000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>21</td>
      <td>Clementi, Upper Bukit Timah, Hume Avenue</td>
      <td>1</td>
      <td>8</td>
      <td>2487.63</td>
      <td>1435.272186</td>
      <td>1</td>
      <td>3922.902186</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>Cairnhill, Killiney, Leonie Hill, Orchard, Oxley</td>
      <td>12</td>
      <td>4</td>
      <td>3330.82</td>
      <td>-743.563908</td>
      <td>0</td>
      <td>3330.820000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Keppel, Mount Faber, Sentosa, Telok Blangah</td>
      <td>6</td>
      <td>2</td>
      <td>2856.02</td>
      <td>219.751631</td>
      <td>1</td>
      <td>3075.771631</td>
    </tr>
    <tr>
      <th>20</th>
      <td>22</td>
      <td>Boon Lay, Jurong, Tuas</td>
      <td>7</td>
      <td>1</td>
      <td>2739.10</td>
      <td>166.818600</td>
      <td>1</td>
      <td>2905.918600</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>Chancery, Bukit Timah, Dunearn Road, Newton</td>
      <td>3</td>
      <td>10</td>
      <td>2621.73</td>
      <td>-889.598223</td>
      <td>0</td>
      <td>2621.730000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>20</td>
      <td>Ang Mo Kio, Bishan, Braddell Road, Thomson</td>
      <td>8</td>
      <td>11</td>
      <td>2078.15</td>
      <td>433.117750</td>
      <td>1</td>
      <td>2511.267750</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Alexandra Road, Tiong Bahru, Queenstown</td>
      <td>1</td>
      <td>3</td>
      <td>2086.49</td>
      <td>-66.447290</td>
      <td>0</td>
      <td>2086.490000</td>
    </tr>
  </tbody>
</table>
</div>




```python
districts_to_focus = total_savings_for_longer_lease_lowest_months.head(5).district.unique()
```

#### Districts to focus are 7, 1, 21 ,2, 3, 4, 9, 22, 11, 20

### Plot out top 10 districts
- plot out prediction, trend
- plot out seasonal on the side


```python
(
    total_savings_for_longer_lease_lowest_months
    .head()
    [['district','location','lowest_month','highest_month','savings_24months_lease','should_extend_lease','savings_per_year','total_savings_per_year']]
    .rename(columns={"savings_per_year":"savings_per_year_from_renting_lowest_month","should_extend_lease":"should_sign_24_months_lease"})
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>district</th>
      <th>location</th>
      <th>lowest_month</th>
      <th>highest_month</th>
      <th>savings_24months_lease</th>
      <th>should_sign_24_months_lease</th>
      <th>savings_per_year_from_renting_lowest_month</th>
      <th>total_savings_per_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>Beach Road, Bencoolen Road, Bugis, Rochor</td>
      <td>11</td>
      <td>2</td>
      <td>3373.822052</td>
      <td>1</td>
      <td>9196.77</td>
      <td>12570.592052</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Boat Quay, Chinatown, Havelock Road, Marina Sq...</td>
      <td>6</td>
      <td>4</td>
      <td>-506.960948</td>
      <td>0</td>
      <td>4816.43</td>
      <td>4816.430000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Anson Road, Chinatown, Neil Road, Raffles Plac...</td>
      <td>3</td>
      <td>11</td>
      <td>-2700.657228</td>
      <td>0</td>
      <td>4324.38</td>
      <td>4324.380000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>21</td>
      <td>Clementi, Upper Bukit Timah, Hume Avenue</td>
      <td>1</td>
      <td>8</td>
      <td>1435.272186</td>
      <td>1</td>
      <td>2487.63</td>
      <td>3922.902186</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>Cairnhill, Killiney, Leonie Hill, Orchard, Oxley</td>
      <td>12</td>
      <td>4</td>
      <td>-628.058165</td>
      <td>0</td>
      <td>3330.82</td>
      <td>3330.820000</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in districts_to_focus:
    focus_plot_df = (
        one_bedroom_figures[1]
        .query("district == " + str(i))
        [['lease_month','month','district','prediction','monthly_rent','trend','seasonal']]
    )
    
    focus_predict_plot_df = (
        predict_future_values
        .query("district == " + str(i))
        [['ds','actual_monthly_rent','ensemble_model_output','holtwinter']]
    )
    
    lowest_month = savings_for_each_district_with_location.query("district == " + str(i)).lowest_month.values[0]
    lowest_month_rent = savings_for_each_district_with_location.query("district == " + str(i)).lowest_month_rent.values[0]
    highest_month = savings_for_each_district_with_location.query("district == " + str(i)).highest_month.values[0]
    highest_month_rent = savings_for_each_district_with_location.query("district == " + str(i)).highest_month_rent.values[0]
    
    plt.figure(figsize=(20,5))
    plt.subplot(1, 2, 1)
    plt.plot(focus_plot_df.lease_month,focus_plot_df.prediction,label='linear regression model')
    plt.plot(focus_plot_df.lease_month,focus_plot_df.monthly_rent,label='actual rent')
    plt.plot(focus_plot_df.lease_month,focus_plot_df.trend,label='trend')
    plt.plot(focus_predict_plot_df.query("ds > '2018-12-01'").ds,focus_predict_plot_df.query("ds > '2018-12-01'").holtwinter,label='predict next 24 months')
    plt.xticks(rotation=45)
    plt.title("District " + str(i) + ": Actual monthly rent against predicted\n(and trend of actual monthly rent)")
    plt.xlabel("date\n fig(a)")
    plt.ylabel("Amount (SGD)")
    plt.legend(loc='best')
    
    plt.subplot(1, 2, 2)
    plt.bar(focus_plot_df.month,focus_plot_df.seasonal,label='seasonal')
    plt.title("District " + str(i) + ": Seasonal months for monthly rent (of when you start the leasing contract)")
    plt.text(lowest_month-0.5,lowest_month_rent+10,"lowest\nmonth,"+str(lowest_month),color='red',fontsize='12')
    plt.text(highest_month-0.5,highest_month_rent+10,"highest\nmonth,"+str(highest_month),color='red',fontsize='12')
    plt.xlabel("month\n fig(b)")
    plt.ylim(bottom=lowest_month_rent-200)
    plt.ylim(top=highest_month_rent+100)
    plt.ylabel("Amount (SGD)")
    
    plt.tight_layout()
```


![png](property-rental-price_files/property-rental-price_48_0.png)



![png](property-rental-price_files/property-rental-price_48_1.png)



![png](property-rental-price_files/property-rental-price_48_2.png)



![png](property-rental-price_files/property-rental-price_48_3.png)



![png](property-rental-price_files/property-rental-price_48_4.png)


### Conclusion
- Assumptions:
    1. the prediction of the rental amount in the next 24 months is assuming all things stay constant, and there is no new cooling property measure introduced
    1. we did an average over district, and did not look into specific area (example: bukit panjang or tiong bahru). if looked into specific areas, there might be differences between areas.
- Constraints:
    1. the prediction accuracy can be further improved if a better model is applied, however to reduce complexity, 2 simple models were used to

### Next steps
- while this is done for rental lease, an similar exercise will be replicated for property sale
