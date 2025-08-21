#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
data={
    "shear":[2160.70,1680.15,2318.00,2063.30,2209.50,1710.30,1786.70,2577.00,2359.90,2258.70,2167.20,2401.55,1781.80,2338.75,
             1767.30,2055.50,2416.40,2202.50,2656.20,1755.70],
    "age":[15.50,23.75,8.00,17.00,5.50,19.00,24.00,2.50,7.50,11.00,13.00,3.75,25.00,9.75,22.00,18.00,6.00,12.50,2.00,21.50]
}
print(data.items())
df=pd.DataFrame(data)
df.head()
y=data['shear']
x=data['age']
x=sm.add_constant(x)
l_r=sm.OLS(y,x)
f_m=l_r.fit()
f_m.summary()
intercept=f_m.params[0]
slope=f_m.params[1]
print("\nintercept:",intercept)
print("\nslope:",slope)


# In[ ]:




