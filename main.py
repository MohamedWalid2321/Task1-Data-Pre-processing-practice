import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif

np.random.seed(0) #every time you run the code, you’ll get the same random values
data = pd.DataFrame({
    'feature1': np.random.normal(loc=0, scale=1, size=1000),
    # 1000 samples from a normal distribution with mean 0 and stddev 1 
    'feature2': np.random.normal(loc=5, scale=2, size=1000),
    # 1000 samples from a normal distribution with mean 5 and stddev 2
    'categorical_feature': np.random.choice(['Category1', 'Category2', 'Category3'], size=1000),
    'label': np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
    # 1000 samples with 70% of class 0 and 30% of class 1
    # bacuase we are randoming choices of 0 and 1 it printed 718 0's and 282 1's not giving a
    # perfect 70% and 30% split
})

#1) Handle missing values

#1.Introduce some missing values if non exist

data.loc[data.sample(frac=0.05).index, 'feature1'] = np.nan
#data.sample(frac=0.05) randomly selects 5% of the rows in the DataFrame
#.index gets the indices of these rows
data.loc[data.sample(frac=0.05).index, 'feature2'] = np.nan


#2.Handle missing values
imputer = SimpleImputer(strategy='mean')
# We used mean because our numerical data follows a roughly normal distribution
data[['feature1', 'feature2']] = imputer.fit_transform(data[['feature1', 'feature2']])
#then we fill the missing values in 'feature1' and 'feature2' with the mean of their respective columns




#2) Unsampling the minority class


#1.separate 0's and 1's we need to separate the two classes first because 
# we want to upsample the minority with majority
minority=data[data.label==1]
majority=data[data.label==0]
#return a DataFrame containing only the rows where the label is 1 and 0 respectively


#2.upsample the minority class
minority_upsampled=resample(minority #the data to be resampled
                            ,replace=True, 
                            n_samples=len(majority), #see how many samples we need to match the majority class
                            random_state=123) 
#By setting random_state to a specific number (e.g., random_state=42 or random_state=123), you’re telling Python:
# “Use the same random pattern every time I run this code.”
# This makes your results reproducible — meaning you (and others) can get the exact same output when running the code again.

#here is return a new DataFrame where the minority class 
#has been upsampled to match the number of samples in the majority class from 282 to 718


#3.combine the upsampled minority class with the majority class
#since we came up with a new 718 samples for the minority class now we need to merge it with the old 718 of the majority class
data_balanced=pd.concat([majority,minority_upsampled])
#the concat is row-wise by default (axis=0)

#we need to upsample or downsample a data because most ML will biased towards the majority class
#we choosed to upsample rather than downsample because we don't want to lose any data in addition we 
#don't have a very large dataset to begin with

#3) Standardize numerical features
#we use standardization because some ML algorithms might give more importance to features with larger scales
#standardization transforms the data to have a mean of 0 and a standard deviation of 1
#so we cal the mean of each feature and subtract it from each value in that feature and then divide by the standard deviation of that feature
#After standardization, the feature is centered around 0 and its spread is scaled equally
scaler=StandardScaler()

data[['feature1','feature2']]=scaler.fit_transform(data[['feature1','feature2']])

# #4) Normalize numerical features
# Normalize is another scaling tech same as Standardization
# but it scales the data to a fixed range, usually 0 to 1
#both are used in different cases or algorithms

# normalizer=MinMaxScaler()
# data[['feature1','feature2']]=normalizer.fit_transform(data[['feature1','feature2']])

#in this case we will using the standardized data because it is roughly normally distributed
