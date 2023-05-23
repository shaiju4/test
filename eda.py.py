#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data=pd.read_csv('activity_context_tracking_data(1) (1).csv')
data.head()


# no missng data points in dataset
# 

# In[2]:


data.isnull().sum()


# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def check_missing_data(df):
    """
    This function checks for missing data in the dataframe and returns the total count and percentage.
    """
    missing_data = df.isnull().sum().sort_values(ascending=False)
    percent_missing = (missing_data/len(df))*100
    missing_data_df = pd.concat([missing_data, percent_missing], axis=1, keys=['Total', 'Percentage'])
    return missing_data_df

def calculate_descriptive_stats(df):
    """
    This function calculates descriptive statistics for the numerical columns in the dataframe.
    """
    stats = df.describe().transpose()
    stats['skewness'] = df.skew(numeric_only=True)
    stats['kurtosis'] = df.kurtosis(numeric_only=True)

    return stats

def plot_histogram(df, column):
    """
    This function plots a histogram for a given column in the dataframe.
    """
    plt.hist(df[column])
    plt.title('Histogram of {}'.format(column))
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_boxplot(df, column, target):
    """
    This function plots a boxplot for a given column in the dataframe with respect to the target variable.
    """
    sns.boxplot(x=target, y=column, data=df)
    plt.title('Boxplot of {} by {}'.format(column, target))
    plt.show()

def plot_correlation_heatmap(df):
    """
    This function plots a correlation heatmap for the entire dataframe.
    """
    corr = df.corr(numeric_only=True)

    sns.heatmap(corr, cmap="YlGnBu")
    plt.title('Correlation Heatmap')
    plt.show()


def perform_eda(df):
    """
    This function performs exploratory data analysis on the given dataframe.
    """
    # Check for missing data
    missing_data_df = check_missing_data(df)
    print(missing_data_df)

    # Plot histogram
    plot_histogram(df, 'soundLevel')

    # Calculate descriptive statistics
    descriptive_stats = calculate_descriptive_stats(df)
    print(descriptive_stats)


    # Plot boxplot
    plot_boxplot(df, 'activity', 'soundLevel')

    # # Plot correlation heatmap
    plot_correlation_heatmap(df)


df = pd.read_csv('activity_context_tracking_data(1) (1).csv')
perform_eda(df)



# In[21]:


# Plot a bar chart of gender frequency
soundLevel_freq = df['soundLevel'].value_counts()
plt.bar(soundLevel_freq.index, soundLevel_freq.values)
plt.title('soundLevel Frequency')
plt.xlabel('soundLevel')
plt.ylabel('Count')
plt.show()

# Plot a pie chart of gender frequency
plt.pie(soundLevel_freq.values, labels=soundLevel_freq.index, autopct='%1.1f%%')
plt.title('soundLevel Frequency')
plt.show()

# Plot a grouped bar chart of education and gender frequencies
lux_soundLevel_freq = df.groupby(['lux', 'soundLevel']).size().reset_index(name='count')
sns.catplot(x='lux', y='count', hue='soundLevel', data=lux_soundLevel_freq, kind='bar')
plt.title('lux and soundLevel Frequencies')
plt.xlabel('lux')
plt.ylabel('Count')
plt.show()

# Plot a stacked bar chart of age and gender frequencies
activity_soundLevel_freq = df.groupby(['activity', 'soundLevel']).size().reset_index(name='count')
activity_soundLevel_freq =activity_soundLevel_freq.pivot(index='activity', columns='soundLevel', values='count')
activity_soundLevel_freq.plot(kind='bar', stacked=True)
plt.title('activity and soundLevel Frequencies')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('activity_context_tracking_data(1) (1).csv')

# Count the number of instances for each class
class_counts = df['activity'].value_counts()

# Plot the class distribution as a bar chart
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Class Distribution')
plt.show()


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

# Load the dataset into a pandas DataFrame
df = pd.read_csv('activity_context_tracking_data(1) (1).csv')

# Count the number of instances for each class
class_counts = df['activity'].value_counts()

# Plot the class distribution as a bar chart
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('activity')
plt.ylabel('Number of Instances')
plt.title('Class Distribution (Before Oversampling)')
plt.show()

# Perform oversampling
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(df.drop('activity', axis=1), df['activity'])
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='activity')], axis=1)

# Count the number of instances for each class after oversampling
class_counts_resampled = df_resampled['activity'].value_counts()

# Plot the class distribution as a bar chart after oversampling
plt.bar(class_counts_resampled.index, class_counts_resampled.values)
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Class Distribution (After Oversampling)')
plt.show()

