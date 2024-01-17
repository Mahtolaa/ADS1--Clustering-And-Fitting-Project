# -*- coding: utf-8 -*-
"""
Created on Wed Jan 3 23:15:06 2024

@author: Mariam Maliki
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.cluster import KMeans
# Define all functions which will be used in the program


def read(x, y):
    """
    Reads and imports files from excel spreadsheet to a python DataFrame

    Returns two dataframes with the second dataframe being the transpose of
    the first

    """
    data = pd.read_excel(x, skiprows=y)
    return data, data.transpose()


def poly(x, a, b, c):
    """
    Calculates the value of a polynomial function of the form ax^2 + bx + c.

    """
    return a*x**2 + b*x + c


def get_error_estimates(x, y, degree):
    """
   Calculates the error estimates of a polynomial function.
       """

    coefficients = np.polyfit(x, y, degree)
    y_estimate = np.polyval(coefficients, x)
    residuals = y - y_estimate

    return np.std(residuals)


# specify parameters and call the function to read the excel sheet containing
# the data

x = 'API_19_DS2_en_excel_v2_5455559.xls'
y = 3

# call the function to read the data
data, data_transpose = read(x, y)

# creating and cleaning a new dataframe with CO2 emissions for countries
CO2_emission = data[(data['Indicator Name'] == 'CO2 emissions (kt)')]

# drop unnecessary columns and empty fields so we have data for the years
# of interest
CO2_emission = CO2_emission.drop(['Country Code',
                                  'Indicator Code',
                                  'Indicator Name'], axis=1)
desired_columns = ['Country Name', '2009', '2019']
CO2_emission = CO2_emission[desired_columns]
CO2_emission = CO2_emission.dropna(how='any')
CO2_emission = CO2_emission.set_index('Country Name')

# We need population data for the 2 years so we can calculate CO2 emission
# per head
pop = data[(data['Indicator Name'] == 'Population, total')]
pop = pop[desired_columns]
pop = pop[pop['Country Name'].isin(CO2_emission.index)]
pop = pop.set_index('Country Name')

# Calculation CO2 emission per head for each country
CO2_emission['2009'] = CO2_emission['2009']/pop['2009']
CO2_emission['2019'] = CO2_emission['2019']/pop['2019']

# calling the function to read data for GDP per capita
GDP, GDP_transpose = read('API_NY.GDP.PCAP.CD_DS2_en_excel_v2_5454823.xls', 3)

# Cleaning the dataframe to include countries in the CO2 emission dataframe
GDP = GDP[desired_columns]
GDP = GDP[GDP['Country Name'].isin(CO2_emission.index)]
GDP = GDP.set_index('Country Name')

# Creating a new dataframe that has CO2emission per head and GDP per capita
# for the year 2009
data2009 = CO2_emission.drop(['2019'], axis=1)
data2009 = data2009.rename(columns={'2009': 'CO2 Emissions per head'})
data2009['GDP per capita'] = GDP['2009']
data2009 = data2009.dropna(how='any')

# Creating a new dataframe that has CO2emission per head and GDP per capita
# for the year 2019
data2019 = CO2_emission.drop(['2009'], axis=1)
data2019 = data2019.rename(columns={'2019': 'CO2 Emissions per head'})
data2019['GDP per capita'] = GDP['2019']
data2019 = data2019.dropna(how='any')

# visualising data
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(data2009['CO2 Emissions per head'],
               data2009['GDP per capita'], c='blue')
axs[0].set_title('2009')
axs[0].set_xlabel('CO2 emissions per head')
axs[0].set_ylabel('GDP per capita')

axs[1].scatter(data2019['CO2 Emissions per head'],
               data2019['GDP per capita'], c='red')
axs[1].set_title('2019')
axs[1].set_xlabel('CO2 emissions per head')
axs[1].set_ylabel('GDP per capita')
plt.tight_layout()

plt.show()