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

# Normalize the data using MinMax Normalization method
p1 = data2009[['CO2 Emissions per head', 'GDP per capita']].values
p2 = data2019[['CO2 Emissions per head', 'GDP per capita']].values

max_value1 = p1.max()
min_value1 = p1.min()

max_value2 = p2.max()
min_value2 = p2.min()
# normalized data
p1_norm = (p1 - min_value1) / (max_value1 - min_value1)
p2_norm = (p2 - min_value2) / (max_value2 - min_value2)

sse1 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(p1_norm)
    sse1.append(kmeans.inertia_)

sse2 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(p2_norm)
    sse2.append(kmeans.inertia_)

# plotting to check for appropriate number of clusters using elbow method
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(range(1, 11), sse1, color='red')
axs[0].set_title('Elbow Method 2009')
axs[0].set_xlabel('Number of clusters')
axs[0].set_ylabel('SSE')

axs[1].plot(range(1, 11), sse2, color='red')
axs[1].set_title('Elbow Method 2019')
axs[1].set_xlabel('Number of clusters')
axs[1].set_ylabel('SSE')
plt.tight_layout()
plt.show()

# finding the Kmeans clusters
ncluster = 3
kmeans1 = KMeans(n_clusters=ncluster, init='k-means++',
                 max_iter=300, n_init=10, random_state=0)
kmeans2 = KMeans(n_clusters=ncluster, init='k-means++',
                 max_iter=300, n_init=10, random_state=0)


# Fit the model to the data
kmeans1 = kmeans1.fit(p1_norm)
kmeans2 = kmeans2.fit(p2_norm)

# labels
labels1 = kmeans1.labels_
labels2 = kmeans2.labels_

# Extracting clusters centers
cen1 = kmeans1.cluster_centers_
cen2 = kmeans2.cluster_centers_

# Plot the scatter plot of the clusters
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=140)
axs[0].scatter(p1_norm[:, 0], p1_norm[:, 1], c=labels1)
axs[0].set_title('K-Means Clustering 2009')
axs[0].set_xlabel('CO2 emissions per head')
axs[0].set_ylabel('GDP per capita')

axs[1].scatter(p2_norm[:, 0], p2_norm[:, 1], c=labels2)
axs[1].set_title('K-Means Clustering 2019')
axs[1].set_xlabel('CO2 emissions per head')
axs[1].set_ylabel('GDP per capita')
plt.tight_layout()

plt.show()

# Getting the Kmeans
kmeans1 = KMeans(n_clusters=ncluster, init='k-means++',
                 max_iter=300, n_init=10, random_state=0)
kmeans2 = KMeans(n_clusters=ncluster, init='k-means++',
                 max_iter=300, n_init=10, random_state=0)
y_predict1 = kmeans1.fit_predict(p1_norm)
y_predict2 = kmeans2.fit_predict(p2_norm)

# creating new dataframe with the labels for each year
data2009['cluster'] = y_predict1
data2019['cluster'] = y_predict2

data2009.to_csv('cluster_results2009.csv')
data2019.to_csv('cluster_results2019.csv')


# plotting the normalised graphs

plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=140)
axs[0].scatter(p1_norm[y_predict1 == 0, 0], p1_norm[y_predict1 ==
                                                    0, 1], s=50,
               c='orange', label='cluster 0')
axs[0].scatter(p1_norm[y_predict1 == 1, 0], p1_norm[y_predict1 ==
                                                    1, 1], s=50,
               c='blue', label='cluster 1')
axs[0].scatter(p1_norm[y_predict1 == 2, 0], p1_norm[y_predict1 ==
                                                    2, 1], s=50,
               c='yellow', label='cluster 2')
axs[0].scatter(cen1[:, 0], cen1[:, 1], s=50, c='black', label='Centroids')
axs[0].set_title(
    'Country clusters based on CO2 emissions and GDP par capita(2009)',
    fontweight='bold', fontsize=8)
axs[0].set_xlabel('CO2 Emissions per head', fontweight='bold')
axs[0].set_ylabel('GDP per capita', fontweight='bold')
axs[0].legend()

axs[1].scatter(p2_norm[y_predict2 == 0, 0], p2_norm[y_predict2 ==
                                                    0, 1], s=50,
               c='orange', label='cluster 0')
axs[1].scatter(p2_norm[y_predict2 == 1, 0], p2_norm[y_predict2 ==
                                                    1, 1], s=50,
               c='blue', label='cluster 1')
axs[1].scatter(p2_norm[y_predict2 == 2, 0], p2_norm[y_predict2 ==
                                                    2, 1], s=50,
               c='yellow', label='cluster 2')
axs[1].scatter(cen2[:, 0], cen2[:, 1], s=50, c='black', label='Centroids')
axs[1].set_title(
    'Country clusters based on CO2 emissions and GDP par capita(2019)',
    fontweight='bold', fontsize=8)
axs[1].set_xlabel('CO2 Emissions per head', fontweight='bold')
axs[1].set_ylabel('GDP per capita', fontweight='bold')
axs[1].legend()
plt.tight_layout()

plt.show()

# converting centroid to it's unnormalized form
cent1 = cen1 * (max_value1 - min_value1) + min_value1
cent2 = cen2 * (max_value2 - min_value2) + min_value2

# plotting the Population in their clusters with the centroid points
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=600)
axs[0].scatter(p1[y_predict1 == 0, 0], p1[y_predict1 == 0, 1],
               s=50, c='orange', label='Cluster 1')
axs[0].scatter(p1[y_predict1 == 1, 0], p1[y_predict1 == 1, 1],
               s=50, c='blue', label='Cluster 2')
axs[0].scatter(p1[y_predict1 == 2, 0], p1[y_predict1 == 2, 1],
               s=50, c='yellow', label='Cluster 3')
axs[0].scatter(cent1[:, 0], cent1[:, 1], s=50, c='black', label='Centroids')
axs[0].set_title(
    'Country clusters based on CO2 emissions and GDP per capita(2009)',
    fontweight='bold', fontsize=9)
axs[0].set_xlabel('CO2 Emissions per head', fontweight='bold', fontsize=8)
axs[0].set_ylabel('GDP per capita', fontweight='bold', fontsize=8)
axs[0].legend()

axs[1].scatter(p2[y_predict2 == 0, 0], p2[y_predict2 == 0, 1],
               s=50, c='orange', label='Cluster 1')
axs[1].scatter(p2[y_predict2 == 1, 0], p2[y_predict2 == 1, 1],
               s=50, c='blue', label='Cluster 2')
axs[1].scatter(p2[y_predict2 == 2, 0], p2[y_predict2 == 2, 1],
               s=50, c='yellow', label='Cluster 3')
axs[1].scatter(cent2[:, 0], cent2[:, 1], s=50, c='black', label='Centroids')
axs[1].set_title(
    'Country clusters based on CO2 emissions and GDP per capita(2019)',
    fontweight='bold', fontsize=9)
axs[1].set_xlabel('CO2 Emissions per head', fontweight='bold', fontsize=8)
axs[1].set_ylabel('GDP per capita', fontweight='bold', fontsize=8)
axs[1].legend()
plt.tight_layout()
plt.savefig('Clusters')
plt.show()

# We will now perform GDP/capita curve fitting for a country within
# each cluster (Honduras, Argentina and Barbados)

# call the function to read the data
gdp, gdp_transpose = read('API_NY.GDP.PCAP.CD_DS2_en_excel_v2_5454823.xls', 3)

# Clean data and obtain desired columns
gdp = gdp_transpose.drop(['Country Code', 'Indicator Name', 'Indicator Code'])
gdp.columns = gdp.iloc[0, :]
gdp = gdp[1:]
gdp = gdp[gdp.columns[gdp.columns.isin(['Honduras', 'Argentina', 'Barbados'])]]
gdp = gdp.dropna(how='any')
gdp['Year'] = gdp.index

# fit data for Honduras
gdp_hon = gdp[['Year', 'Honduras']].apply(pd.to_numeric, errors='coerce')

# Forecast for the next 20 years
year = np.arange(1970, 2041)

# fits the linear data
param_b, cov_b = opt.curve_fit(poly, gdp_hon['Year'],
                               gdp_hon['Honduras'])

# calculate standard deviation
sigma_b = np.sqrt(np.diag(cov_b))

# creates a new column for the fit figures
gdp_hon['fit'] = poly(gdp_hon['Year'], *param_b)

# forecasting the fit figures
forecast_b = poly(year, *param_b)

# error estimates
error_b = get_error_estimates(gdp_hon['Honduras'], gdp_hon['fit'], 2)
print('\n Error Estimates for Honduras GDP/Capita:\n', error_b)

# Plotting the fit
plt.style.use('fivethirtyeight')
plt.figure(dpi=600)
plt.plot(gdp_hon["Year"], gdp_hon["Honduras"],
         label="GDP/Capita", c='green')
plt.plot(year, forecast_b, label="Forecast", c='black')

plt.xlabel("Year", fontweight='bold', fontsize=14)
plt.ylabel("GDP per capita(US$)", fontweight='bold', fontsize=14)
plt.legend()
plt.title('Honduras', fontweight='bold', fontsize=14)
plt.savefig('Honduras.png')
plt.show()

# let's do the same thing for Barbados
gdp_bbs = gdp[['Year', 'Barbados']].apply(pd.to_numeric, errors='coerce')

# fits the linear data
param_lux, cov_lux = opt.curve_fit(poly, gdp_bbs['Year'],
                                   gdp_bbs['Barbados'])

# calculates the standard deviation
sigma_lux = np.sqrt(np.diag(cov_lux))

# creates a column for the fit data
gdp_bbs['fit'] = poly(gdp_bbs['Year'], *param_lux)

# forecasting for the next 20 years
forecast_lux = poly(year, *param_lux)

# error estimates
error_lux = get_error_estimates(gdp_bbs['Barbados'], gdp_bbs['fit'], 2)
print('\n Error Estimates for Barbados GDP/Capita:\n', error_lux)

# Plotting the fit
plt.style.use('fivethirtyeight')
plt.figure(dpi=600)
plt.plot(gdp_bbs["Year"], gdp_bbs['Barbados'], label="GDP/Capita",
         c='red')
plt.plot(year, forecast_lux, label="Forecast", c='black')
plt.xlabel("Year", fontweight='bold', fontsize=14)
plt.ylabel("GDP per Capita (US$)", fontweight='bold', fontsize=14)
plt.legend()
plt.title('Barbados', fontweight='bold', fontsize=14)
plt.savefig('Barbados.png')
plt.show()

# And lastly for Argentina
gdp_arg = gdp[['Year', 'Argentina']].apply(pd.to_numeric, errors='coerce')

# fits the linear data
param_q, cov_q = opt.curve_fit(poly, gdp_arg['Year'], gdp_arg['Argentina'])

# calculates the standard deviation
sigma_q = np.sqrt(np.diag(cov_q))

# creates a column for the fit data
gdp_arg['fit'] = poly(gdp_arg['Year'], *param_q)

# forecasting for the next 20 years
forecast_q = poly(year, *param_q)

# error estimates
error_q = get_error_estimates(gdp_arg['Argentina'], gdp_arg['fit'], 2)
print('\n Error Estimates for Argentina GDP/Capita:\n', error_q)

# Plotting the fit
plt.style.use('fivethirtyeight')
plt.figure(dpi=600)
plt.plot(gdp_arg["Year"], gdp_arg['Argentina'], label="GDP/Capita",  c='blue')
plt.plot(year, forecast_q, label="Forecast", c='black')
plt.xlabel("Year", fontweight='bold', fontsize=14)
plt.ylabel("GDP per Capita (US$)", fontweight='bold', fontsize=14)
plt.legend()
plt.title('Argentina', fontweight='bold', fontsize=14)
plt.savefig('Argentina.png')
plt.show()