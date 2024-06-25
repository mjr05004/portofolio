#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:33:00 2024

@author: majo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

df = pd.read_csv("airline_passenger_satisfaction.csv")
df.isnull()
df['Arrival Delay'].interpolate(method='linear', inplace = True)
df2 = df.copy()
# # hot encoding
# Hot encoding, also known as one-hot encoding, is a technique used in machine learning and data preprocessing to convert categorical data into a numerical format that can be provided to machine learning algorithms. In Python, this is often implemented using libraries such as scikit-learn or pandas.

# Here's an explanation of hot encoding:

#     Categorical Data:
#     Categorical data refers to data that represents categories or labels, such as color (red, green, blue) or animal types (cat, dog, bird). These categories are not inherently numerical and cannot be directly used as input to machine learning algorithms.

#     Hot Encoding Process:
#     Hot encoding transforms categorical data into a numerical format by creating binary columns (also known as dummy variables) for each category in the original data. For each category, a new binary column is created, and a value of 1 is placed in the column corresponding to the category for each observation, while all other columns are set to 0.

#     Example:
#     Let's say we have a dataset with a column called "Color" containing categorical data: ["Red", "Green", "Blue"]. After hot encoding, this single column would be transformed into three columns: "Color_Red", "Color_Green", and "Color_Blue". For each row, one of these columns would have a value of 1, indicating the color of that observation, while the other columns would have values of 0.

columns_to_encode = ['Gender', 'Customer Type', 'Type of Travel','Class', 'Satisfaction']

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=columns_to_encode)

# Overwrite original dataframe 
df=df_encoded

#normalize the columns that are non-binary and non-Likert scale: Age, Fight Distance, Departure Delay, Arrival Delay
from sklearn.preprocessing import StandardScaler

#first 4 columns after the ID column
columns_to_normalize = df.columns[1:5] 

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the selected columns
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# 7. Dropping the “ID” column before running the dataset through the K-Means algorithm. This will eliminate irrelevant info, reduce dimensionality, and improve interpretability.
df_without_id = df.drop('ID',axis=1)

from sklearn.cluster import KMeans

# Assuming your preprocessed DataFrame is named df

# Remove any non-numeric columns if present
df_numeric = df_without_id.select_dtypes(include='number')

# Determine the range of cluster numbers to consider
max_clusters = 23  # 23 Maximum number of clusters to consider

# Run K-means clustering and calculate distortions
distortions = []
for i in range(1, max_clusters+1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_numeric)
    distortions.append(kmeans.inertia_)

# Calculate the differences in distortions
differences = [distortions[i]-distortions[i-1] for i in range(1,len(distortions))]

# Find the cluster number with the significant change in distortion
elbow_index = differences.index(max(differences)) + 1
elbow_cluster_num = elbow_index + 1
# Plot the distortions and mark the elbow point
plt.plot(range(1, max_clusters+1), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Curve')
plt.axvline(x=elbow_cluster_num, color='r', linestyle='--',
 label='Elbow Point')
plt.legend()
plt.show()

# Use the cluster number at the elbow point for further analysis
print("Selected Number of Clusters:", elbow_cluster_num)

# Calculate the explained variance ratio
explained_var = pca.explained_variance_ratio_

# Print the explained variance ratio for each principal component
for i, ratio in enumerate(explained_var):
    print(f"Explained Variance Ratio for Principal Component {i+1}:{ratio}")

# Plot the explained variance ratio
plt.bar(range(len(explained_var)), explained_var)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component')
plt.show()
# Redo the amount of PCA n_compoents based on where it levels off. 
pca = PCA(n_components=9) #standard is 2
df_pca = pca.fit_transform(df_numeric)