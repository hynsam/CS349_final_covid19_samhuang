import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json
from json import JSONEncoder
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
n_components = 5
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_deaths_global.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []

for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    cases_sum = np.sum(cases, axis=0)
    previous_day = np.concatenate((np.array([0]),cases_sum[:-1]))
    diff = cases_sum - previous_day
    features.append(diff)
    targets.append(labels[0][1])

print(len(targets))

predictions = {}


gmm = GaussianMixture(n_components=n_components)
gmm.fit(features)
means = gmm.means_


for degree in [1,2,3,4,5,6,7,8,9,10]:
    next_day_case = []
    for mean_idx, mean in enumerate(means):
        coef = np.polyfit(range(len(mean)), mean, deg=degree)
        next_day = len(mean) + 1
        predict_case = np.polyval(coef, next_day)
        next_day_case.append(predict_case)
    next_day_case = np.array(next_day_case)
    print(next_day_case)

country_cluster = gmm.predict(features)
print(len(country_cluster))
count = 0
for val in np.unique(confirmed["Country/Region"]):
    predictions[val] = country_cluster[count].tolist()
    count += 1

with open('results/final.json', 'w') as f:
    json.dump(predictions, f, indent=4)




"""
for _dist in ['minkowski', 'manhattan']:
    for val in np.unique(confirmed["Country/Region"]):
        # test data
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)

        # filter the rest of the data to get rid of the country we are
        # trying to predict
        mask = targets[:, 1] != val
        tr_features = features[mask]
        tr_targets = targets[mask][:, 1]

        above_min_cases = tr_features.sum(axis=-1) > MIN_CASES
        tr_features = np.diff(tr_features[above_min_cases], axis=-1)
        if NORMALIZE:
            tr_features = tr_features / tr_features.sum(axis=-1, keepdims=True)

        tr_targets = tr_targets[above_min_cases]

        # train knn
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric=_dist)
        knn.fit(tr_features, tr_targets)

        # predict
        cases = np.diff(cases.sum(axis=0, keepdims=True), axis=-1)
        # nearest country to this one based on trajectory
        label = knn.predict(cases)
        
        if val not in predictions:
            predictions[val] = {}
        predictions[val][_dist] = label.tolist()

with open('results/final.json', 'w') as f:
    json.dump(predictions, f, indent=4)
"""