# -*- coding: utf-8 -*-
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer

def findCluster(X):
    """ Find clusters using GaussianMixture
    Call getElbow to find out the best number of clusters, and then use that to form clusters.

    Parameters
    ----------
    X : numpy array
        Features of training dataset

    Returns
    -------
    gmm : GaussianMixture()
        The trained cluster that can be used to predict which cluster an observation belongs to.
    """
    n_clusters = getElbow(X)
    gmm = GaussianMixture(n_components=n_clusters, random_state=8)
    gmm.fit(X)

    return gmm

def getElbow(X):
    """ Use Elbow approach to find best number of clusters
    Call yellowbrick.cluster.KElbowVisualizer to find out the best number of clusters.

    Parameters
    ----------
    X : numpy array
        Features of training dataset

    Returns
    -------
    visualizer.elbow_value_ : int
        The best number of clusters based on the passed in dataset
    """
    model = KMeans(random_state=8)
    visualizer = KElbowVisualizer(model, k=(4,12))
    visualizer.fit(X)
    return visualizer.elbow_value_

def addClusterFeatures(gmm, X, columns):
    """ Add cluster grouping feature and the distance to the cluster centre feature to dataframe

    Parameters
    ----------
    gmm : GaussianMixture()
        The trained cluster that can be used to predict which cluster an observation belongs to.
    X : numpy array
        Features of training dataset
    columns : List
        The list of column names to restructure the dataframe for X

    Returns
    -------
    df : DataFrame
        The reconstructed DataFrame from passed in numpy array plus features cluster and distance_to_cluster_mean
    """
    means = gmm.means_
    df = pd.DataFrame(X, columns=columns)
    df['cluster'] = gmm.predict(df)
    df['distance_to_cluster_mean'] = df.apply(lambda row: distance.euclidean(row[0:21], means[int(row['cluster'])]), axis=1)

    return df
