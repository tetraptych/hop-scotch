import dryscrape
from bs4 import BeautifulSoup
import requests
from time import sleep
import numpy as np
import cPickle as pickle
import pandas as pd
import re
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def pickle_object(something, name):
    with open('{}.pkl'.format(name), 'w') as f:
        pickle.dump(something, f)
    return None


def one_if_different_distance(A,B):
    return float(A != B)

def user_defined_distance(u, v):
    '''
    Important factors:
    alcohol
    avg_rating
    distillery
    country
    style
    volume
    unit_price
    '''
    rel_cols =     ['alcohol' , 'avg_rating', 'distillery', 'country', 'style', 'volume', 'unit_price']
    n = len(rel_cols)
    dist_arr = np.zeros(n)
    weig_arr = np.ones(n)

    dist_arr[0] = np.abs(u.alcohol - v.alcohol)/15
    dist_arr[1] += one_if_different_distance(u.distillery, v.distillery)
    dist_arr[2] += one_if_different_distance(u.country, v.country)
    dist_arr[3] += one_if_different_distance(u.style, v.style)
    dist_arr[4] += (u.unit_price - v.unit_price)**2

    return np.sum(dist_arr * weig_arr)



def dummy_distance(df,cols,weights):
    ndf = pd.get_dummies(df, prefix = cols, columns = cols)
    new_cols = set(ndf.cols) - set(df.cols)

    nweights = []

    for col, weight in zip(cols,weights):
        new_cols_with_given_prefix = {x for x in new_cols if str(col + '_') in x}
        n_cols_with_given_prefix = len(new_cols_with_given_prefix)
        for i in xrange(n_cols_with_given_prefix):
            nweights.append(weight)

    for col, weight in zip(new_cols,nweights):
        ndf[new_col] = ndf[new_col] * weight

    for col in cols:
        ndf.drop(col, inplace = True, axis = 1)

    X = ndf.values.astype(float)

    dists = pdist(X)

    return squareform(dists)



def pairwise_distances(df, f):

    pdist_list = []

    itera = xrange(len(df))
    combos = combinations(itera, 2)

    for combo in combos:
        pdist_list.append(f(df.loc[combo[0]],df.loc[combo[1]])  )
        if combo[1] == len(df-1):
            print 'Done with item number', combo[0]

    pickle_object(pdist_list, 'pairwise_distances')
    # y = pdist(df, f)
    return squareform(pdist_list)


def multidim_scaling(dist_mat):
    mds = MDS(dissimilarity = 'precomputed',n_jobs=-1, random_state=0)
    data = mds.fit_transform(dist_mat)

    # mds3d = MDS(dissimilarity = 'precomputed', n_components = 3, n_jobs=-1, random_state=0)
    # data3d = mds3d.fit_transform(dist_mat)

    styles = list(np.unique(rdf['style']))
    d = {}
    for i, style in enumerate(styles):
        d[style] = i

    dumb_function = lambda x : d[x]
    colors = rdf['style'].apply(dumb_function).astype(int)

    plt.scatter(data[:,0], data[:,1], c = colors[:10] )
    plt.legend()




if __name__ == '__main__':

    df  = pd.read_csv('whiskey_data_partial.csv', encoding = 'utf-8')

    df.drop_duplicates(subset = ['url'], inplace = True)
    df.drop(['tasting_note','bottling_note', 'Unnamed: 0'], axis = 1, inplace = True)

    edf = pd.read_csv('extra_cols.csv', encoding = 'utf-8')
    edf.drop_duplicates(subset = ['url'], inplace = True)
    edf.drop(['Unnamed: 0'], axis = 1, inplace = True)

    rdf = pd.merge(df, edf, how='inner', on='url')
    rdf['num_ratings'].fillna(0, inplace = True)

    rdf = rdf[~(rdf['volume'].isnull())]
    rdf['volume'] = rdf['volume'].apply(lambda x: x[:-2]).astype(float)
    rdf['unit_price'] = rdf['volume'] / rdf['price']

    rdf.reset_index(inplace=True)
    rdf.drop(['index'], axis = 1, inplace = True)

    rdf['review_list'] = rdf['review_list'].apply(lambda x: ''*(x == '()') + (x != '()')*x )
    rdf['review_list'] = rdf['review_list'].apply(lambda x: re.split('(?<=}),',  x))

    rdf['style'] = rdf['style'].apply(lambda x: x.replace('Whisky', 'Whiskey'))
    rdf['country'] = rdf['country'].apply(lambda x: x.replace('Whisky', 'Whiskey'))

    rdf['alcohol'] = rdf['alcohol'].apply(lambda x: float(x.strip('%')))


    with open('pairwise_distances.pkl') as f:
        pdist_list = pickle.load(f)

    dist_mat = squareform(pdist_list)

    tiny_dist_mat = dist_mat[:10,:10]
    del dist_mat

    dist_mat = tiny_dist_mat
