import pandas as pd
import numpy as np
from scipy import sparse
import graphlab
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import numbers
import decimal
import cPickle as pickle
import re
import unidecode

# valid_numeric_chars = [x for x in range(0,10)]
# valid_numeric_chars += ['.']
#

def isfloat(value):
      try:
          float(value)
          return True
      except ValueError:
          return False

def only_numerics(one_row):
    inf = one_row['info'].copy()
    bad_keys = []
    for key, val  in inf.iteritems():
        if not isinstance(val, numbers.Number):
            bad_keys.append(key)

    for key in bad_keys:
        inf.pop(key)

    return inf


def clean_up_info_column(one_row):
    inf = one_row['info'].copy()

    for key, val in inf.iteritems():
        if (key == 'Bottled' or key == 'Vintage'):
            inf[key] = -10.0
            temp = re.search('\d{4}',val)
            if temp is not None:
                temp = temp.group(0)
                if isfloat(temp):
                    inf[key] = float(temp)
        if key == 'Number of bottles':
            inf[key] = -10.0
            if isfloat(val):
                inf[key] = float(val)
        if key == 'Size':
            inf[key] = -10.0
            if len(val) > 3 and isfloat( val[:-3] ):
                inf[key] = float(val[:-3])
        if key == 'Strength':
            if ('%' in val):
                pos = val.find('%')
                temp = val[:pos].strip()
                if isfloat(temp):
                    inf[key] = float(temp)
        if key == 'Price':
            inf[key] = -10.0
            if ('$' in val):
                temp = re.search('\$ *?\d+\.\d+',val)
                if temp is not None:
                    temp = temp.group(0)
                    temp = temp.replace('$', '').replace(' ', '')
                    if isfloat(temp):
                        inf[key] = float(temp)
        if key == 'Age':
            inf[key] = -10.0
            temp = re.search('\d+',val)
            if temp is not None:
                temp = temp.group(0)
                if isfloat(temp):
                    inf[key] = float(temp)

    if '' in inf:
        inf.pop('')
    if 'Details' in inf:
        inf.pop('Details')

    return inf

def other_model(ratings,item_info):
    '''
    graphlab.recommender.ranking_factorization_recommender.create(observation_data, user_id='user_id', item_id='item_id', target=None, user_data=None, item_data=None, num_factors=32, regularization=1e-09, linear_regularization=1e-09, side_data_factorization=True, ranking_regularization=0.25, unobserved_rating_value=None, num_sampled_negative_examples=4, max_iterations=25, sgd_step_size=0, random_seed=0, binary_target=False, solver='auto', verbose=True, **kwargs)
    '''

    sf_train, sf_validation = sf.random_split(0.8)

    item_data = graphlab.SFrame(item_info)
    params = {'num_factors': [18, 25],  'linear_regularization': [1e-6, 1e-7, 1e-8], 'regularization': [8e-4, 1e-3], 'user_id': 'user_name', 'item_id' : 'item_id', 'target' : 'score', 'max_iterations': 100,  'solver' : ['auto'] }

    job = graphlab.model_parameter_search.create((sf_train, sf_validation), graphlab.graphlab.recommender.factorization_recommender.create,  params)

    '''
    {'item_id': 'item_id',
     'linear_regularization': 1e-07,
     'max_iterations': 50,
     'num_factors': 18,
     'regularization': 0.0008,
     'solver': 'auto',
     'target': 'score',
     'user_id': 'user_name'}
     '''

    best_params = job.get_best_params()
    best_params['observation_data'] = sf
    best_params['item_data'] = item_data
    best_params['max_iterations'] = 250

    best_model = graphlab.recommender.factorization_recommender.create(**best_params)


    best_params.pop('item_data')
    best_model = graphlab.recommender.factorization_recommender.create(**best_params)
    #
    # training_rmse   = best_model.get('training_rmse')

    print training_rmse

    # pred = np.array(best_model.predict (sf ))

    return best_model, best_params


def convert_to_ratings_df(df):

    vals = list(df['ratings'].values)
    flattened = [item for sublist in vals for item in sublist]

    rdf = pd.DataFrame(np.asarray(flattened), columns = ['item_id', 'user_name', 'score'])
    rdf['item_id'] = rdf['item_id'].astype(int)
    rdf['score']   = rdf['score'].astype(float)
    return rdf



if __name__ == "__main__":
    with open('wb_whiskey_data.pkl') as f:
        df = pickle.load(f)

    rdf = df.copy()
    rdf['info'] = rdf.apply(lambda x: clean_up_info_column(x), axis = 1)

    ratings = convert_to_ratings_df(rdf)

    def dumb_function(row):
        if 'Price' in row['info'].keys():
            return row['info']['Price']
        else:
            return -10.0

    # rdf['info'] = rdf.apply(lambda x: dumb_function(x), axis = 1)
    rdf['info'] = rdf.apply(lambda x: only_numerics(x), axis = 1)
    item_info = rdf[['id', 'info']].rename(index = str, columns = {'id':'item_id', 'info':'dict_feature'})


    sf = graphlab.SFrame(ratings)
    item_data = graphlab.SFrame(item_info)

    m2 = graphlab.ranking_factorization_recommender.create(sf,user_id = 'user_name', item_id = 'item_id', target='score',item_data=item_data)
