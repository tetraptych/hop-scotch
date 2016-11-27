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
import itertools


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

def get_one_column_from_dict(one_row, col_name, key):
    inf = one_row.copy()[col_name]
    result = None
    if key in inf:
        result = inf[key]
    return result


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

def proprietary_grid_search(ratings,item_info = None):
    sf = graphlab.SFrame(ratings)
    sf_train, sf_validation = sf.random_split(0.7)

    if item_info is not None:
        item_data = graphlab.SFrame(item_info)

    ignore_cols = ['observation_data', 'item_data', 'user_id', 'item_id', 'target', 'max_iterations,' 'solver', 'max_iterations']
    df = pd.DataFrame()

    all_possible_params = list(itertools.product([30], [1e-8, 5e-7], [1e-4, 5e-4]))

    for index, i in enumerate(all_possible_params):
        print 'Here goes model {} of {}...'.format(index+1, len(all_possible_params))
        param_dict = {}
        param_dict['num_factors']           = i[0]
        param_dict['linear_regularization'] = i[1]
        param_dict['regularization']        = i[2]

        param_dict['observation_data'] = sf_train
        param_dict['user_id'] = 'user_name'
        param_dict['item_id'] = 'item_id'
        param_dict['target']  = 'score'
        param_dict['solver']  = 'auto'
        param_dict['max_iterations'] = 150
        param_dict['verbose'] = False

        if item_info is not None:
            param_dict['item_data'] = item_data

        model = graphlab.recommender.factorization_recommender.create(**param_dict)
        d = {}
        for param in param_dict:
            if param not in ignore_cols:
                d[str(param)] = param_dict[param]
        d['training_rmse'] = model.get('training_rmse')
        d['test_rmse']     = model.evaluate_rmse(sf_validation, 'score')['rmse_overall']

        print 'Model {} Training RMSE: '.format(index+1), d['training_rmse']
        print 'Model {} Validation RMSE: '.format(index+1), d['test_rmse']

        one_row_df = pd.DataFrame.from_dict([d])
        df = df.append(one_row_df, ignore_index = True)

        #BSOFAR:
        # no items...
        # 40, -5, -3 --> 3.68
        # 40, -6, -3 --> 3.66
        # 40, -7, -3 --> 3.67
        # 40, 5e-7, 5e-4 --> 3.488
        # 30, 5e-7, 5e-4 --> 3.484
        # 80, 5e-7, 5e-4 --> 3.483
        # 200, 5e-7, 5e-4 --> 3.4788
        # 400, .... --> 3.47655
        # 1000, .... --> 3.54603896453
        # 2000, .... --> 3.47841900283

        # side data present (just price)
        #

    # {'item_id': 'item_id',
    #  'linear_regularization': 5e-07,
    #  'max_iterations': 150,
    #  'num_factors': 350,
    #  'regularization': 0.0005,
    #  'solver': 'auto',
    #  'target': 'score',
    #  'user_id': 'user_name',
    #  'verbose': False,
    #  'observation_data': sf_train
    #  }
    #


    return None

def ranking_test():
    # Signature: graphlab.recommender.ranking_factorization_recommender.create(observation_data, user_id='user_id', item_id='item_id', target=None, user_data=None, item_data=None, num_factors=32, regularization=1e-09, linear_regularization=1e-09, side_data_factorization=True, ranking_regularization=0.25, unobserved_rating_value=None, num_sampled_negative_examples=4, max_iterations=25, sgd_step_size=0, random_seed=0, binary_target=False, solver='auto', verbose=True, **kwargs)

    sf_train, sf_validation = sf.random_split(0.85)


    folds = graphlab.cross_validation.KFold(sf_train, 3)

    params = { 'num_factors': [70],'linear_regularization': [0, 5e-7, 1e-6], 'regularization': [1e-6, 5e-4], 'user_id': ['user_name'], 'item_id' : ['item_id'], 'target': ['score'], 'side_data_factorization': [True], 'max_iterations': [50], 'solver':['auto'], 'ranking_regularization':[0.10, 0.2, 0.5], 'unobserved_rating_value':[76, 84], 'num_sampled_negative_examples': [4], 'item_data':[item_data], 'verbose': ['True']}

    # 86.61654695190293 is mean score in ratings data
    # d = {'item_data':item_data}
    # params = params.update(d)

    job = graphlab.grid_search.create(folds, graphlab.graphlab.recommender.ranking_factorization_recommender.create,  params)

    # {'item_id': 'item_id',
    #  'linear_regularization': 1e-08,
    #  'max_iterations': 100,
    #  'num_factors': 60,
    #  'num_sampled_negative_examples': 6,
    #  'ranking_regularization': 0.1,
    #  'regularization': 1e-08,
    #  'side_data_factorization': True,
    #  'solver': 'auto',
    #  'target': 'score',
    #  'unobserved_rating_value': 70,
    #  'user_id': 'user_name'}
    #  {'test_rmse': 9.816224056062088, 'training_rmse': 1.6767458847979384}

    # {'item_id': 'item_id',
    #  'linear_regularization': 1e-05,
    #  'max_iterations': 100,
    #  'num_factors': 70,
    #  'num_sampled_negative_examples': 4,
    #  'ranking_regularization': 0.075,
    #  'regularization': 1e-05,
    #  'side_data_factorization': True,
    #  'solver': 'auto',
    #  'target': 'score',
    #  'unobserved_rating_value': 80,
    #  'user_id': 'user_name'}
    #  {'test_rmse': 4.911302350167563, 'training_rmse': 0.80120231530767}

    # WITH SIDE DATA (just price)
    #  {    'item_id': 'item_id',
    #  'linear_regularization': 5e-07,
    #  'max_iterations': 100,
    #  'num_factors': 70,
    #  'num_sampled_negative_examples': 4,
    #  'ranking_regularization': 0.8,
    #  'regularization': 0.0005,
    #  'side_data_factorization': True,
    #  'solver': 'auto',
    #  'target': 'score',
    #  'unobserved_rating_value': 80,
    #  'user_id': 'user_name'}
    # {'test_rmse': 5.317625945681785, 'training_rmse': 5.267574589697656}

    # WITH ALL SIDE DATA (ROUGHLY)
    # | Parameter                      | Description                                      | Value    |
    # +--------------------------------+--------------------------------------------------+----------+
    # | num_factors                    | Factor Dimension                                 | 70       |
    # | regularization                 | L2 Regularization on Factors                     | 1e-05    |
    # | solver                         | Solver used for training                         | adagrad  |
    # | linear_regularization          | L2 Regularization on Linear Coefficients         | 1e-10    |
    # | ranking_regularization         | Rank-based Regularization Weight                 | 0.2      |
    # | unobserved_rating_value        | Ranking Target Rating for Unobserved Interacti...| 84       |
    # | side_data_factorization        | Assign Factors for Side Data                     | True     |
    # | max_iterations                 | Maximum Number of Iterations                     | 100      |


    #  'item_id': 'item_id',
    #  'linear_regularization': 1e-06,
    #  'max_iterations': 150,
    #  'num_factors': 70,
    #  'num_sampled_negative_examples': 4,
    #  'ranking_regularization': 0.2,
    #  'regularization': 0.0005,
    #  'side_data_factorization': True,
    #  'solver': 'auto',
    #  'target': 'score',
    #  'unobserved_rating_value': 76,
    #  'user_id': 'user_name',
    #  'verbose': 'True'}
    # Out[39]: {'test_rmse': 4.991928258308022, 'training_rmse': 4.951283703940456}
    # SAME BUT WITHOUT SIDE DATA: Out[46]: {'test_rmse': 4.28515925633731, 'training_rmse': 4.143998434352305}
    # SAME BUT WITHOUT SIDE DATA AND WITH 80 as unobserved_rating_value: Out[56]: {'test_rmse': 3.947561502817736, 'training_rmse': 3.787815991410602}
    # SAME BUT WITH num_sampled_negative_examples = 8: Out[60]: {'test_rmse': 4.001269245274644, 'training_rmse': 3.850898652935453}
    # SAME BUT WITH 0.16 RANKING REG  Out[72]: {'test_rmse': 3.8952205577398638, 'training_rmse': 3.7439633013450937}
    # SAME BUT WITH 0.125 RANKING REG Out[78]: {'test_rmse': 3.7969037835764192, 'training_rmse': 3.6319951405020534}
    # 0.0 Ranking REg: Out[84]: {'test_rmse': 3.4881022333847236, 'training_rmse': 3.283864964165282}


    ###### THIS IS WHAT YOU USED ####
    # {'item_id': 'item_id',
    #  'linear_regularization': 1e-06,
    #  'max_iterations': 125,
    #  'num_factors': 70,
    #  'num_sampled_negative_examples': 8,
    #  'ranking_regularization': 0.05,
    #  'regularization': 0.0005,
    #  'side_data_factorization': False,
    #  'solver': 'auto',
    #  'target': 'score',
    #  'unobserved_rating_value': 78,
    #  'user_id': 'user_name',
    #  'verbose': 'True'}
    # Out[117]: {'test_rmse': 3.636764055807749, 'training_rmse': 3.4510265770143094}


    # {'item_id': 'item_id',
    #  'linear_regularization': 1e-06,
    #  'max_iterations': 80,
    #  'num_factors': 90,
    #  'num_sampled_negative_examples': 8,
    #  'ranking_regularization': 0.075,
    #  'regularization': 0.0008,
    #  'side_data_factorization': False,
    #  'solver': 'auto',
    #  'target': 'score',
    #  'unobserved_rating_value': 79,
    #  'user_id': 'user_name',
    #  'verbose': 'True'}
    # Out[51]: {'test_rmse': 3.922520281678877, 'training_rmse': 3.7553683913405482}


    # {'item_id': 'item_id',
    #  'linear_regularization': 2e-06,
    #  'max_iterations': 125,
    #  'num_factors': 90,
    #  'num_sampled_negative_examples': 8,
    #  'ranking_regularization': 0.06,
    #  'regularization': 0.0007,
    #  'side_data_factorization': False,
    #  'solver': 'auto',
    #  'target': 'score',
    #  'unobserved_rating_value': 79,
    #  'user_id': 'user_name',
    #  'verbose': 'True'}
    # Out[67]: {'test_rmse': 3.82808198738577, 'training_rmse': 3.6321160542955058}


    # {'item_id': 'item_id',
    #  'linear_regularization': 1.5e-06,
    #  'max_iterations': 125,
    #  'num_factors': 90,
    #  'num_sampled_negative_examples': 8,
    #  'ranking_regularization': 0.06,
    #  'regularization': 0.0006,
    #  'side_data_factorization': False,
    #  'solver': 'auto',
    #  'target': 'score',
    #  'unobserved_rating_value': 80,
    #  'user_id': 'user_name',
    #  'verbose': 'True'}
    # Out[77]: {'test_rmse': 3.757597723761622, 'training_rmse': 3.529633094748577}

    params = { 'num_factors': [90],'linear_regularization': [1e-6, 2e-6], 'regularization': [5e-4, 7e-4], 'user_id': ['user_name'], 'item_id' : ['item_id'], 'target': ['score'], 'side_data_factorization': [False], 'max_iterations': [80], 'solver':['auto'], 'ranking_regularization':[0.06, 0.05], 'unobserved_rating_value':[79], 'num_sampled_negative_examples': [8], 'verbose': ['True']}

    best_params = job.get_best_params()
    best_params['observation_data'] = sf_train
    # best_params['item_data'] = item_data
    best_params['max_iterations'] = 225

    # best_params.pop('item_data')
    # best_params['side_data_factorization'] = False

    model = graphlab.recommender.ranking_factorization_recommender.create(**best_params)

    d = dict()
    d['training_rmse'] = model.get('training_rmse')
    d['test_rmse']     = model.evaluate_rmse(sf_validation, 'score')['rmse_overall']

    models = job.get_models()
    for index, model in enumerate(models):
        print '@@@ MODEL NUMBER {} @@@'.format(index)
        print 'Training RMSE: {}'.format(model.get('training_rmse'))
        print 'Validate RMSE: {}'.format(model.evaluate_rmse(sf_validation, 'score')['rmse_overall'])






def other_model(ratings,item_info):
    '''
    graphlab.recommender.ranking_factorization_recommender.create(observation_data, user_id='user_id', item_id='item_id', target=None, user_data=None, item_data=None, num_factors=32, regularization=1e-09, linear_regularization=1e-09, side_data_factorization=True, ranking_regularization=0.25, unobserved_rating_value=None, num_sampled_negative_examples=4, max_iterations=25, sgd_step_size=0, random_seed=0, binary_target=False, solver='auto', verbose=True, **kwargs)
    '''
    sf = graphlab.SFrame(ratings)
    # item_data = graphlab.SFrame(item_info)

    sf_train, sf_validation = sf.random_split(0.7)
    folds = graphlab.cross_validation.KFold(sf, 5)
    params = {'num_factors': [35],  'linear_regularization': [1e-10, 1e-8], 'regularization': [1e-8, 1e-6], 'user_id': 'user_name', 'item_id' : 'item_id', 'target' : 'score', 'max_iterations': 100,  'solver' : ['auto'] }

    job = graphlab.model_parameter_search.create((sf_train, sf_validation), graphlab.graphlab.recommender.factorization_recommender.create,  params)

    job = graphlab.model_parameter_search.create(folds, graphlab.graphlab.recommender.factorization_recommender.create,  params, max_models = 10)

    '''
    {'item_id': 'item_id',
     'linear_regularization': 1e-06,
     'max_iterations': 100,
     'num_factors': 50,
     'regularization': 1e-04,
     'solver': 'auto',
     'target': 'score',
     'user_id': 'user_name'}
     '''

    '''
    {'item_id': 'item_id',
     'linear_regularization': 0,
     'max_iterations': 100,
     'num_factors': 35,
     'regularization': 0,
     'solver': 'auto',
     'target': 'score',
     'user_id': 'user_name'}
     '''


    best_params = job.get_best_params()
    best_params['observation_data'] = sf
    # best_params['item_data'] = item_data
    best_params['max_iterations'] = 150

    best_model = graphlab.recommender.factorization_recommender.create(**best_params)


    # best_params.pop('item_data')
    best_model = graphlab.recommender.factorization_recommender.create(**best_params)
    #
    # training_rmse   = best_model.get('training_rmse')

    print training_rmse

    pred = np.array(best_model.predict (sf_validation ))
    best_model.evaluate_rmse(sf_validation, 'score')
    # no reg: 4.28
    # some reg: 4.26
    # high reg: 3.718584296891607

    return best_model, best_params

'''
def print_topics(gl_model, rdf, items_to_show = 0, users_to_show = 0):

    item_sf = gl_model.coefficients['item_id']

    col1 = item_sf['item_id'].to_numpy()
    col2 = item_sf['linear_terms'].to_numpy()
    col3 = item_sf['factors'].to_numpy()

    df = pd.DataFrame()
    df['item_id'] = col1
    df['linear_terms'] = col2
    for i in range(len(col3[0])):
        new_col = 'factor{}'.format(i)
        df[new_col] = col3[:,i]

    H = np.array(df)

    H_items_in_topic = np.asarray(np.argsort(H,axis=1)[:])[:,(-1)*(items_to_show):]
    W_users_in_topic = np.asarray(np.argsort(W.T,axis = 1)[:])[:,(-1)*(users_to_show):]

    for index, (i1, i2) in enumerate(zip(H_items_in_topic, W_users_in_topic)):
        print 'Topic number: {}'.format(index)
        user_list =[]
        item_list =[]
        if items_to_show > 0:
            for j in i1:
                item_list.append(rdf.url[j])
            print 'Topic {} consists of the following items: '.format(index)
            print topic_list
        # print '****'
        # if users_to_show > 0:
        #     for j in i2:
        #         user_list.append(rdf.url[j])
        #     print 'Topic {} consists of the following users: '.format(index)
        #     print names
        #     print '********'*3

    return None
'''



def convert_to_ratings_df(df):

    vals = list(df['ratings'].values)
    flattened = [item for sublist in vals for item in sublist]

    rdf = pd.DataFrame(np.asarray(flattened), columns = ['item_id', 'user_name', 'score'])
    rdf['item_id'] = rdf['item_id'].astype(int)
    rdf['score']   = rdf['score'].astype(float)
    return rdf



def throw_out_unrated_items(rdf, ratings, item_cutoff = 0 , user_cutoff = 0):
    g = ratings.groupby('item_id').count()
    valid_ids = g[ g['user_name'] > item_cutoff ].index

    ndf = rdf[rdf['id'].isin(valid_ids)]
    nratings = ratings[ratings['item_id'].isin(valid_ids)]

    g2 = nratings.groupby('user_name').count()
    valid_users = g2[ g2['item_id'] > user_cutoff ].index

    nratings2 = nratings[nratings['user_name'].isin(valid_users)]

    return ndf, nratings2


# temp = []
# for cutoff_num in range(0,20):
#
#     g = ratings.groupby('item_id').count()
#     valid_ids = g[ g['user_name'] > cutoff_num ].index
#
#     temp.append(len(rdf[rdf['id'].isin(valid_ids)]))


def extract_additional_columns(rdf):

    # numerics
    rdf['price']    = rdf.apply(lambda row: get_one_column_from_dict(row, 'info', 'Price'    ), axis = 1)
    rdf['strength'] = rdf.apply(lambda row: get_one_column_from_dict(row, 'info', 'Strength' ), axis = 1)
    rdf['age']      = rdf.apply(lambda row: get_one_column_from_dict(row, 'info', 'Age'      ), axis = 1)
    rdf['bottled']  = rdf.apply(lambda row: get_one_column_from_dict(row, 'info', 'Bottled'  ), axis = 1)
    rdf['vintage']  = rdf.apply(lambda row: get_one_column_from_dict(row, 'info', 'Vintage'  ), axis = 1)

    # categorical
    rdf['category'] = rdf.apply(lambda row: get_one_column_from_dict(row, 'info', 'Category' ), axis = 1)
    rdf['district'] = rdf.apply(lambda row: get_one_column_from_dict(row, 'info', 'District' ), axis = 1)
    rdf['casktype'] = rdf.apply(lambda row: get_one_column_from_dict(row, 'info', 'Casktype' ), axis = 1)

    # numerics (note: these all show 0.0 for missing values )
    rdf['average_body']          = rdf.apply( lambda row: get_one_column_from_dict(row, 'summary', 'Body'          ), axis = 1)
    rdf['average_finish']        = rdf.apply( lambda row: get_one_column_from_dict(row, 'summary', 'Finish'        ), axis = 1)
    rdf['average_initial_taste'] = rdf.apply( lambda row: get_one_column_from_dict(row, 'summary', 'Initial taste' ), axis = 1)
    rdf['average_nose']          = rdf.apply( lambda row: get_one_column_from_dict(row, 'summary', 'Nose - Aroma'  ), axis = 1)
    rdf['average_presentation']  = rdf.apply( lambda row: get_one_column_from_dict(row, 'summary', 'Presentation'  ), axis = 1)
    rdf['average_price']         = rdf.apply( lambda row: get_one_column_from_dict(row, 'summary', 'Price'         ), axis = 1)
    rdf['average_weighted_rate'] = rdf.apply( lambda row: get_one_column_from_dict(row, 'summary', 'Weighted Rate' ), axis = 1)

    # Location, location, type
    rdf['scotch_bool']           = rdf['category'].apply(lambda row: int('Scotland'      in row ))
    rdf['united_states_bool']    = rdf['category'].apply(lambda row: int('United States' in row ))
    rdf['canada_bool']           = rdf['category'].apply(lambda row: int('Canada'        in row ))
    rdf['japan_bool']            = rdf['category'].apply(lambda row: int('Japan'         in row ))
    rdf['ireland_bool']          = rdf['category'].apply(lambda row: int('Ireland'       in row ))
    rdf['single_malt_bool']      = rdf['category'].apply(lambda row: int('Single Malt'   in row ))
    rdf['blend_bool']            = rdf['category'].apply(lambda row: int('Blend'         in row ))

    # Number of ratings
    rdf['num_ratings']           = rdf['ratings'].apply(len)

    return rdf






if __name__ == "__main__":

    print 'Loading data...'

    with open('wb_whiskey_data.pkl') as f:
        df = pickle.load(f)

    with open('wb_whiskey_data_part2.pkl') as f:
        df2 = pickle.load(f)

    with open('wb_whiskey_data_part3.pkl') as f:
        df3 = pickle.load(f)

    df4 = df.append(df2,ignore_index = True).append(df3,ignore_index = True)
    df4.drop_duplicates(subset = ['url'], inplace = True)

    rdf = df4.copy()

    print 'Extracting columns...'

    rdf['info'] = rdf.apply(lambda x: clean_up_info_column(x), axis = 1)

    rdf = extract_additional_columns(rdf)
    rdf.dropna(subset = ['price'],inplace = True)


    print 'Building ratings data...'

    ratings = convert_to_ratings_df(rdf)

    #np.cumsum(rdf.groupby('num_ratings').count()['id'])
    item_cutoff = 9
    user_cutoff = 7
    ndf, nratings = throw_out_unrated_items(rdf,ratings,item_cutoff,user_cutoff)

    ratings = nratings
    rdf = ndf

    with open('wb_whiskey_photo_data.pkl') as f:
        photo_df = pickle.load(f)

    photo_df['id'] = photo_df['id'].apply(int)
    photo_df['brand'] = photo_df['brand'].apply(lambda x: x.replace('\n', ''))

    rdf = pd.merge(left = rdf, right = photo_df, how = 'left', left_on='id', right_on = 'id')

    rdf['null_photo_url'] = rdf['photo_url'].isnull()
    rdf['photo_url'].fillna('https://static.whiskybase.com/storage/whiskies/default/normal.png' , inplace = True)
    # rdf = extract_additional_columns(rdf)
    # rdf['info'] = rdf.apply(lambda x: only_numerics(x), axis = 1)

    # item_info = rdf[['id', 'info']].rename(index = str, columns = {'id':'item_id'})
    # item_info = rdf[['id', 'price']].rename(index = str, columns = {'id':'item_id'})
    item_info = rdf[['id', 'price', 'scotch_bool', 'united_states_bool', 'single_malt_bool', 'blend_bool']].rename(index = str, columns = {'id':'item_id'})

    sf = graphlab.SFrame(ratings)
    item_data = graphlab.SFrame(item_info)


    # articles = np.array(rdf['notes'])
    # one_big_article = ' ;;; '.join(list(articles))
    # one_big_article_as_list = one_big_article.lower().split()
    # c = Counter(one_big_article_as_list)
    '''
    sweet,     fruits, cask, vanilla, malt, peat, spicy, wood, fresh, orange, hints, honey, lemon, smoke, chocolate, citrus, sweetness, caramel, toffee, oak, burnt, tropical, coffee
    '''
    # vectorizer = CountVectorizer(stop_words = 'english', tokenizer = word_tokenize, preprocessor = PorterStemmer().stem, max_features = 4000 , min_df = 50, max_df = 0.5, lowercase = True)
    # vectors = vectorizer.fit_transform(articles)


    print 'Building model...'

    best_params = {'item_id': 'item_id', \
     'linear_regularization': 5e-08, \
     'max_iterations': 100, \
     'num_factors': 80, \
     'regularization': 1e-04, \
     'solver': 'auto', \
     'target': 'score', \
     'user_id': 'user_name'}


    best_params['observation_data'] = sf
    # best_params['item_data'] = item_data
    best_params['max_iterations'] = 300

    # best_model = graphlab.recommender.factorization_recommender.create(**best_params)

    # model = best_model

    if False:
        user = 'slothmister'
        # user = 'Miss Islay'
        recommendations = pd.DataFrame(model.recommend([user],len(rdf)).to_numpy(), columns = ['user_name', 'item_id', 'predicted_rating', 'item_rank'])
        recommendations['item_id'] = recommendations['item_id'].astype(int)
        recommendations['predicted_rating'] = recommendations['predicted_rating'].astype(float)
        recommendations['item_rank'] = recommendations['item_rank'].astype(int)

        recommendations_df = pd.merge(left = recommendations, right = rdf, how = 'inner', left_on='item_id', right_on = 'id')
        relevant_cols = ['user_name', 'item_id', 'predicted_rating', 'item_rank', 'id', 'url', 'price', 'district', 'scotch_bool', 'united_states_bool', 'canada_bool', 'japan_bool', 'ireland_bool', 'single_malt_bool', 'blend_bool']
        recommendations_df = recommendations_df[relevant_cols]

        q75, q25 = np.percentile(recommendations_df['predicted_rating'], [75 ,25])
        recommendations_df = recommendations_df[recommendations_df['predicted_rating'] > q75]

        min_price = 0
        max_price = 70.00

        cheapos = recommendations_df[(recommendations_df['price'] < max_price) & ((recommendations_df['price'] > min_price))]

        jena = pd.DataFrame(columns = ['item_id', 'user_name','score'])
        jena['item_id'] = jena['item_id'].astype(int)
        jena['score'] = jena['score'].astype(float)

        d1 = {'item_id': 18132, 'user_name': 'JENAMYF', 'score': 95}
        d2 = {'item_id': 15797, 'user_name': 'JENAMYF', 'score': 95}
        d3 = {'item_id': 23799, 'user_name': 'JENAMYF', 'score': 95}
        d4 = {'item_id': 40702, 'user_name': 'JENAMYF', 'score': 90}
        d5 = {'item_id': 15270, 'user_name': 'JENAMYF', 'score': 100}
        d6 = {'item_id': 15251, 'user_name': 'JENAMYF', 'score': 100}


        one_row_df = pd.DataFrame.from_dict([d1,d2,d3,d4,d5,d6])
        jena = jena.append(one_row_df, ignore_index = True)
        jena_sf = graphlab.SFrame(jena)

        jena_recs = model.recommend_from_interactions(jena_sf,k = len(rdf),verbose = True)

        recommendations = pd.DataFrame(jena_recs.to_numpy(), columns = ['item_id', 'predicted_rating', 'item_rank'])
        recommendations['item_id'] = recommendations['item_id'].astype(int)
        recommendations['predicted_rating'] = recommendations['predicted_rating'].astype(float)
        recommendations['item_rank'] = recommendations['item_rank'].astype(int)

        recommendations_df = pd.merge(left = recommendations, right = rdf, how = 'inner', left_on='item_id', right_on = 'id')
        relevant_cols = ['item_id', 'predicted_rating', 'item_rank', 'id', 'url', 'price', 'district', 'scotch_bool', 'united_states_bool', 'canada_bool', 'japan_bool', 'ireland_bool', 'single_malt_bool', 'blend_bool']
        recommendations_df = recommendations_df[relevant_cols]

        q75, q25 = np.percentile(recommendations_df['predicted_rating'], [52 ,25])
        recommendations_df = recommendations_df[recommendations_df['predicted_rating'] > q75]

        cheapos = recommendations_df[(recommendations_df['price'] < max_price) & ((recommendations_df['price'] > min_price))]


        model.get_similar_items([15797], k = 30).print_rows(30)
        model.coefficients['item_id'][0]['factors']
