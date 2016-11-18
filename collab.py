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

    if item_info:
        item_data = graphlab.SFrame(item_info)

    if item_info:
        possible_params['item_data'] = [item_data]

    ignore_cols = ['observation_data', 'item_data', 'user_id', 'item_id', 'target', 'max_iterations,' 'solver', 'max_iterations']
    df = pd.DataFrame()

    all_possible_params = list(itertools.product([40], [1e-5, 1e-4], [1e-5, 1e-4]))

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

    return None



def other_model(ratings,item_info):
    '''
    graphlab.recommender.ranking_factorization_recommender.create(observation_data, user_id='user_id', item_id='item_id', target=None, user_data=None, item_data=None, num_factors=32, regularization=1e-09, linear_regularization=1e-09, side_data_factorization=True, ranking_regularization=0.25, unobserved_rating_value=None, num_sampled_negative_examples=4, max_iterations=25, sgd_step_size=0, random_seed=0, binary_target=False, solver='auto', verbose=True, **kwargs)
    '''
    sf = graphlab.SFrame(ratings)
    # item_data = graphlab.SFrame(item_info)

    sf_train, sf_validation = sf.random_split(0.7)
    folds = graphlab.cross_validation.KFold(sf, 5)
    params = {'num_factors': [35],  'linear_regularization': [1e-10, 1e-8], 'regularization': [1e-8, 1e-6], 'user_id': 'user_name', 'item_id' : 'item_id', 'target' : 'score', 'max_iterations': 100,  'solver' : ['auto'] }

    # job = graphlab.model_parameter_search.create((sf_train, sf_validation), graphlab.graphlab.recommender.factorization_recommender.create,  params)

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
    best_params['observation_data'] = sf_train
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

    return rdf






if __name__ == "__main__":
    with open('wb_whiskey_data.pkl') as f:
        df = pickle.load(f)

    with open('wb_whiskey_data_part2.pkl') as f:
        df2 = pickle.load(f)

    df3 = df.append(df2,ignore_index = True)
    df3.drop_duplicates(subset = ['url'], inplace = True)

    rdf = df3.copy()
    rdf['info'] = rdf.apply(lambda x: clean_up_info_column(x), axis = 1)

    ratings = convert_to_ratings_df(rdf)

    item_cutoff = 6
    user_cutoff = 6
    ndf, nratings = throw_out_unrated_items(rdf,ratings,item_cutoff,user_cutoff)

    ratings = nratings
    rdf = ndf

    # rdf = extract_additional_columns(rdf)
    rdf = extract_additional_columns(rdf)


    rdf['info'] = rdf.apply(lambda x: only_numerics(x), axis = 1)

    # rdf2 = rdf.dropna(inplace = False)

    # item_info = rdf[['id', 'info']].rename(index = str, columns = {'id':'item_id'})
    # item_info = rdf2[['id', 'price', 'strength']].rename(index = str, columns = {'id':'item_id'})


    sf = graphlab.SFrame(ratings)
    # item_data = graphlab.SFrame(item_info)


    # articles = np.array(rdf['notes'])
    # one_big_article = ' ;;; '.join(list(articles))
    # one_big_article_as_list = one_big_article.lower().split()
    # c = Counter(one_big_article_as_list)
    '''
    sweet,     fruits, cask, vanilla, malt, peat, spicy, wood, fresh, orange, hints, honey, lemon, smoke, chocolate, citrus, sweetness, caramel, toffee, oak, burnt, tropical, coffee
    '''
    # vectorizer = CountVectorizer(stop_words = 'english', tokenizer = word_tokenize, preprocessor = PorterStemmer().stem, max_features = 4000 , min_df = 50, max_df = 0.5, lowercase = True)
    # vectors = vectorizer.fit_transform(articles)

    # model = best_model
    # recommendations = pd.DataFrame(model.recommend(['slothmister'],400).to_numpy(), columns = ['user_name', 'item_id', 'predicted_rating', 'item_rank'])
    # recommendations['item_id'] = recommendations['item_id'].astype(int)
    # recommendations['predicted_rating'] = recommendations['predicted_rating'].astype(float)
    # recommendations['item_rank'] = recommendations['item_rank'].astype(int)
    #
    # recommendations_df = pd.merge(left = recommendations, right = rdf, how = 'inner', left_on='item_id', right_on = 'id')
    # relevant_cols = ['user_name', 'item_id', 'predicted_rating', 'item_rank', 'id', 'url', 'price', 'district', 'scotch_bool', 'united_states_bool', 'canada_bool', 'japan_bool', 'ireland_bool', 'single_malt_bool', 'blend_bool']
    #
    # recommendations_df = recommendations_df[relevant_cols]
    #
    # min_price = 30.00
    # max_price = 59.99
    #
    # recommendations_df[(recommendations_df['price'] < max_price) & ((recommendations_df['price'] > min_price))]['item_id']
    #
