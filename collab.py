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
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import pdist, squareform


def isfloat(value):
    '''
    INPUT: str
    OUTPUT: bool

    Determines if the value can be cast as a float. If so, returns True. Otherwise, returns False.
    '''
    try:
      float(value)
      return True
    except ValueError:
      return False

def only_numerics(one_row):
    '''
    INPUT: dict with "info" column/key
    OUTPUT: dict

    Removes all non-numeric values from dictionary and returns the updated dictionary. Intended for use in pd.DataFrame.apply().
    '''
    inf = one_row['info'].copy()
    bad_keys = []
    for key, val  in inf.iteritems():
        if not isinstance(val, numbers.Number):
            bad_keys.append(key)

    for key in bad_keys:
        inf.pop(key)

    return inf

def get_one_column_from_dict(one_row, col_name, key):
    '''
    INPUT: single row of DataFrame, column name with dict values, particular key of this dictionary
    OUTPUT: value

    Searches in a dict-valued column for a particular key, then returns the value corresponding to this key. Intended for use in pd.DataFrame.apply() to create new columns from dict-valued columns.
    '''
    inf = one_row.copy()[col_name]
    result = None
    if key in inf:
        result = inf[key]
    return result


def clean_up_info_column(one_row):
    '''
    INPUT: single row of DataFrame
    OUTPUT: dict

    Converts the scraped *string* values of particular keys in the dict-valued columns 'info' into numeric values where applicable. Uses regular expressions to interpret price and date information.
    '''

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


def grid_search(sf):
    '''
    INPUT: ratings SFrame
    OUTPUT: Ranking Factorization Recommender model

    Uses GraphLab's grid search functionality to optimize parameter choices for a RankingFactorizationRecommender model. Cross-validates using five folds. Returns a model trained on the entire SFrame.
    '''
    folds = graphlab.cross_validation.KFold(sf, 5)

    params = { 'num_factors': [90],'linear_regularization': [1e-6, 2e-6], 'regularization': [5e-4, 7e-4], 'user_id': ['user_name'], 'item_id' : ['item_id'], 'target': ['score'], 'side_data_factorization': [False], 'max_iterations': [80], 'solver':['auto'], 'ranking_regularization':[0.06, 0.05], 'unobserved_rating_value':[79], 'num_sampled_negative_examples': [8], 'verbose': ['True']}

    job = graphlab.grid_search.create(folds, graphlab.graphlab.recommender.ranking_factorization_recommender.create,  params)

    best_params = job.get_best_params()
    best_params['observation_data'] = sf
    # best_params['item_data'] = item_data
    best_params['max_iterations'] = 225

    model = graphlab.recommender.ranking_factorization_recommender.create(**best_params)

    return model


def convert_to_ratings_df(df):
    '''
    INPUT: scraped DataFrame
    OUTPUT: ratings DataFrame

    Takes a DataFrame with a column 'ratings' in dictionary form and converts it into a DataFrame with each user, item interaction occupying its own row.
    '''

    vals = list(df['ratings'].values)
    flattened = [item for sublist in vals for item in sublist]

    rdf = pd.DataFrame(np.asarray(flattened), columns = ['item_id', 'user_name', 'score'])
    rdf['item_id'] = rdf['item_id'].astype(int)
    rdf['score']   = rdf['score'].astype(float)
    return rdf



def throw_out_unrated_items(rdf, ratings, item_cutoff = 0 , user_cutoff = 0):
    '''
    INPUT: scraped DataFrame, ratings DataFrame, ratings-per-item and ratings-per-user cutoff
    OUTPUT: subsets of original DataFrames meeting given cutoffs
    '''

    g = ratings.groupby('item_id').count()
    valid_ids = g[ g['user_name'] > item_cutoff ].index

    ndf = rdf[rdf['id'].isin(valid_ids)]
    nratings = ratings[ratings['item_id'].isin(valid_ids)]

    g2 = nratings.groupby('user_name').count()
    valid_users = g2[ g2['item_id'] > user_cutoff ].index

    nratings2 = nratings[nratings['user_name'].isin(valid_users)]

    return ndf, nratings2



def extract_additional_columns(rdf):
    '''
    INPUT: scraped DataFrame
    OUTPUT: DataFrame with additional columns extracted from original dictionary columns.
    '''

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


def load_data():
    '''
    Loads the scraped data into a DataFrame and builds a model using default parameters.
    These objects are saved to a location for use by the app.
    '''
    print 'Loading data...'

    with open('data/wb_whiskey_data.pkl') as f:
        df = pickle.load(f)

    with open('data/wb_whiskey_data_part2.pkl') as f:
        df2 = pickle.load(f)

    with open('data/wb_whiskey_data_part3.pkl') as f:
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

    item_cutoff = 9
    user_cutoff = 7
    ndf, nratings = throw_out_unrated_items(rdf,ratings,item_cutoff,user_cutoff)

    ratings = nratings
    rdf = ndf

    with open('data/wb_whiskey_photo_data.pkl') as f:
        photo_df = pickle.load(f)

    photo_df['id'] = photo_df['id'].apply(int)
    photo_df['brand'] = photo_df['brand'].apply(lambda x: x.replace('\n', ''))

    rdf = pd.merge(left = rdf, right = photo_df, how = 'left', left_on='id', right_on = 'id')

    rdf['null_photo_url'] = rdf['photo_url'].isnull()
    rdf['photo_url'].fillna('https://static.whiskybase.com/storage/whiskies/default/normal.png' , inplace = True)
    item_info = rdf[['id', 'price', 'scotch_bool', 'united_states_bool', 'single_malt_bool', 'blend_bool']].rename(index = str, columns = {'id':'item_id'})

    print 'Building model...'

    sf = graphlab.SFrame(ratings)

    best_params = {'item_id': 'item_id', \
     'linear_regularization': 1e-06, \
     'max_iterations': 125, \
     'num_factors': 70, \
     'num_sampled_negative_examples': 8, \
     'ranking_regularization': 0.05, \
     'regularization': 0.0005, \
     'side_data_factorization': False, \
     'solver': 'auto', \
     'target': 'score', \
     'unobserved_rating_value': 78, \
     'user_id': 'user_name', \
     'verbose': 'True'}

    best_params['observation_data'] = sf
    best_params['max_iterations'] = 225
    model = graphlab.recommender.factorization_recommender.create(**best_params)

    return rdf, ratings, item_info, model


def get_item_name_from_url(item_link):
    '''
    INPUT: link to a whiskey page
    OUTPUT: item name
    '''
    res = None
    pos = item_link.rfind('/')
    if pos is not None:
        res = item_link[pos+1:]
    return res


def textual_analysis(rdf, ngram_range = (1,1), max_df = 0.50, min_df = 20, max_features = 800, learning_decay = 0.7, learning_offset = 10, max_iter = 80, n_topics = 6):
    '''
    INPUT: DataFrame with "notes" columns, parameters for LDA topic modeling
    OUTPUT: TF-vectorizer, item-to-topic and word-to-topic matrices

    Uses Latent Dirichlet Allocation to assign topics based on the text content of the "notes" field. This topic decomposition will eventually be used to compute similarity scores between items.
    '''
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation) )
    replace_numbers     = string.maketrans( '0123456789', ' '*10)

    documents = rdf['notes']
    documents = documents.apply(lambda x: unidecode.unidecode(x).replace('\n', '  ').lower().translate(replace_punctuation).translate(replace_numbers)).values

    vectorizer = CountVectorizer(stop_words = 'english', ngram_range = ngram_range, max_df = max_df, tokenizer = word_tokenize, preprocessor = WordNetLemmatizer().lemmatize , max_features = max_features)
    tf = vectorizer.fit_transform(documents).todense()

    # class sklearn.decomposition.LatentDirichletAllocation(n_topics=10, doc_topic_prior=None, topic_word_prior=None, learning_method=None, learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1, total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=1, verbose=0, random_state=None)

    lda = LatentDirichletAllocation(n_topics = n_topics, learning_decay = learning_decay,learning_offset = learning_offset,  max_iter = max_iter, n_jobs = -1, verbose = 1)
    W = lda.fit_transform(tf)
    H = lda.components_

    Y = pdist(W,'cosine')
    similarity_matrix = 1.0 - squareform(Y)

    return vectorizer, W, H, similarity_matrix

# def convert_similarity_matrix_into_dataframe(similarity_matrix, rdf):
#     for idx, row in enumerate(similarity_matrix):
#         item_id = rdf.loc[idx]['id']
#         for other_idx, similarity_score in enumerate(row):
#             other_item_id = rdf.loc[other_idx]['id']
#             row_to_add = {'item_id': item_id, 'other_item_id': other_item_id, 'score': similarity_score}



def print_topics(rdf, vectorizer, W,H, display_words = True, display_articles = True, words_to_show = 25, articles_to_show = 5):
    '''
    INPUT: a vectorizer, item-to-topic matrix, word-to-topic matrix
    OUPUT: None

    Prints the most important words per topic and items (called articles here) per topic. Provides a sanity check on the quality of the topics discovered during LDA.
    '''
    features_list = vectorizer.get_feature_names()
    H_tword_topics = np.asarray(np.argsort(H,axis=1)[:])[:,(-1)*(words_to_show):]
    W_articles = np.asarray(np.argsort(W.T,axis = 1)[:])[:,(-1)*(articles_to_show):]

    for index, (i1, i2) in enumerate(zip(H_tword_topics, W_articles)):
        print 'Topic number: {}'.format(index)
        names =[]
        topic_list =[]
        if display_words:
            for j in i1:
                topic_list.append(features_list[j])
            print 'Topic {} consists of the following words: '.format(index)
            print topic_list
        print '****'
        if display_articles:
            for j in i2:
                names.append(rdf.brand_and_name[j])
            print 'Topic {} consists of the following items: '.format(index)
            print names
            print '********'*3

    return None



if __name__ == "__main__":

    pass
    # rdf, ratings, item_info, model = load_data()

    # rdf['brand_and_name'] = rdf.apply(lambda row: (row.null_photo_url == 1)*get_item_name_from_url(row.url) + (row.null_photo_url == 0)*( (row.brand + ' - ' ) + row['name'] ), axis = 1)

    # sf = graphlab.SFrame(ratings)
    # item_data = graphlab.SFrame(item_info)
    #
    # model.get_similar_items([15797], k = 30).print_rows(30)
    # model.coefficients['item_id'][0]['factors']
