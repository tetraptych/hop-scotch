import pandas as pd
import numpy as np
import numbers
import decimal
import cPickle as pickle
import re
import unidecode
import itertools
import random, string
from scipy.spatial.distance import pdist, squareform
import os
import pip

b   = os.environ['GL_KEY']
a = os.environ['GL_EMAIL']

install_string = 'https://get.graphlab.com/GraphLab-Create/2.1/{}/{}/GraphLab-Create-License.tar.gz'.format(a, b)
pip.main(['install', '--upgrade', install_string])

import graphlab


def get_rdf():
    '''
    Loads the DataFrame with item and rating information. Used by app.py to filter recommendations.
    '''
    with open('data/whiskey_data_final.pkl') as f:
        rdf = pickle.load(f)

    return rdf


def get_model():
    '''
    Loads the GraphLab Ranking Factorization Recommender model.
    '''
    model = graphlab.load_model('model')
    return model

def get_similarity_matrix():
    '''
    Loads the LDA topic decomposition and returns both the loaded matrix and the derived matrix of similarity scores.
    '''
    W = np.load('data/topic_decomposition.npy')
    Y = pdist(W,'cosine')
    similarity_matrix = 1.0 - squareform(Y)
    return W, similarity_matrix


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


def random_word(length, source = string.lowercase):
    '''
    INPUT: int, list
    OUTPUT: string

    Generates a random word from the given source with the given length. We'll use this to generate random temporary user names.
    By default, the available symbols are lowercase letters.
    '''
    return ''.join(random.choice(source) for i in range(length))


# def profile_input(user_name):
#     '''
#     INPUT: WhiskeyBase user name OR profile URL
#     OUTPUT: item list, score list of actual ratings
#
#     Scrapes the given profile for ratings data and returns items, scores in list format.
#     '''
#     session = dryscrape.Session()
#
#     # Convert the user name to an acceptable format to be used in a URL...
#     # OR detect that a URL was passed as input and do nothing.
#     user_name = user_name.lower().strip()
#
#     if ' ' in user_name:
#         user_name = user_name.replace(' ', '-')
#     if '/profile/' in user_name:
#         url = user_name
#         if '/lists/ratings' not in url:
#             if url[-1] != '/':
#                 url = url + '/lists/ratings'
#             else:
#                 url = url + 'lists/ratings'
#     else:
#         url = 'https://www.whiskybase.com/profile/{}/lists/ratings'.format(user_name)
#
#     # Visit the URL and extract the information
#     response = None
#
#     while response == '' or response is None:
#         session.visit(url)
#         response = session.body()
#
#     soup = BeautifulSoup(response, 'lxml')
#     body = soup.find('tbody')
#     item_list  = []
#     score_list = []
#
#     if body is not None:
#         rows = body.find_all('tr')
#
#         for row in rows:
#             link = row.find('a')
#
#             if link is not None:
#                 if 'href' in link.attrs:
#                     item_link = link['href']
#                     item_list.append(get_item_id_from_url(item_link))
#
#             info_list = row.find_all('td')
#             if info_list is not None:
#                 if info_list != []:
#                     rating = info_list[-1]
#                     if rating is not None:
#                         score_list.append(float(rating.text))
#
#     # Reset the session for future use
#     session.reset()
#     del session
#
#     return item_list, score_list
#

def get_item_id_from_url(item_link):
    '''
    INPUT: link to a whiskey page
    OUTPUT: item ID (integer) of the whiskey with that URL
    '''
    res = re.search('\d+', item_link)
    if res is not None:
        res = int(res.group(0))
    return res


def recommend(model, item_list, score_list = None, cutoff_size = 4, rdf = None, similarity_matrix = None ):
    '''
    INPUT: list of items, list of scores, size at which to use model directly instead of similarity scores
    OUTPUT: list of item IDs matching the filter terms sorted according to predicted rating
    '''
    if score_list is None or score_list == []:
        score_list = list(100*np.ones(len(item_list)))
        if len(score_list) != len(item_list):
            score_list = list(100*np.ones(len(item_list)))

    model_weight = float(len(item_list)) / cutoff_size
    # Modality 1: too few ratings, use similarity
    if len(item_list) < cutoff_size:
        recs = recommend_on_similarity(model, item_list, score_list, rdf = rdf, similarity_matrix = similarity_matrix, model_weight = model_weight)

    # Modality 2: enough ratings to use model directly
    if len(item_list) >= cutoff_size:
        recs = recommend_from_model(model, item_list, score_list)

    return recs


def filter_results(recommendations, rdf, type_ = 'any', minprice = 0, maxprice = 100000, percentile_min = 25, num_to_show = 30):
    '''
    INPUT: DataFrame or SFrame containing item IDs, dictionary of search specifications
    OUTPUT: DataFrame containing item IDs matching the specifications

    Finds which items from a given list match the given specifications and returns only the matching ones.
    '''

    if 'predicted_rating' in recommendations.columns:
        lower_bound     = np.percentile(recommendations['predicted_rating'], percentile_min)
        recommendations = recommendations[recommendations['predicted_rating'] > lower_bound]

    full_recommendations = pd.merge(left = recommendations, right = rdf, how = 'inner', left_on='item_id', right_on = 'id')

    relevant_cols = full_recommendations.columns
    # ['item_id', 'id', 'url', 'category', 'price', 'district', 'scotch_bool', 'united_states_bool', 'canada_bool', 'japan_bool', 'ireland_bool', 'single_malt_bool', 'blend_bool', 'brand', 'name', 'photo_url']

    full_recommendations = full_recommendations[relevant_cols]

    filtered_recs = full_recommendations[(full_recommendations['price'] < maxprice) & (full_recommendations['price'] > minprice)]

    if type_ != 'any':
        col = None
        if type_ == 'scotch':
            col = 'scotch_bool'
        elif type_ == 'bourbon':
            col = 'united_states_bool'
        elif type_ == 'ireland':
            col = 'ireland_bool'
        elif type_ == 'canadian':
            col = 'canada_bool'
        elif type_ == 'japanese':
            col = 'japan_bool'
        filtered_recs = filtered_recs[filtered_recs[col] == 1]

    return filtered_recs[:num_to_show]


def recommend_on_similarity(model, item_list, score_list = None, rdf = None, similarity_matrix = None, model_weight = 0.5):
    '''
    INPUT: list of rated items, list of ratings, number of items to return
    OUTPUT: list of recommendations in sorted order

    Finds items similar to the ones in the given list (weighted according to the input ratings).
    Returns the items that best match the given ones, noting that scores of 50 or below result in negative contributions to the overall score.

    If a similarity matrix is given, also uses cosine similarity in the topic space (discovered by LDA) to find similar items. The balance between the model and the LDA topics is governed by the model_weight parameter.

    This method is most useful when there are only a handful of rated items, preventing the model from giving recommendations directly. Otherwise, use the recommend function instead.
    '''
    if score_list is None or score_list == []:
        score_list = list(100*np.ones(len(item_list)))

    weights = [float(score) - 50 for score in score_list]
    total_weight = sum(weights)

    weights = [ float(w) / total_weight for w in weights]

    sf = model.get_similar_items([], k = 1)
    sf['weighted_score'] = []

    for item, weight in zip(item_list, weights):
        temp_sf = model.get_similar_items([item], k = model.num_items)
        temp_sf['weighted_score'] = weight*temp_sf['score']
        sf = sf.append(temp_sf)

    out_sf = sf.groupby(key_columns = 'similar',  operations = {'combined_score': graphlab.aggregate.SUM('weighted_score')})
    out = out_sf.to_numpy()
    out_df = pd.DataFrame(out[np.argsort(out[:, 1])[::-1]], columns = ['item_id', 'score'])
    out_df['item_id'] = out_df['item_id'].astype(int)
    out_df['score']   = out_df['score'].astype(float)

    ## Rescale so that score is between 0 and 1:
    min_score = np.min(out_df['score'])
    max_score = np.max(out_df['score'])
    out_df['score'] = (out_df['score'] - min_score) / (max_score - min_score)

    if similarity_matrix is not None:

        weights = np.matrix([float(score) - 50 for score in score_list])
        total_weight = np.sum(weights)
        weights = weights / total_weight

        indexes = list(rdf[rdf['id'].isin(item_list)].index)
        weighted_scores = np.dot(weights,similarity_matrix[indexes])
        df = pd.DataFrame(weighted_scores.T, columns = ['score']).sort_values('score', ascending = False)
        df.reset_index(inplace = True)
        df['item_id'] = df['index'].map(lambda x: rdf['id'].loc[x])
        df = df[~(df['item_id'].isin(item_list))]
        min_score = np.min(df['score'])
        max_score = np.max(df['score'])
        df['score'] = (df['score'] - min_score) / (max_score - min_score)

        mdf = pd.merge(left = df, right = out_df, how = 'left', on = 'item_id', suffixes = ('_lda', '_model'))
        mdf['score'] = model_weight * mdf['score_model'] + (1.0-model_weight)*mdf['score_lda']
        mdf = mdf[['item_id', 'score']]
        mdf = mdf.sort_values('score', ascending = False)
        out_df = mdf

    return out_df



def recommend_from_model(model, item_list, score_list):
    '''
    INPUT: list of item IDs and associated scores
    OUTPUT: sorted DataFrame of item IDs, scores

    Makes recommendations using the model directly. Suitable in the case when there are a sufficient number of user ratings supplied.
    '''
    # Generate a random user name
    user = random_word(10)

    # Create an empty DataFrame to store the ratings data
    user_df            = pd.DataFrame(columns = ['item_id', 'user_name','score'])
    user_df['item_id'] = user_df['item_id'].astype(int)
    user_df['score']   = user_df['score'].astype(float)

    # Fill the DataFrame with the given ratings
    for item, score in zip(item_list, score_list):
        d = {'item_id': item, 'user_name': user, 'score': score}
        one_row_df = pd.DataFrame.from_dict([d])
        user_df = user_df.append(one_row_df, ignore_index = True)

    user_sf = graphlab.SFrame(user_df)

    # Use the model to recommend other items.
    # Translate results from SFrame --> DataFrame
    user_recs = model.recommend_from_interactions(user_sf,k = model.num_items, verbose = False)
    recommended_ids = pd.DataFrame(user_recs.to_numpy(), columns = ['item_id', 'predicted_rating', 'item_rank'])

    names = ['item_id', 'predicted_rating', 'item_rank']
    types = [int, float, int]

    for col_name, dtype in zip(names, types):
        recommended_ids[col_name] = recommended_ids[col_name].astype(dtype)

    return recommended_ids


def generate_dissimilarity_scores(model, item_list, score_list):
    '''
    INPUT: list of item IDs and associated scores
    OUTPUT: sorted DataFrame of item IDs, similarity scores

    Finds the mean factor vector for the given item IDs then returns all items sorted by the similarity to this mean vector.
    '''
    W = model.coefficients['item_id']
    num_factors = model.num_factors

    d = ((W['factors'])[W['item_id'].is_in(item_list)])

    for index, row in enumerate(d):
        if index == 0:
            arr = np.array(row)[:,np.newaxis]
            arr = arr.reshape(1,num_factors)
        else:
            arr1 = np.array(row)[:,np.newaxis].reshape(1,num_factors)
            arr = np.vstack( (arr, arr1 ) ).reshape(index+1,num_factors)

    mean_vec = np.mean(arr, axis = 0)

    Wdf = W.to_dataframe()
    Wdf.set_index('item_id', inplace = True)

    Wdf['mean_similarity'] = Wdf.apply(lambda row: np.dot(row['factors'], mean_vec), axis = 1)
    Wdf.sort_values(by = 'mean_similarity', inplace = True, ascending = False)

    return Wdf['mean_similarity']
