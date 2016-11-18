from time import sleep
import numpy as np
import cPickle as pickle
import pandas as pd
import re
import unidecode
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.decomposition import NMF as skNMF
from scrappy import load_data
from sklearn.lda import LDA
from sklearn.decomposition import LatentDirichletAllocation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from itertools import combinations


replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))


def fit_transform_tfidf(documents, max_features = None):
    vectorizer = TfidfVectorizer(stop_words = 'english', tokenizer = word_tokenize, preprocessor = WordNetLemmatizer().lemmatize, max_features = max_features , lowercase = True)
    vectors = vectorizer.fit_transform(documents)
    return vectorizer, vectors


def vectorize(articles, ngram_range = (1,1), max_df = 1.0, min_df = 0, max_features = None):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    articles = [unidecode.unidecode(doc).translate(replace_punctuation) for doc in articles]

    vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = ngram_range, max_df = max_df, tokenizer = word_tokenize, preprocessor = WordNetLemmatizer().lemmatize , max_features = max_features)

    vectors = vectorizer.fit_transform(articles)

    return vectorizer, vectors


def find_topics(vectorizer, vectors, num_topics = 12, max_iter = 400, random_state = 1, alpha = 0):

    features_list = vectorizer.get_feature_names()
    k = num_topics
    sklearn_nmf = skNMF(k, solver = 'cd', max_iter = max_iter , random_state = random_state, alpha = alpha )
    W = sklearn_nmf.fit_transform(vectors.todense())
    H = sklearn_nmf.components_

    return W, H


def find_LDA_topics(vectorizer, vectors, num_topics = 12):

    lda = LatentDirichletAllocation(n_topics = num_topics)
    features_list = vectorizer.get_feature_names()
    W = lda.fit_transform(vectors.todense())
    H = lda.components_

    return W, H


def print_topics(vectorizer, W,H, display_words = True, display_articles = True, words_to_show = 25, articles_to_show = 5):
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
                names.append(rdf.name[j])
            print 'Topic {} consists of the following items: '.format(index)
            print names
            print '********'*3

    return None


def assign_topics(W,H):
    y = np.argmax(W, axis = 1)
    return y


def upsample_one_string(phrase, target_length = 2000):
    result = ''
    if phrase != '':
        while (len(result) < target_length ):
            result += phrase + ' *** '

    return result


def upsample_reviews(rdf, target_length = 2000):

    # rdf['num_reviews'] = rdf['review_list'].apply(lambda x: (x == []) * tuple() )
    rdf['has_reviews'] = rdf['review_list'].apply(lambda x: (x == [''] )*False + (x != [''] ) * True    )
    rdf['num_reviews'] = rdf['review_list'].apply(lambda x: (x == [''] )*0 + (x != [''] ) * len(x)    )
    rdf['review_string'] = rdf['review_list'].apply( lambda x: ' ; '.join(x))
    rdf['review_string'] = rdf['review_string'].apply(lambda x: x.replace('u\'rev_title\':', ' ').replace('u\'rev_text\':', ' ').replace('u\'rating\':', ' ').replace('u\'', ' '))
    rdf['review_string'] = rdf['review_string'].apply(lambda x: unidecode.unidecode(x).replace('\n', '  ').lower().translate(replace_punctuation) )

    rdf['review_length'] = rdf['review_string'].apply(len)

    rdf['upsampled_reviews'] = rdf['review_string'].apply(lambda x: upsample_one_string(x, target_length))


    return rdf


if __name__ == '__main__':
    rdf = load_data()

    # rdf['clean_reviews'] = rdf['review_list'].apply( lambda x: ' ; '.join(x))
    # rdf['clean_reviews'] = rdf['clean_reviews'].apply(lambda x: x.replace('u\'rev_title\':', ' ').replace('u\'rev_text\':', ' ').replace('u\'rating\':', ' ').replace('u\'', ' '))
    # replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    # rdf['clean_reviews'] = rdf['clean_reviews'].apply(lambda x: unidecode.unidecode(x).replace('\n', '  ').lower().translate(replace_punctuation) )

    rdf = upsample_reviews(rdf, target_length = 1)

    w_bottling_note = 300
    w_tasting_note = 300

    rdf['combined_text'] = (rdf['bottling_note']*(w_bottling_note) + rdf['tasting_note']*w_tasting_note  + rdf['upsampled_reviews'])
    rdf['combined_text'] = rdf['combined_text'].apply(lambda x: unidecode.unidecode(x).replace('\n', '  ').lower().translate(replace_punctuation) )

    articles = (rdf['combined_text']).values

    vectorizer, vectors = vectorize(articles, ngram_range = (1,1), max_df = 0.1, min_df = 250, max_features = 400)
    num_topics = 8

    Y = pdist(vectors.todense(),'cosine')
    dist_mat = squareform(Y)

    lda = False
    for num_topics in xrange(12,13):
        print '------------{} TOTAL TOPICS------------'.format(num_topics)

        W, H = find_topics(vectorizer, vectors, num_topics)
        if lda == True:
            W, H = find_LDA_topics(vectorizer, vectors, num_topics)

        rdf['cluster'] = assign_topics(W,H)
        g = rdf.groupby('cluster')

        print_topics(vectorizer, W,H, display_words = True, display_articles = False)
        print pd.crosstab(index = rdf.country, columns = rdf.cluster)
        print g.mean()

        for cluster in range(len(np.unique(rdf['cluster'])) ):

            curr = list(rdf.index[(rdf['cluster'] == cluster)])
            curr_dist_mat = dist_mat[curr].transpose()[curr].transpose()
            print np.mean(curr_dist_mat)

    # 18 isn't bad
    # 12 isn't bad!
    for cluster1 in range(len(np.unique(rdf['cluster'])) ):
        curr1 = list(rdf.index[(rdf['cluster'] == cluster1)])
        for cluster2 in range(len(np.unique(rdf['cluster'])) ):
            curr2 = list(rdf.index[(rdf['cluster'] == cluster2)])
            curr_dist_mat = dist_mat[curr1].transpose()[curr2].transpose()
            print cluster1, cluster2
            print np.mean(curr_dist_mat)


    # find mean - within cluster distance




    # pd.crosstab(index = rdf.country, columns = rdf.cluster)
    # pd.crosstab(index = rdf.region, columns = rdf.cluster)
    # pd.crosstab(index = rdf.avg_rating, columns = rdf.cluster)
    # pd.crosstab(index = rdf.distillery, columns = rdf.cluster)

    # pd.crosstab(index = rdf.distillery, columns = rdf.cluster).loc['Buffalo Trace']
    # pd.crosstab(index = rdf.distillery, columns = rdf.cluster).loc['Bulleit']
    # pd.crosstab(index = rdf.distillery, columns = rdf.cluster).loc['Famous Grouse']
    # pd.crosstab(index = rdf.distillery, columns = rdf.cluster).loc['Jim Beam']
    # pd.crosstab(index = rdf.distillery, columns = rdf.cluster).loc['Wild Turkey']


    # pd.crosstab(index = rdf['style'], columns = rdf.cluster)
    # blah = rdf.groupby('distillery').count()['url']
    # blah2 = list(np.argsort(blah.values)[::-1])
    # blah[blah2]
    # plt.scatter(rdf['unit_price'], rdf['actual_avg_rating'], c = rdf['cluster'] )
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(W[:,0], W[:,1], W[:,2], c = rdf['cluster'])

    #
    # rdf['cluster_lda'] = assign_topics(W_lda,H_lda)
    # print Counter(rdf['cluster_lda'])
    # g = rdf.groupby('cluster_lda')
    # g.mean()
