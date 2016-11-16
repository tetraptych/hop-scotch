import dryscrape
from bs4 import BeautifulSoup
from pymongo import MongoClient
import requests
from time import sleep
import numpy as np
import cPickle as pickle
import pandas as pd
import re
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string


n_max = 28
american_urls = ['https://www.masterofmalt.com/search/#search=&page={}&sort=price&direction=asc&targettemplates=1759&countries=14'.format(n) for n in range(1, n_max+1) ]

n_max = 23
other_urls = ['https://www.masterofmalt.com/search/#search=&page={}&sort=userrating&direction=desc&targettemplates=1759&countries=12,4,7,13,23,76,77'.format(n) for n in range(1, n_max + 1) ]
# n_max = int(2500/25) + 1
# n_max = 97

in_stock_urls = ['https://www.masterofmalt.com/search/#search=&page={}&sort=_score&direction=desc&targettemplates=1759&stocklevel=1'.format(n) for n in range(1, n_max+1) ]

in_stock_urls = american_urls

n_max = int(10972./25) + 1
urls = ['https://www.masterofmalt.com/search/#search=&page={}&sort=_score&direction=desc&targettemplates=1759'.format(n) for n in range(1,n_max+1) ]

content_list = []


def pickle_object(something, name):
    with open('{}.pkl'.format(name), 'w') as f:
        pickle.dump(something, f)
    return None



#
# with open('in_stock_urls.txt', 'a') as f:
#     for page in in_stock_urls:
#         f.write(page+'\n')

# for i, url in enumerate(in_stock_urls):
#     print 'Scraping page number {}...'.format(i+1)
#     html_page = url
#     r = requests.get(html_page)
#     content_list.append(r.content)
#     sleep(5)

# session = dryscrape.Session()
# whiskey_dict = {}
def scrape_some_search_results(whiskey_dict = {}, urls = other_urls, end_page = 1000):
    if whiskey_dict == {}:
        start_page = -1
    else:
        start_page = sorted(whiskey_dict.keys(),key= lambda x: x[0], reverse = True)[0][0]

    for i, url in enumerate(urls):
        if i > start_page and i < end_page:
            print 'Scraping page number {}...'.format(i+1)
            sleep(2)
            session = dryscrape.Session()
            sleep(3)
            session.visit(url)
            sleep(10)
            print '\tURL visited!'
            response = session.body()
            soup = BeautifulSoup(response, 'lxml')
            sleep(3)
            print '\t\tSoup\'s up!'
            mydivs = soup.find_all("div", class_="boxBgr product-box-wide h-gutter")
            sleep(1)
            d = {}
            print '\t\t\tKeep looking...'
            for j, res in enumerate(mydivs):
                link = res.find('a').get('href')[2:]
                d[(i,j)] = link
            sleep(1)
            whiskey_dict.update(d)
            print '\t\t\tThere are now {} unique values!'.format(len(np.unique(whiskey_dict.values())))
    return whiskey_dict


def scrape_one_page_of_search_results(page_num, whiskey_dict = {}, urls = other_urls, session = dryscrape.Session()):
    i = page_num
    url = urls[i]
    print 'Scraping page number {}...'.format(i+1)
    sleep(1)
    session.visit(url)
    print '\tURL visited!'
    sleep(1)
    response = session.body()
    sleep(1)
    soup = BeautifulSoup(response, 'lxml')
    sleep(1)
    print '\t\tSoup\'s up!'
    mydivs = soup.find_all("div", class_="boxBgr product-box-wide h-gutter")
    sleep(1)
    d = {}
    print '\t\t\tKeep looking...'
    for j, res in enumerate(mydivs):
        link = res.find('a').get('href')[2:]
        d[(i,j)] = link
    whiskey_dict.update(d)
    sleep(1)
    num_scraped = len(np.unique(whiskey_dict.values()))
    print '\t\t\t\tThere are now {} unique values!'.format(num_scraped)
    session.reset()
    return whiskey_dict, num_scraped


def scrape_all_search_results(whiskey_dict = {},  how_many_pages = 1, max_strikes = 4):
    if whiskey_dict == {}:
        last_num_scraped = 0
        page_num = 0
        start_page = 0
        print 'Starting from page {}'.format(page_num+1)

    else:
        last_num_scraped = len(np.unique(whiskey_dict.values()))
        start_page = sorted(whiskey_dict.keys(),key= lambda x: x[0], reverse = True)[0][0]
        page_num = start_page + 1
        print 'Picking up from page {}'.format(page_num+1)

    strikes = 0

    while ( page_num < start_page + how_many_pages + 1 ):
        whiskey_dict, num_scraped = scrape_one_page_of_search_results(page_num, whiskey_dict)

        retry = (num_scraped <= last_num_scraped + 5)

        if retry:
            strikes += 1
            print '\n*****LET\'S TRY THAT PAGE AGAIN...STRIIIIIIIKE {}*****\n'.format(strikes)
            if strikes == max_strikes:
                return whiskey_dict
            sleep(8 * strikes)

        if not retry:
            page_num += 1
            last_num_scraped = num_scraped
            pickle_object(whiskey_dict,'other_whiskey_dict')
            strikes = 0

    return whiskey_dict


def write_urls_from_dictionary(dictionary, filename = 'whiskey_pages.txt'):
    with open('whiskey_pages.txt', 'w') as f:
        for page in whiskey_dict.values():
            f.write(page+'\n')
    return 'Complete!'


def load_urls_from_file(filename = 'whiskey_pages.txt'):
    urls = []
    with open(filename, 'r') as f:
        for line in f:
            urls.append(line)
    urls = ['https://' + str(url) for url in urls]
    urls_len = len(urls)
    urls = sorted(urls)
    return urls

#mydivs = soup.findAll("div", { "class" : "stylelistrow" })
url = 'https://www.masterofmalt.com/whiskies/glenfarclas-25-year-old-whisky/?srh=1'
url = 'https://www.masterofmalt.com/whiskies/ardbeg/ardbeg-uigeadail-whisky/?srh=1'


def scrape_one_whiskey_page(url, session = dryscrape.Session() ):

    info = {}
    info['url'] =  url
    digits = [str(x) for x in range(11)]
    digits.append('.')

    session.visit(url)
    print('\tURL visited!')
    print('\t' + url)
    # sleep(3)
    response = session.body()
    soup = BeautifulSoup(response, 'lxml')

    pre_agg_rating  = soup.find("div", class_='product-user-rating')
    has_reviews = (pre_agg_rating is not None)

    if has_reviews:
        agg_rating = pre_agg_rating.find_all('div')
        has_reviews = (agg_rating is not None)

    if has_reviews:
        for i, item in enumerate(agg_rating):
            if i == 0:
                if 'title' in item.attrs:
                    info['avg_rating'] = float(''.join([ch for ch in item['title'] if ch in digits]))
            if i == 1:
                if 'content' in item.attrs:
                    info['num_ratings'] = int(item['content'])

    pre_details = soup.find("div", class_ ='expandContainer kv-list')
    details = pre_details.find_all('span')

    for i, detail in enumerate(details):
        if i % 2 == 0:
            field = detail.text.lower()
        if i % 2 == 1:
            value = detail.text
            info[str(field)] = value

    temp = soup.find('div', class_ = 'priceDiv')
    if temp is not None:
        info['price'] = float(''.join( [ ch for ch in temp.find('span').text if ch in digits]))

    temp = soup.find_all('div', id = 'reviewslist')[0]
    reviews = temp.find_all('div', class_ = 'userReviewBlock')

    review_list = []
    total_score = 0.0

    for i, rev in enumerate(reviews):
        d = {}
        rating_info = rev.find("div")
        if rating_info is not None:
            d['rating'] = float(rating_info.find("meta", itemprop = 'ratingValue').attrs['content'])
            d['rev_text'] = rev.find('p').text
            d['rev_title'] = rev.find('span').text
            total_score += d['rating']
            review_list.append(d)

    if total_score > 0 and has_reviews:
        info['actual_avg_rating'] = total_score / info['num_ratings']

    info['review_list'] = tuple(review_list)
    info['full_soup'] = soup

    class_name = 'pageCopy productPageBottlingNote'
    info['bottling_note'] = look_for_anything(soup, 'div' , 'class', class_name)

    id_name = 'ContentPlaceHolder1_ctl00_ctl03_TastingNoteBox_ctl00_productTastingNote2'
    info['tasting_note'] = look_for_anything(soup, 'div' , 'id', id_name)

    id_name = 'ContentPlaceHolder1_pageH1'
    info['name'] = look_for_anything(soup, 'h1', 'id', id_name)
    session.reset()

    return info



# data = []
# # urls = whiskey_dict.values()
#

#df = pd.read_csv('other_whiskey_data.csv', encoding = 'utf-8')

def main(df):

    with open('unvisited_urls.pkl') as f:
            urls = pickle.load(f)

    # urls = load_urls_from_file()
    url = urls[0]
    # urls.pop(181)
    url_len = len(urls)

    if df is None:
        df = pd.DataFrame()

    # else:
    #     df = df.drop(df.columns[1], axis = 1)
    columns = list(df.columns)
    if 'full_soup' in columns:
        columns.remove('full_soup')

    for i, url in enumerate(urls):
        if i >= len(df):
            print 'Scraping page number {}...of {}'.format(i+1, url_len + 1)
            info = scrape_one_whiskey_page(url)
            one_row_df = pd.DataFrame.from_dict([info])
            df = df.append(one_row_df, ignore_index = True)
            print 'Modulo 1: ', i % 1


            if i % 1 == 0:
                print '>>>>>>>>>>>SAVING PROGRESS>>>>>>>>>>>'
                print '>>>>>>>>>>>SAVING PROGRESS>>>>>>>>>>>'
                print '>>>>>>>>>>>SAVING PROGRESS>>>>>>>>>>>'

                columns = list(df.columns)
                if 'full_soup' in columns:
                    columns.remove('full_soup')

                df.to_csv('other_whiskey_data.csv', encoding = 'utf-8')
                df.to_csv('other_whiskey_data_partial.csv', encoding = 'utf-8', columns = columns)

                break

        # df.to_csv('other_whiskey_data.csv', encoding = 'utf-8')
        # df.to_csv('other_whiskey_data_partial.csv', encoding = 'utf-8', columns = columns)


    return df


def soupify(df, column_name = 'full_soup'):
    df[column_name] = df[column_name].apply(lambda x: BeautifulSoup(x, 'lxml') )
    return df

def look_for_a_class(x, class_name):

    new_feature = x.find('div',class_=class_name)
    if new_feature is not None:
        return new_feature.text
    else:
        return None

def look_for_an_id(x, id_name):

    new_feature = x.find('div',id=id_name)
    if new_feature is not None:
        return new_feature.text
    else:
        return None

def look_for_anything(soup, thing_type, other_thing_type, other_thing_name):
    if other_thing_type == 'class':
        new_feature = soup.find(thing_type, class_=other_thing_name)
    if other_thing_type == 'id':
        new_feature = soup.find(thing_type, id=other_thing_name)

    if new_feature is not None:
        return new_feature.text
    else:
        return None


def clean_df(df):

    class_name = 'pageCopy productPageBottlingNote'
    df['bottling_note'] = df['full_soup'].apply(lambda x: look_for_a_class(x, class_name) )

    id_name = 'ContentPlaceHolder1_ctl00_ctl03_TastingNoteBox_ctl00_productTastingNote2'
    df['tasting_note'] = df['full_soup'].apply(lambda x: look_for_an_id(x, id_name) )

    id_name = 'ContentPlaceHolder1_pageH1'
    df['name'] = df['full_soup'].apply(lambda x: look_for_anything(x, 'h1', 'id', id_name))

    return df
    # actual name
    # other info I can scrape from the soup

def make_extra_cols():
    df = pd.read_csv('whiskey_data.csv', encoding = 'utf-8')
    # df.drop('Unnamed: 0.1', axis=1, inplace=True)
    df.drop('Unnamed: 0', axis=1, inplace=True)

    soup_df = pd.DataFrame(df[['full_soup','url']])
    del df

    soup_dfs = np.array_split(soup_df, 10)

    df = pd.DataFrame()
    for i, sdf_ in enumerate(soup_dfs):
        sdf = soupify(sdf_)
        sdf = clean_df(sdf)
        del sdf['full_soup']
        sdf.to_csv('extra_cols_{}.csv'.format(i), encoding = 'utf-8')
        df = df.append(sdf, ignore_index = True)

    df.to_csv('extra_cols.csv', encoding = 'utf-8')
    return df

def load_data():
    df  = pd.read_csv('whiskey_data_partial.csv', encoding = 'utf-8')

    df.drop_duplicates(subset = ['url'], inplace = True)
    df.drop(['tasting_note','bottling_note', 'Unnamed: 0'], axis = 1, inplace = True)

    edf = pd.read_csv('extra_cols.csv', encoding = 'utf-8')
    edf.drop_duplicates(subset = ['url'], inplace = True)
    edf.drop(['Unnamed: 0'], axis = 1, inplace = True)

    rdf = pd.merge(df, edf, how='inner', on='url')
    rdf['url'] = rdf['url'].apply(lambda x: x.replace('\n', ''))

    odf = pd.read_csv('other_whiskey_data_partial.csv', encoding = 'utf-8')
    odf.drop_duplicates(subset = ['url'], inplace = True)
    odf = odf.dropna(subset = ['price'])
    odf.drop(['Unnamed: 0'], axis = 1, inplace = True)

    rdf = rdf.append(odf, ignore_index=True)

    rdf['num_ratings'].fillna(0, inplace = True)
    rdf.drop_duplicates(subset = ['url'], inplace = True)

    rdf = rdf[~(rdf['volume'].isnull())]
    rdf['volume'] = rdf['volume'].apply(lambda x: x[:-2]).astype(float)
    rdf['unit_price'] = rdf['volume'] / rdf['price']

    rdf.reset_index(inplace=True)
    rdf.drop(['index'], axis = 1, inplace = True)

    rdf['review_list'] = rdf['review_list'].apply(lambda x: ''*(x == '()') + (x != '()')*x )
    rdf['review_list'] = rdf['review_list'].apply(lambda x: re.split('(?<=}),',  x))
    rdf['has_reviews'] = (rdf['review_list'] != '' ).astype(bool)

    rdf['style'] = rdf['style'].apply(lambda x: x.replace('Whisky', 'Whiskey'))
    rdf['country'] = rdf['country'].apply(lambda x: x.replace('Whisky', 'Whiskey'))

    rdf['alcohol'] = rdf['alcohol'].apply(lambda x: float(x.strip('%')))

    rdf['bottling_note'] = rdf['bottling_note'].fillna('')
    rdf['tasting_note'] = rdf['tasting_note'].fillna('')
    rdf['bottling_note'] = rdf['bottling_note'].apply( lambda x: unidecode.unidecode(x).replace('\n', ' ').lower().translate(None, string.punctuation) )
    rdf['tasting_note'] = rdf['tasting_note'].apply( lambda x: unidecode.unidecode(x).replace('\n', ' ').lower().translate(None, string.punctuation) )

    rdf['bottling_tasting'] = rdf['bottling_note'] + rdf['tasting_note']


    return rdf





if __name__ == '__main__':

    with open('unvisited_urls.pkl') as f:
            unvisited_urls = pickle.load(f)

    df = pd.read_csv('other_whiskey_data.csv', encoding = 'utf-8')
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis = 1, inplace = True)
    if 'Unnamed: 0.1' in df.columns:
        df.drop(['Unnamed: 0.1'], axis = 1, inplace = True)


    # with open('america_whiskey_dict.pkl') as f:
    #         america_whiskey_dict = pickle.load(f)
    #
    # with open('other_whiskey_dict.pkl') as f:
    #         other_whiskey_dict = pickle.load(f)
    #
    # america_whiskey_list = america_whiskey_dict.values()
    # america_whiskey_list = ['https://' + str(url) for url in america_whiskey_list]
    # america_whiskey_list = sorted(america_whiskey_list)
    # other_whiskey_list = other_whiskey_dict.values()
    # other_whiskey_list = ['https://' + str(url) for url in other_whiskey_list]
    # other_whiskey_list = sorted(other_whiskey_list)
    #
    # new_whiskey_set = set(america_whiskey_list + other_whiskey_list)
    #
    # rdf = load_data()
    # visited_urls_set = set(rdf['url'].apply(lambda x: unidecode.unidecode(x.replace('\n', ''))))
    # unvisited_urls = list(new_whiskey_set - visited_urls_set)
    #
    # pickle_object(unvisited_urls,'unvisited_urls')
