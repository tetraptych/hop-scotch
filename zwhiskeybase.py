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
import string

country_names = ['Scotland', 'United States', 'Japan', 'Ireland', 'Australia', 'Germany', 'UK', 'Netherlands', 'Sweden', 'Austria', 'France', 'Switzerland']
countries = ['191', '227', '108', '104', '14', '81', '226', '151', '206', '15', '74', '207']
pages = [12, 19, 2, 2, 2, 7, 1, 2, 1, 3, 3, 4]
pages2 = [range(1,n+1) for n in pages]
names_ids_pages = zip(country_names, countries, pages2)


brand_pages = [7, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
brand_pages2 = [range(1,n+1) for n in brand_pages]
names_ids_brand_pages = zip(country_names, countries, brand_pages2)

brand_search_urls = []
for name, id_, pages in names_ids_brand_pages:
    phrase = 'https://www.whiskybase.com/whiskies/brands?page=@@@&brandname%5B%5D=&country_id%5B%5D={}&sort=whiskies&direction=desc'.format(id_)
    for page in pages:
        brand_search_urls.append(phrase.replace('@@@','{}').format(page))


urls = []
for name, id_, pages in names_ids_pages:
    phrase = 'https://www.whiskybase.com/distilleries?page=@@@&name%5B%5D=&country_id%5B%5D={}&sort=name&direction=asc'.format(id_)
    for page in pages:
        urls.append(phrase.replace('@@@','{}').format(page))

urls = []
max_page = 14
for page in range(1,max_page+1):
    phrase =  'https://www.whiskybase.com/distilleries?page={}&name%5B%5D=&sort=whiskies&direction=desc'.format(page)
    urls.append(phrase)


def pickle_object(something, name):
    with open('{}.pkl'.format(name), 'w') as f:
        pickle.dump(something, f)
    return None


def scrape_one_search_page(page_num, urls, distillery_urls = [], session = dryscrape.Session() ):
    i = page_num
    url = urls[i]
    tries = 0
    print 'Scraping page number {}...'.format(i+1)
    response = None
    while (response == '' or response is None):
        session.visit(url)
        response = session.body()
        tries += 1
        sleep(5)
        if tries >= 4:
            session.reset()

    soup = BeautifulSoup(response, 'lxml')
    print '\t\tSoup\'s up!'
    links = soup.find_all('a')
    links2 = [link['href'] for link in links]

    links3 = filter(lambda link: '.com/distillery/' in link, links2)
    links4 = list(np.unique(links3))

    return links4


def scrape_best_distillery_urls(urls):
    session = dryscrape.Session()
    best_distillery_urls = []
    urls = urls
    for page_num in range(len(urls)):
        best_distillery_urls = best_distillery_urls + scrape_one_search_page(page_num, urls)

    pickle_object(best_distillery_urls, 'whiskeybase_best_distillery_urls')
    return best_distillery_urls


def scrape_distillery_urls(urls):
    session = dryscrape.Session()
    distillery_urls = []
    for page_num in range(len(urls)):
        distillery_urls += scrape_one_search_page(page_num, urls)

    pickle_object(distillery_urls, 'whiskeybase_distillery_urls')
    return distillery_urls


def scrape_brand_urls():
    session = dryscrape.Session()

    urls = brand_search_urls
    brand_ = []
    brand_urls = []
    for page_num in range(len(urls)):
        brand_urls.append(scrape_one_search_page(page_num, urls, []) )

    brand_urls = list(np.unique(sum(brand_urls, [])))

    pickle_object(brand_urls, 'whiskeybase_brand_urls')
    return brand_urls

def find_other_brand_pages():
    url = 'https://www.whiskybase.com/brand/81357?page=1sort=rating&direction=desc&highlight=none'
    tries = 0
    print 'Scraping page number {}...'.format(i+1)
    response = None
    while (response == '' or response is None):
        session.visit(url)
        response = session.body()
        tries += 1
        sleep(5)
        if tries >= 4:
            session.reset()

    soup = BeautifulSoup(response, 'lxml')
    links = soup.find_all('a')
    links2 = [link['href'] for link in links]



def scrape_distillery_page(url, session = dryscrape.Session() ):
    url = url + '?votes[]=1&sort=undefined&direction=undefined&highlight=none'

    # url = url + '?itemsforsale[]=1&votes[]=1&sort=undefined&direction=undefined&highlight=none'
    response = None
    tries = 0
    print '\t' + url

    while (response == '' or response is None):
        print '\tAttempt number: {}...'.format(tries)
        session.visit(url)
        response = session.body()
        tries += 1
        if (response == '' or response is None):
            sleep(max(5*tries, 20))
        if tries == 4:
            session.reset()

    print '\tURL visited!'
    soup = BeautifulSoup(response, 'lxml')
    print '\t\tSoup\'s up!'
    links = soup.find_all('a')

    links2 = [link.attrs for link in links]
    links3 = []
    for link in links2:
        if 'href' in link:
            links3.append(link['href'])

    links4 = filter(lambda link: '.com/whisky/' in link, links3)
    links5 = list(np.unique(links4))

    return links5


def scrape_all_whiskey_urls(session, actual_distillery_urls, actual_whiskey_urls_part2, actually_visited):
    # session = dryscrape.Session()

    for page_num, dist_url in enumerate(actual_distillery_urls):
        if dist_url not in actually_visited:
            print 'Scraping page number {}...'.format(page_num+1)
            res = scrape_distillery_page(dist_url, session)
            actual_whiskey_urls_part2 = actual_whiskey_urls_part2 + res
            actually_visited.append(dist_url)

            if page_num % 5 == 0:
                pickle_object(actual_whiskey_urls_part2, 'whiskeybase_actual_whiskey_urls_part2')
                pickle_object(actually_visited, 'whiskeybase_actually_visited_urls')
                session.reset()

                print 'ALWAYS SAVE YOUR WORK, SOMETIMES'
                print 'SOMETIMES SAVE YOUR WORK, ALWAYS'

    # with open('whiskeybase_whiskey_urls.pkl') as f:
    #     whiskey_urls = pickle.load(f)

    return actual_whiskey_urls, actually_visited


def scrape_one_whiskey_page(url):
    info = {}
    info['url'] =  url

    session.visit(url)
    print('\tURL visited!')
    print('\t' + url)
    response = session.body()
    soup = BeautifulSoup(response, 'lxml')

    info['full_soup'] = soup
    return info


def soups_up(df, whiskey_urls):

    if df is None:
        df = pd.DataFrame()

    url_len = len(whiskey_urls)

    for page_num, url in enumerate(whiskey_urls):
        if page_num >= len(df):
            print 'Scraping page number {}...of {}'.format(i+1, url_len + 1)
            info = scrape_one_whiskey_page(url)
            one_row_df = pd.DataFrame.from_dict([info])
            df = df.append(one_row_df, ignore_index = True)
            print 'Modulo 25: ', i % 25

            if i % 25 == 0:
                print '>>>>>>>>>>>SAVING PROGRESS>>>>>>>>>>>'
                print '>>>>>>>>>>>SAVING PROGRESS>>>>>>>>>>>'
                print '>>>>>>>>>>>SAVING PROGRESS>>>>>>>>>>>'
                pickle_object(df,'whiskeybase_soups_df')

    return df





if __name__ == '__main__':
    visited = []
    whiskey_urls = []

    session = dryscrape.Session()


    with open('whiskeybase_actual_distillery_urls.pkl') as f:
        actual_distillery_urls = pickle.load(f)

    with open('whiskeybase_actual_whiskey_urls.pkl') as f:
        actual_whiskey_urls = pickle.load(f)

    with open('whiskeybase_visited_urls.pkl') as f:
        visited = pickle.load(f)

    with open('whiskeybase_actual_whiskey_urls_part2.pkl') as f:
        actual_whiskey_urls_part2 = pickle.load(f)

    with open('whiskeybase_actually_visited_urls.pkl') as f:
        actually_visited = pickle.load(f)

    remaining_to_visit = list(set(visited) - set(actually_visited))

    first_run = False
    if first_run:
        actual_whiskey_urls_part2 = []
        actually_visited = [ 'https://www.whiskybase.com/distillery/85/bunnahabhain','https://www.whiskybase.com/distillery/82/springbank','https://www.whiskybase.com/distillery/75/mortlach','https://www.whiskybase.com/distillery/101/glendronach','https://www.whiskybase.com/distillery/76/glenrothes','https://www.whiskybase.com/distillery/115/tobermory','https://www.whiskybase.com/distillery/52/ben-nevis','https://www.whiskybase.com/distillery/229/buffalo-trace-distillery','https://www.whiskybase.com/distillery/125/longmorn','https://www.whiskybase.com/distillery/109/glen-grant','https://www.whiskybase.com/distillery/113/edradour','https://www.whiskybase.com/distillery/48/aberlour','https://www.whiskybase.com/distillery/51/auchentoshan' ]

    scrape_all_whiskey_urls(session, actual_distillery_urls, actual_whiskey_urls_part2, actually_visited)
