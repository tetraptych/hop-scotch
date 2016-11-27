from flask import Flask, request, render_template, url_for
from flask_bootstrap import Bootstrap
from helper import *
import recommend as rec


def create_app():
  app = Flask(__name__)

  Bootstrap(app)

  model = rec.get_model()
  rdf   = rec.get_rdf()

  return app, model, rdf

app, model, rdf = create_app()


# Home page
@app.route('/')
def index():
    return "hello"

@app.route('/dashboard')
def dashboard():
    rendered_template = render_template('index.html')
    return rendered_template

@app.route('/results', methods=['POST', 'GET'])
def results():
    data     = request.form['exampleTextarea']
    scores   = request.form['exampleTextarea2']
    minprice = request.form['minprice']
    maxprice = request.form['maxprice']
    type_    = request.form['type']

    if not rec.isfloat(minprice):
        minprice = 0.0
    else:
        minprice = float(minprice)

    if not rec.isfloat(maxprice):
        maxprice = 100000.0
    else:
        maxprice = float(maxprice)

    url_list = data.split('\n')
    score_list = scores.split('\n')
    score_list = [int(score) for score in score_list if rec.isfloat(score)]
    item_list = [rec.get_item_id_from_url(url) for url in url_list]

    if len(url_list) > 0:
        first = url_list[0]
        if ('www.whiskybase.' not in first) and (not rec.isfloat(first)):
            # Profile page, scrape to get ratings info
            item_list, score_list = rec.profile_input(first)

    # pred_url_list = predict(url_list)
    guess = rec.recommend(model, item_list, score_list)
    results = rec.filter_results(guess, rdf, type_, minprice, maxprice)
    # res = []
    # for whisk in guess:
    #     res.append('https://www.whiskybase.com/whisky/{}'.format(whisk))

    cols_needed_for_display = ['id', 'url', 'category', 'brand', 'name', 'price', 'photo_url']
    results2 = results[cols_needed_for_display]

    rec_list = results2.T.to_dict().values()


    # return rec_list.__repr__()

    return render_template('results.html', items = rec_list)

    # return results['item_id'].values.__repr__()

    # albums_list = list()
    # for album_url, art_id in zip(pred_url_list, art_id_list):
    #     albums_list.append({'album_url': album_url, 'art_id': art_id})
    # return render_template('results.html', items = albums_list)


if __name__ == '__main__':
  app.run(host = "0.0.0.0", port = int("8000"), debug = True)
