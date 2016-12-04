from flask import Flask, request, render_template, url_for
from flask_bootstrap import Bootstrap
import recommend as rec


def create_app():
  app = Flask(__name__)
  Bootstrap(app)
  model = rec.get_model()
  rdf   = rec.get_rdf()
  W, similarity_matrix = rec.get_similarity_matrix()

  return app, model, rdf, W, similarity_matrix

app, model, rdf, W, similarity_matrix = create_app()

# Home page
@app.route('/')
def homepage():
    return 'Welcome to Hop/Scotch! <br><a href="/recommend">Click here to get some recommendations.</a>'

@app.route('/recommend')
def search_page():
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
    guess = rec.recommend(model, item_list, score_list, rdf = rdf, similarity_matrix = similarity_matrix )
    results = rec.filter_results(guess, rdf, type_, minprice, maxprice, num_to_show = 20)
    # res = []
    # for whisk in guess:
    #     res.append('https://www.whiskybase.com/whisky/{}'.format(whisk))

    cols_needed_for_display = ['id', 'url', 'category', 'brand', 'name', 'price', 'photo_url', 'null_photo_url', 'brand_and_name']

    results2 = results[results['null_photo_url'] == 0]
    results3= results2[cols_needed_for_display]
    # results3 = results[cols_needed_for_display]

    rec_list = results3.T.to_dict().values()

    return render_template('results.html', items = rec_list)



if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = int("8000"), debug = True)
