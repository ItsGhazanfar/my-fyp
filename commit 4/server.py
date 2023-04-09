import snscrape
import snscrape.modules.twitter as scraper
import pandas as panda

from flask import Flask, request, send_file, render_template
 

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def home(): 
    return send_file('templates\home.html')

@app.route('/analysis', methods=["POST"])
def analysis():
    topic = limit = name = toDate = fromDate = ""
    if "topic" in request.form:
        topic = request.form['topic']
    if "limit" in request.form:
        limit = request.form['limit']
    if "name" in request.form:
        name = request.form['name']
    if "to-date" in request.form:
        toDate = request.form['to-date']
    if "from-date" in request.form:
        fromDate = request.form['from-date']

    df = getTweet(topic, limit, name, toDate, fromDate)
    result = df
    
    return render_template('analysis.html', result=result)

def getTweet(topic, limit, name, toDate, fromDate):

    nameQuery = ""
    toDateQuery = ""
    fromDateQuery = ""

    if name == "":
        nameQuery = ""
        print("no name")
    else:
        nameQuery = "(from:" + name + ")"

    if toDate == "":
        toDateQuery = ""
        print("no to date")
    else:
        toDateQuery = "until:" + toDate

    if fromDate == "":
        fromDateQuery = ""
        print("no from date")
    else:
        fromDateQuery = "since:" + fromDate
    
    query = topic + " " + nameQuery + " " + toDateQuery + " " + fromDateQuery + " " + "lang:en -filter:links -filter:replies"

    tweetsList = []

    for tweet in scraper.TwitterSearchScraper(query).get_items():
        if len(tweetsList) == int(limit):
            break
        else:
            tweetsList.append(tweet.rawContent)
    
    df = panda.DataFrame(tweetsList, columns=["Tweet"])

    return df


app.run(host="0.0.0.0", port=8080, debug=True)