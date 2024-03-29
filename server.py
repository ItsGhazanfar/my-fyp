import snscrape
import snscrape.modules.twitter as scraper
import pandas as panda

from flask import Flask, request, send_file, render_template

import re
import string
import pandas as pd
from nltk.sentiment.util import mark_negation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle  


app = Flask(__name__, static_url_path='/static')

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vec.pkl','rb'))

@app.route('/')
def home():
    return send_file('templates\home.html')

@app.route('/analysis', methods=["POST"])
def analysis():
    topic = limit = toDate = fromDate = ""
    if "topic" in request.form:
        topic = request.form['topic']
    if "limit" in request.form:
        limit = request.form['limit']
    if "to-date" in request.form:
        toDate = request.form['to-date']
    if "from-date" in request.form:
        fromDate = request.form['from-date']

    tweetsList = getTweet(topic, limit, toDate, fromDate)
    df = panda.DataFrame(tweetsList, columns=["Tweet"])

    df['Tweet']=df['Tweet'].map(str.lower)

    df['Tweet'] = preprocess_text(df['Tweet'])

    y = df.Tweet.astype(str)
    q = vectorizer.transform(y)

    tempSentimentList = model.predict((q))
    sentimentList = []
    for sentiment in tempSentimentList:
        if sentiment == 0:
            sentimentList.append('Negative')
        elif sentiment == 1:
            sentimentList.append('Positive')

    data = {}
    data['tweets'] = tweetsList
    data['sentiments'] = sentimentList
    data['topic'] = topic;

    result = data
    
    return render_template('analysis.html', result=result)

@app.route('/about')
def about(): 
    return render_template('about.html')

def preprocess_text(text):

    # Remove URLs from text
    text = text.replace(r'http\S+|https\S+|www\S+|\S+\.com\S+', '', regex=True)
    
    # Remove mentions from text
    text = text.replace(r'@\w+', '', regex=True)
    
    # Remove hashtags from text
    text = text.replace(r'#\w+', '', regex=True)

    # Remove  numeric characters from text
    text = text.replace(r'[0-9]+', '', regex=True)
    
    # Replace any sequence of repeated characters with that single character
    text = text.apply(lambda x: re.sub(r'(.)\1+', r'\1', x))
    
    # Remove punctuation from text
    text = text.str.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    text = text.apply(word_tokenize)
    text = text.apply(mark_negation)

    # Remove stopwords from text
    stop_words = set(stopwords.words('english'))
    text = text.apply(lambda words: [word for word in words if word not in stop_words])

    # Perform stemming and lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    text = text.apply(lambda words: [stemmer.stem(lemmatizer.lemmatize(word)) for word in words])

    return text

def getTweet(topic, limit, toDate, fromDate):

    toDateQuery = ""
    fromDateQuery = ""

    if toDate == "":
        toDateQuery = ""
    else:
        toDateQuery = "until:" + toDate

    if fromDate == "":
        fromDateQuery = ""
    else:
        fromDateQuery = "since:" + fromDate
    
    query = topic + " " + toDateQuery + " " + fromDateQuery + " " + "lang:en -filter:links -filter:replies"

    tweetsList = []

    for tweet in scraper.TwitterSearchScraper(query).get_items():
        if len(tweetsList) == int(limit):
            break
        else:
            tweetsList.append(tweet.rawContent)

    return tweetsList

app.run(host="0.0.0.0", port=8080, debug=True)