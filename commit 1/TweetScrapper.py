import snscrape

import snscrape.modules.twitter as scraper
import pandas as panda

def getTweet(topic, limit):

    userQuery = ""
    """
    if username == "":
        userQuery = ""
    else:
        userQuery = "(from:" + username + ")"
    """
    query = topic + userQuery + " lang:en -filter:links -filter:replies"

    tweetsList = []
    tweetInfo = {}
    for tweet in scraper.TwitterSearchScraper(query).get_items():
        if len(tweetsList) == int(limit):
            break
        else:
            columns = ["Date", "Text", "Topic", "Username", "Language", "Source"]
            tweetDetails = [tweet.date, tweet.rawContent, topic, tweet.user.username, tweet.lang, tweet.source]
            for key, value in zip(columns, tweetDetails):
                tweetInfo[key] = value
            tweetsList.append(tweetInfo["Text"])
    return tweetsList
