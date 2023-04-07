import snscrape

import snscrape.modules.twitter as scraper
import pandas as panda

topic = "Twitter"
query = topic + " lang:en -filter:links -filter:replies"
limit = 100

tweetsList = []
for tweet in scraper.TwitterSearchScraper(query).get_items():
    if len(tweetsList) == limit:
        break
    else:
        tweetsList.append([tweet.date, tweet.rawContent, topic, tweet.user.username, tweet.lang, tweet.source])

currentDataFrame = panda.DataFrame(tweetsList, columns=["Date", "Tweet", "Topic", "Username", "Language", "Source"])

currentDataFrame.to_csv("currentData.csv", index = False)

try:
    completeDataFrame = panda.read_csv("completeData.csv")
    panda.concat([completeDataFrame, currentDataFrame]).to_csv("completeData.csv", index = False)
except:
    currentDataFrame.to_csv("completeData.csv", index = False)

completeDataFrame = panda.read_csv("completeData.csv")
completeDataFrame.to_html("index.html")

print("Done...")