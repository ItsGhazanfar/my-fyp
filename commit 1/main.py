from TwitterSentimentAnalysisModel import *
from TweetScrapper import getTweet

print()
topic = input("Please Enter topic related to the tweet : ")
limit = input("Please Enter how many tweets do you want : ")
tweetsList = getTweet(topic, limit)

count = 1
print("Result : ")
for tweet in tweetsList:
    p = naive_bayes_predict(tweet, logprior, loglikelihood)
    p = round(p, 1)
    sentiment = checkSentiment(p)
    
    print()
    print(str(count) + ".")
    print("Tweet : " + tweet)
    print("Probability : " + str(p))
    print("Sentiment : " + sentiment)
    count += 1
