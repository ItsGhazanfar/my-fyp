#Import Packages
import nltk
import numpy as np

import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer 

import csv

# Converting the Dataset from csv file to list
csv_filename = 'data.csv'
all_positive_tweets = []
all_negative_tweets = []
with open(csv_filename) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['sentiment'] == 'positive':
            all_positive_tweets.append(row['text'])
        elif row['sentiment'] == 'negative':
            all_negative_tweets.append(row['text'])

# Next, we'll print a report with the number of positive and negative tweet
# Preprocess Tweets
# Remove hyperlinks, Twitter marks and styles

# We do not want to use every word in a tweet because many tweets have hashtags, retweet marks, and hyperlinks. 
# We will use regular expressions to remove them from a tweet.

def remove_hyperlinks_marks_styles(tweet):
    
    # remove old style retweet text "RT"
    new_tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    new_tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', new_tweet)

    # remove hashtags
    # only removing the hash # sign from the word
    new_tweet = re.sub(r'#', '', new_tweet)
    
    return new_tweet


# Tokenize the string
# instantiate tokenizer class
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

def tokenize_tweet(tweet):
    
    tweet_tokens = tokenizer.tokenize(tweet)
    
    return tweet_tokens


# Remove stop works and punctuations

# Remove stop words and punctuations. Stop words are words that don't add significant meaning to the text. 
# For example, 'me', 'myself' and 'and', etc.

nltk.download('stopwords')

# Import the english stop words list from NLTK
stopwords_english = stopwords.words('english')

punctuations = string.punctuation

def remove_stopwords_punctuations(tweet_tokens):
    
    tweets_clean = []
    
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in punctuations):
            tweets_clean.append(word)
            
    return tweets_clean


# Stemming - converting a word to its most general form
#For example: running becomes run

stemmer = PorterStemmer()

def get_stem(tweets_clean):
    
    tweets_stem = []
    
    for word in tweets_clean:
        stem_word = stemmer.stem(word)
        tweets_stem.append(stem_word)
        
    return tweets_stem

"""
tweet_example = all_positive_tweets[1000]
print(tweet_example)

processed_tweet = remove_hyperlinks_marks_styles(tweet_example)
print("\nRemoved hyperlinks, Twitter marks and styles:")
print(processed_tweet)

tweet_tokens = tokenize_tweet(processed_tweet)
print("\nTokenize the string:")
print(tweet_tokens)

tweets_clean = remove_stopwords_punctuations(tweet_tokens)
print("\nRemove stop words and punctuations:")
print(tweets_clean)

tweets_stem = get_stem(tweets_clean)
print("\nGet stem of each word:")
print(tweets_stem)
"""

# An Example showing when we combine all preprocess techniques

def process_tweet(tweet):
    
    processed_tweet = remove_hyperlinks_marks_styles(tweet)
    tweet_tokens = tokenize_tweet(processed_tweet)
    tweets_clean = remove_stopwords_punctuations(tweet_tokens)
    tweets_stem = get_stem(tweets_clean)
    
    return tweets_stem

# Removing the elements which won't have any words left after preprocessing

def lengthOfProcessedTweet(tweet):
    return len(process_tweet(tweet))

index = 0
while index < len(all_positive_tweets):
    if lengthOfProcessedTweet(all_positive_tweets[index]) == 0:
        all_positive_tweets.pop(index)
    index += 1

index = 0
while index < len(all_negative_tweets):
    if lengthOfProcessedTweet(all_negative_tweets[index]) == 0:
        all_negative_tweets.pop(index)
    index += 1

"""
print('Number of positive tweets: ', len(all_positive_tweets))
print('Number of negative tweets: ', len(all_negative_tweets))

tweet_example = all_negative_tweets[1000]
print(tweet_example)

processed_tweet = process_tweet(tweet_example)
print(processed_tweet, len(processed_tweet))
"""

# Split data into two pieces, one for training and one for testing
                                        
test_pos = all_positive_tweets[3500:]   
train_pos = all_positive_tweets[:3500]         

test_neg = all_negative_tweets[3500:]
train_neg = all_negative_tweets[:3500]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

# Create frequency dictionary

def create_frequency(tweets, ys):
    
    freq_d = {}

    for tweet, y in zip(tweets, ys):
        for word in process_tweet(tweet):
            pair = (word, y)
            
            if pair in freq_d:
                freq_d[pair] += 1
                
            else:
                freq_d[pair] = freq_d.get(pair, 1)
    
    return freq_d

"""
# Testing frequency dictionary

tweets = ['i am happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
ys = [1, 0, 0, 0, 0]

freq_d = create_frequency(tweets, ys)
print(freq_d)
"""
# Train model using Naive Bayes
# Build the freqs dictionary

freqs = create_frequency(train_x, train_y)

def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. 
        loglikelihood: the log likelihood of you Naive bayes equation.
    '''
    
    loglikelihood = {}
    logprior = 0
    
    # calculate the number of unique words in vocab
    unique_words = set([pair[0] for pair in freqs.keys()])
    V = len(unique_words)
    
    # calculate N_pos and N_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        
        if pair[1] > 0:
            N_pos += freqs[(pair)]
        else:
            N_neg += freqs[(pair)]
            
    # calculate the number of documents (tweets)
    D = train_y.shape[0]
    
    # calculate D_pos, the number of positive documents (tweets)
    D_pos = sum(train_y)
    
    # calculate D_neg, the number of negative documents (tweets)
    D_neg = D - sum(train_y)
    
    # TODO: calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)
    
    # for each unqiue word
    for word in unique_words:
        
        # get the positive and negative frequency of the word
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)
        
        # calculate the probability that word is positive, and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
        
        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)
        
    return logprior, loglikelihood

logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)

# Predict Tweets!
def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)
    '''

    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # get log likelihood of each keyword
        if word in loglikelihood:
            p += loglikelihood[word]
            
    return p

#Sentiment Checking
def checkSentiment(prob):
    if prob > 0.0:
        return "Positive"
    elif prob < 0.0:
        return "Negative"
    elif prob == 0.0:
        return "Neutral"

# Accuracy
def checkAccuracy():
    totalCount = count = 0
    for tweet, sentiment in zip(test_x, test_y):
        p = naive_bayes_predict(tweet, logprior, loglikelihood)
        if p > 0.0 and sentiment == 1.0:
            count += 1
        elif p < 0.0 and sentiment == 0.0:
            count += 1
        totalCount += 1
        #print(f'{tweet} -> {p:.2f} | {sentiment}')
    print("Count of Accurate Predictions : ")
    print(count)
    print("\nTotal Predictions Made : ")
    print(totalCount)
    print("\nAccuracy : ")
    print((count/totalCount) * 100)
