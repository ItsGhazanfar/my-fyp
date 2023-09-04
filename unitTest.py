# import unittest
# import re
# import string
# import pandas as pd
# from nltk.sentiment.util import mark_negation
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# import pandas as pd
# import snscrape
# import snscrape.modules.twitter as scraper

# def preprocess_text_urls(text):
#     # Remove URLs from text
#     text = text.replace(r'http\S+|https\S+|www\S+|\S+\.com\S+', '', regex=True)
#     return text

# def preprocess_text_mentions(text):
#     # Remove mentions from text
#     text = text.replace(r'@\w+', '', regex=True)
#     return text

# def preprocess_text_hashtags(text):
#     # Remove hashtags from text
#     text = text.replace(r'#\w+', '', regex=True)
#     return text

# def preprocess_text_numeric_characters(text):
#         # Remove  numeric characters from text
#     text = text.replace(r'[0-9]+', '', regex=True)
#     return text

# def preprocess_text_repeated_characters(text):
#     # Replace any sequence of repeated characters with that single character
#     text = text.apply(lambda x: re.sub(r'(.)\1+', r'\1', x))
#     return text

# def preprocess_text_punctuation(text):
#     # Remove punctuation from text
#     text = text.str.translate(str.maketrans('', '', string.punctuation))
#     return text

# def preprocess_text_stopwords(text):
#     # Remove stopwords from text
#     stop_words = set(stopwords.words('english'))
#     text = text.apply(lambda words: [word for word in words if word not in stop_words])
#     return text

# def preprocess_text_stem_and_lemmatize(text):
#     # Perform stemming and lemmatization
#     stemmer = PorterStemmer()
#     lemmatizer = WordNetLemmatizer()
#     text = text.apply(lambda words: [stemmer.stem(lemmatizer.lemmatize(word)) for word in words])
#     return text

# def getTweet(topic, limit, name, toDate, fromDate):

#     nameQuery = ""
#     toDateQuery = ""
#     fromDateQuery = ""

#     if name == "":
#         nameQuery = ""
#         print("no name")
#     else:
#         nameQuery = "(from:" + name + ")"

#     if toDate == "":
#         toDateQuery = ""
#         print("no to date")
#     else:
#         toDateQuery = "until:" + toDate

#     if fromDate == "":
#         fromDateQuery = ""
#         print("no from date")
#     else:
#         fromDateQuery = "since:" + fromDate
    
#     query = topic + " " + nameQuery + " " + toDateQuery + " " + fromDateQuery + " " + "lang:en -filter:links -filter:replies"

#     tweetsList = []

#     for tweet in scraper.TwitterSearchScraper(query).get_items():
#         if len(tweetsList) == int(limit):
#             break
#         else:
#             tweetsList.append(tweet.rawContent)

#     return tweetsList


# class TestPreprocessText(unittest.TestCase):
    
#     def test_remove_urls(self):
#         text = pd.Series(['Visit us at http://www.example.com to learn more.', 
#                           'Check out our website at https://example.com for more information.'])
#         result = preprocess_text_urls(text)
#         expected = pd.Series([['Visit', 'us', 'at', 'to', 'learn', 'more'], 
#                               ['Check', 'out', 'our', 'website', 'at', 'for', 'more', 'information']])

#         print(result)   
#         print(expected)
        
#         self.assertTrue(result.equals(expected))
        
#     def test_remove_mentions(self):
#         text = pd.Series(['@user1 thanks for your help!', 
#                           'Hello, @user2! How are you doing?'])
#         result = preprocess_text_mentions(text)
#         expected = pd.Series([['thanks', 'for', 'your', 'help', '!'], 
#                               ['Hello', ',', 'How', 'are', 'you', 'doing', '?']])

#         print(result)
#         print(expected)

#         self.assertTrue(result.equals(expected))
        
#     def test_remove_hashtags(self):
#         text = pd.Series(['#MondayMotivation: Start your week with a positive attitude!', 
#                           'It was #amazing to see the #sunset last night.'])
#         result = preprocess_text_hashtags(text)
#         expected = pd.Series([['Start', 'your', 'week', 'with', 'a', 'positive', 'attitude', '!'], 
#                               ['It', 'was', 'to', 'see', 'the', 'last', 'night', '.']])
#         self.assertTrue(result.equals(expected))
        
#     def test_remove_numbers(self):
#         text = pd.Series(['I have 5 apples and 3 oranges.', 
#                           'There are 10 people in the room.'])
#         result = preprocess_text_numeric_characters(text)
#         expected = pd.Series([['I', 'have', 'apples', 'and', 'oranges', '.'], 
#                               ['There', 'are', 'people', 'in', 'the', 'room', '.']])
#         self.assertTrue(result.equals(expected))
        
#     def test_replace_repeated_chars(self):
#         text = pd.Series(['I looove this song!!!!', 
#                           'I am soooo happy right now.'])
#         result = preprocess_text_repeated_characters(text)
#         expected = pd.Series([['I', 'love', 'this', 'song', '!'], 
#                               ['I', 'am', 'so', 'happy', 'right', 'now', '.']])
#         self.assertTrue(result.equals(expected))
        
#     def test_remove_punctuation(self):
#         text = pd.Series(['This is a sentence.', 
#                           'How are you doing?'])
#         result = preprocess_text_punctuation(text)
#         expected = pd.Series([['This', 'is', 'a', 'sentence'], 
#                               ['How', 'are', 'you', 'doing']])
#         self.assertTrue(result.equals(expected))
        
#     def test_remove_stopwords(self):
#         text = pd.Series(['I am going to the store.', 
#                           'She is reading a book.'])
#         result = preprocess_text_stopwords(text)
#         expected = pd.Series([['I', 'going', 'store', '.'], 
#                               ['She', 'reading', 'book', '.']])
#         self.assertTrue(result.equals(expected))
        
#     def test_stem_and_lemmatize(self):
#         text = pd.Series(['The dogs are barking loudly.', 
#                         'I am eating a delicious meal.'])
#         result = preprocess_text_stem_and_lemmatize(text)
#         expected = pd.Series([['the', 'dog', 'are', 'bark', 'loudli', '.'], 
#                             ['I', 'am', 'eat', 'a', 'delici', 'meal', '.']])
#         self.assertTrue(result.equals(expected))

#     def test_get_tweet_with_valid_params(self):
#         # Test getting tweets with valid parameters
#         topic = "COVID-19"
#         limit = 5
#         name = "WHO"
#         toDate = "2022-01-01"
#         fromDate = "2021-12-01"
#         tweets = getTweet(topic, limit, name, toDate, fromDate)
#         self.assertEqual(len(tweets), limit)
#         for tweet in tweets:
#             self.assertIn(topic, tweet.lower())
#             self.assertIn(name.lower(), tweet.lower())
        
#     def test_get_tweet_with_empty_name_and_date(self):
#         # Test getting tweets with empty name and date parameters
#         topic = "COVID-19"
#         limit = 10
#         name = ""
#         toDate = ""
#         fromDate = ""
#         tweets = getTweet(topic, limit, name, toDate, fromDate)
#         self.assertEqual(len(tweets), limit)
#         for tweet in tweets:
#             self.assertIn(topic, tweet.lower())

#     def test_get_tweet_with_invalid_params(self):
#         # Test getting tweets with invalid parameters
#         topic = "invalid_topic"
#         limit = 5
#         name = "invalid_name"
#         toDate = "2022-01-01"
#         fromDate = "2021-12-01"
#         tweets = getTweet(topic, limit, name, toDate, fromDate)
#         self.assertEqual(len(tweets), 0)

# if __name__ == '__main__':
#     unittest.main()

# # test = TestPreprocessText()
# # test.test_remove_urls()
# # test.test_remove_mentions()
# # test.test_remove_hashtags()
# # test.test_remove_numeric_characters()
# # test.test_replace_repeated_characters()
# # test.test_remove_punctuation()
# # test.test_tokenize()
# # test.test_remove_stopwords()
# # test.test_stem_and_lemmatize()
