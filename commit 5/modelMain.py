import re
import string
import pandas as pd
from nltk.sentiment.util import mark_negation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle  

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vec.pkl','rb'))
def preprocess_text(text):

    # Remove URLs from 'text' data
    text = text.replace(r'http\S+|https\S+|www\S+|\S+\.com\S+', '', regex=True)
    
    # Remove mentions from 'text' data
    text = text.replace(r'@\w+', '', regex=True)
    
    # Remove hashtags from 'text' data
    text = text.replace(r'#\w+', '', regex=True)

    # Remove  numeric characters from 'text' data
    text = text.replace(r'[0-9]+', '', regex=True)
    
    # Replace any sequence of repeated characters with a single instance of that character
    text = text.apply(lambda x: re.sub(r'(.)\1+', r'\1', x))
    
    # Remove punctuation from 'text' data
    text = text.str.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    text = text.apply(word_tokenize)
    
    text = text.apply(mark_negation)

    # Remove stopwords from 'text' data
    stop_words = set(stopwords.words('english'))
    text = text.apply(lambda words: [word for word in words if word not in stop_words])

    # Perform stemming and lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    text = text.apply(lambda words: [stemmer.stem(lemmatizer.lemmatize(word)) for word in words])

    return text



data = {
  "asd": ["Oscar rubbing my feet w/ alcohol at this time to help with the pain"]
}

axi = pd.DataFrame(data)
axi['asd']=axi['asd'].map(str.lower)

axi['asd'] = preprocess_text(axi['asd'])

y = axi.asd.astype(str)
q = vectorizer.transform(y)

print(model.predict((q)))


