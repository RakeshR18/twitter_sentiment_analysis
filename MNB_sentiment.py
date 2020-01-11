from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
import re
from sys import argv

cols = ["instance_number","tweet_text","topic_id","sentiment","is_sarcastic"]
path = argv[1]

dataset = pd.read_table(path,sep='\t',names = cols, quoting = csv.QUOTE_NONE)
dataset.drop(['instance_number','topic_id','is_sarcastic'],axis=1,inplace = True)

x_train, x_test, y_train, y_test = train_test_split(dataset['tweet_text'],dataset['sentiment'] ,test_size = 0.25, shuffle=False)

def processTweet(tweet):
        
        tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' ', tweet) # remove URLs
        tweet = re.sub(r"[^a-zA-Z0-9#@_$%' ']+", '', tweet) #remove special characters
        return tweet
x_train = x_train.apply(lambda x: processTweet(x))


count = CountVectorizer(token_pattern=r'[a-zA-Z0-9#@%_$][a-zA-Z0-9#@%_$]+',lowercase = False)
bag_of_words = count.fit_transform(x_train)

# Create feature matrix
X = bag_of_words.toarray()
# Create target vector
Y = np.array(y_train)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
model = clf.fit(X, Y)

##############################################################################

test = argv[2]

test_data = pd.read_table(test,quoting =csv.QUOTE_NONE, sep='\t',names = ["instance_number","tweet_text","topic_id","sentiment","is_sarcastic"])
test_data.drop(['topic_id','is_sarcastic','sentiment'],axis = 1,inplace = True)

def predict_test_data(tweet):
    tweet_txt = tweet["tweet_text"]
    #print(tweet_txt)
    instance_number = tweet["instance_number"]
    for i in range(tweet_txt.size):
        test1 = count.transform([tweet_txt[i]]).toarray()
        sentiment_prediction = model.predict(test1)
        print(instance_number[i],sentiment_prediction[0])
        
predict_test_data(test_data)