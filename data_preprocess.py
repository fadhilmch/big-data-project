import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer

tok = WordPunctTokenizer()
mention_web_pat = r'@[A-Za-z0-9_]+|https?://[^ ]+' # regex to remove mentions
www_pat = r'www\.[^ ]+' # regex to remove link
negation_dic = {("isnt", "isn't"): "is not", ("arent", "aren't"): "are not", ("wasnt", "wasn't"): "was not",
                ("werent", "weren't"): "were not", ("havent", "haven't"): "have not", ("hasnt", "hasn't"): "has not",
                ("hadnt", "hadn't"): "had not", ("wont", "won't"): "will not", ("wouldnt", "wouldn't"): "would not",
                ("dont", "don't"): "do not", ("doesnt", "doesn't"): "does not", ("didnt", "didn't"): "did not",
                ("cant", "can't"): "can not", ("couldnt", "couldn't"): "could not",
                ("shouldnt", "shouldn't"): "should not",
                ("mightnt", "mightn't"): "might not", ("musnt", "musn't"): "must not"} # dictionary for contractions
negation_dic = {k: v for kl, v in negation_dic.items() for k in kl}
negation_pat = re.compile(r'\b(' + '|'.join(negation_dic.keys()) + r')\b') # regex for contraction


# clean tweet
def clean_tweet(tweet):
    # remove html tags
    soup = BeautifulSoup(tweet, 'html5lib')
    tweet = soup.get_text()

    # handle encoding
    try:
        tweet = tweet.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        tweet = tweet

    # remove useless things from tweet
    tweet = re.sub(mention_web_pat, '', tweet)  # remove mentions & links (http)
    tweet = re.sub(www_pat, '', tweet)  # remove links (www)
    tweet = tweet.lower()  # lower case
    tweet = negation_pat.sub(lambda x: negation_dic[x.group()], tweet)  # handle contraction
    tweet = re.sub("[^A-Za-z]", " ", tweet)  # remove non alphabetical char

    # remove unnecessary space
    tweet = [word for word in tok.tokenize(tweet) if len(word) > 1]
    tweet = (" ".join(tweet)).strip()

    return tweet


# clean dataset. dataframe columns : 'target', 'tweet'
def clean_dataset(df):
    df_cleaned_tweets = [] # list to gather cleaned tweet

    # clean tweet
    for i in range(0, len(df)):
        if (i + 1) % 100000 == 0:
            print('{} of {} have been cleaned'.format(i + 1, len(df)))
        df_cleaned_tweets.append(clean_tweet(df.tweet[i]))

    df_cleaned_tweets = pd.DataFrame(df_cleaned_tweets, columns=['tweet'])
    df_cleaned_tweets['target'] = df.target
    df_cleaned_tweets = df_cleaned_tweets.replace('', np.nan, regex=True)
    df_cleaned_tweets.dropna(inplace=True)
    df_cleaned_tweets.reset_index(drop=True, inplace=True)
    df_cleaned_tweets.loc[df_cleaned_tweets.target == 4, 'target'] = 1

    return df_cleaned_tweets


def generate_train_test(df, test_size):
    # randomly samples test data for both negative and positive sentiment
    test_neg = df[df.target == 0].sample(int(test_size/2))
    test_pos = df[df.target == 1].sample(int(test_size/2))
    train_neg = df[(df.target == 0) & (~df.tweet.isin(test_neg.tweet))]
    train_pos = df[(df.target == 1) & (~df.tweet.isin(test_pos.tweet))]

    test_neg.reset_index(drop=True, inplace=True)
    test_pos.reset_index(drop=True, inplace=True)
    train_neg.reset_index(drop=True, inplace=True)
    train_pos.reset_index(drop=True, inplace=True)

    test = pd.concat([test_neg, test_pos], ignore_index=True)
    train = pd.concat([train_neg, train_pos], ignore_index=True)

    return train, test
