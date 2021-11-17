import tweepy
from textblob import TextBlob

consumer_key = 'OYS7iHyyIpSqNNDQjIXXFS5J5'
consumer_secret = 'MAY644ChXFBepNbY22ioaihSkv4161fIpkjmjHCaLC7KRWuPxf'

access_token = '829223817268621313-E5vNhSAMSKJQulVJX7AVwgCqRXP9ysN'
access_token_secret = '0L9n68GEcOLo6LwLOJRGJVLYYC796kvQINhndigIf4fwu'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)


public_tweets = api.search_tweets('naveen', count = 100)

for tweet in public_tweets:
    print(tweet.text)

    analysis = TextBlob(tweet.text)

    print(analysis.sentiment)
    print("")
