import string

import pandas as pd
import numpy as np
from IPython.display import display

import email

# from src import convertinput

df_email = pd.read_csv(r'../data/test.csv')

headers = [header for header in df_email.columns]
print("Successfully loaded {} rows and {} columns!".format(df_email.shape[0], df_email.shape[1]))
print(df_email.head())


def insert_value(dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]
    return dictionary


def get_headers(df, header_names):
    headers = {}
    messages = df["email_msg"]
    for message in messages:
        e = email.message_from_string(message)
        for item in header_names:
            header = e.get(item)
            insert_value(dictionary=headers, key=item, value=header)
    print("Successfully retrieved header information!")
    return headers


header_names = ["from", "subject"]

headers = get_headers(df_email, header_names)


def get_messages(df):
    messages = []
    for item in df["email_msg"]:
        e = email.message_from_string(item)
        message_body = e.get_payload()
        message_body = message_body.lower()
        messages.append(message_body)
    print("Successfully retrieved message body from e-mails!")
    return messages


msg_body = get_messages(df_email)

df_email["msg_body"] = msg_body


def add_headers(df, header_list):
    for label in header_list:
        df_new = pd.DataFrame(headers[label], columns=[label])
        if label not in df.columns:
            df = pd.concat([df, df_new], axis=1)
    return df


remaining_headers = ["from", "subject"]
df_email = add_headers(df=df_email, header_list=remaining_headers)

import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords


def text_preproc(text):
    text = text.lower()
    stop_words = stopwords.words("english")
    text = ' '.join([word for word in text.split(' ') if word not in stop_words])

    text = text.encode('ascii', 'ignore').decode()
    text = re.sub(r'https*\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'\'\w+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d+\w*', '', text)
    text = re.sub(r'\s{2,}', ' ', text)

    text = re.sub("\n", '', text)
    text = ' '.join([w for w in text.split() if len(w) > 1])

    wordnet = WordNetLemmatizer()

    text = ' '.join([wordnet.lemmatize(word) for word in text.split(' ')])
    return text.strip()


def remove_startofstring(text):
    text = re.sub('^%s' % 're', '', text)
    return text.strip()


df_email['cleaned_msg'] = df_email["msg_body"].apply(text_preproc)

df_email['cleaned_subject'] = df_email["subject"].apply(text_preproc)

df_email['cleaned_subject'] = df_email["cleaned_subject"].apply(remove_startofstring)

df_email['all_text'] = df_email["cleaned_subject"] + df_email['cleaned_msg']

df_email['Num_words_email'] = df_email['all_text'].apply(lambda x: len(str(x).split()))

df_email.to_csv('../data/preprocess_test.csv', index=False)

print(df_email['Num_words_email'].describe())