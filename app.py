from sqlalchemy import func
import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import re
from textblob import TextBlob
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from flask import (
    Flask,
    render_template,
    jsonify,
    make_response)

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper


import csv
from flask_cors import (CORS , cross_origin)



app = Flask(__name__)


cors = CORS(app, resources={r"/*": {"origins": "*"}})


# !/usr/bin/env python
# coding: utf-8

# In[1]:

# import libraries

# In[2]:

# read csv into dataframe
articles = pandas.read_csv('articles_copy.csv', encoding='cp1252')
articles.head(3)

# In[3]:

# download nltk stuff

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# In[4]:

# methods to clean up articles


stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

def accept_words(token):
   return token not in stop_words_ and token not in list(string.punctuation)

def clean_txt(text):
   clean_text = []
   text = re.sub("'", "", text)
   text = re.sub("[\d\W]+", " ", text)
   clean_text = [wn.lemmatize(word) for word in word_tokenize(text.lower()) if accept_words(word)]
   return " ".join(clean_text)

# In[5]:

# example of how articles get cleaned
articles['article'][0]

# In[6]:

clean_txt(articles['article'][0])

# In[7]:

# additional features (subjectivity and polarity)
# however, have not used polarity since naive bayes can't take negative values

import textstat

def hard_words(text):
   total_words = len(text.split())
   return textstat.difficult_words(text) / total_words

def subj_txt(text):
   return TextBlob(text).sentiment[1]

def polarity_txt(text):
   return (TextBlob(text).sentiment[0] + 1) / 2

def readability(text):
   return textstat.automated_readability_index(text)

def unique_words(text):
   return len(set(clean_txt(text).split())) / len(text.split())

# In[8]:

articles['subj'] = articles['article'].apply(subj_txt)
articles['pol'] = articles['article'].apply(polarity_txt)
articles['difficult_words'] = articles['article'].apply(hard_words)
articles['readability'] = articles['article'].apply(readability)
articles['unique_words'] = articles['article'].apply(unique_words)

# In[9]:

articles['pol'].sort_values(ascending=True)

# In[10]:

# Custom class for feature union

class item_select(BaseEstimator, TransformerMixin):
   def __init__(self, key):
       self.key = key

   def fit(self, x, y=None):
       return self

   def transform(self, data_dict):
       return data_dict[self.key]

class text_data(BaseEstimator, TransformerMixin):
   """Extract features from each document for DictVectorizer"""

   def fit(self, x, y=None):
       return self

   def transform(self, data):
       return [{'sub': row['subj'], 'difficult_words': row['difficult_words'],
                'pol': row['pol'], 'readability': row['readability'],
                'unique_words': row['unique_words']} \
               for _, row in data.iterrows()]

# In[11]:

# custom pipeline

pipeline = Pipeline([
   ('union', FeatureUnion(
       transformer_list=[

           #             Pipeline for pulling features from the text
           ('article', Pipeline([
               ('selector', item_select(key='article')),
               ('tfidf', TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                                         ngram_range=(1, 3), max_features=100,
                                         preprocessor=clean_txt)),
           ])),

           #             Pipeline for pulling metadata features
           ('stats', Pipeline([
               ('selector', item_select(key=['subj', 'difficult_words', 'pol', 'readability', 'unique_words'])),
               ('stats', text_data()),  # returns a list of dicts
               ('vect', DictVectorizer()),  # list of dicts -> feature matrix
           ])),

       ],
   ))
])

# In[12]:

# features and labels
X = articles[['article', 'subj', 'difficult_words', 'pol', 'readability', 'unique_words']]
y = articles['political leaning']
print(y)

# In[13]:

# split training and testing samples
seed = 12
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(y)
print(y)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)

# In[14]:

# fit pipeline
pipeline.fit(x_train)

# In[15]:

train_vec = pipeline.transform(x_train)
test_vec = pipeline.transform(x_test)

# In[16]:

# models to test

clf_sv = LinearSVC(C=1, class_weight='balanced', multi_class='ovr', random_state=40, max_iter=10000)
clf_nb = naive_bayes.MultinomialNB()
clf_lr = LogisticRegression()
clf_rf = RandomForestClassifier(n_estimators=800)

# In[17]:

# cross val score of models

clfs = {'SVC': clf_sv, 'NB': clf_nb, 'LR': clf_lr, 'RF': clf_rf}
cv = 3
for name, clf in clfs.items():
   scores = cross_val_score(clf, test_vec, y_test, cv=cv, scoring="accuracy")
   print(name)
   print(scores)
   print(("Mean score: {0:.3f} (+/-{1:.3f})").format(
       numpy.mean(scores), numpy.std(scores)))
   print('---------------------------')

# In[18]:

articles.groupby('political leaning')['unique_words'].describe()

# In[19]:

articles.groupby('political leaning')['difficult_words'].describe()

# In[20]:

articles.groupby('political leaning')['pol'].describe()

# In[21]:

articles.groupby('political leaning')['readability'].describe()

# In[22]:

articles.groupby('political leaning')['subj'].describe()

# In[23]:

# Testing model with just tfidf
X = articles['article']
y = articles['political leaning']
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)
tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                           ngram_range=(1, 3), max_features=100,
                           preprocessor=clean_txt)
tfidf_vec.fit(x_train)
train_x = tfidf_vec.transform(x_train)
test_x = tfidf_vec.transform(x_test)

# In[24]:

for name, clf in clfs.items():
   scores = cross_val_score(clf, test_x, y_test, cv=cv, scoring="accuracy")
   print(name)
   print(scores)
   print(("Mean score: {0:.3f} (+/-{1:.3f})").format(
       numpy.mean(scores), numpy.std(scores)))

# In[25]:

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1, 3),
                                        max_features=100)
tfidf_vect_ngram_chars.fit(x_train)
train_x = tfidf_vect_ngram_chars.transform(x_train)
test_x = tfidf_vect_ngram_chars.transform(x_test)

# In[26]:

for name, clf in clfs.items():
   scores = cross_val_score(clf, test_x, y_test, cv=cv, scoring="accuracy")
   print(name)
   print(scores)
   print(("Mean score: {0:.3f} (+/-{1:.3f})").format(
       numpy.mean(scores), numpy.std(scores)))

# In[27]:

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100)
count_vect.fit(x_train)
train_x = count_vect.transform(x_train)
test_x = count_vect.transform(x_test)

# In[28]:

for name, clf in clfs.items():
   scores = cross_val_score(clf, test_x, y_test, cv=cv, scoring="accuracy")
   print(name)
   print(scores)
   print(("Mean score: {0:.3f} (+/-{1:.3f})").format(
       numpy.mean(scores), numpy.std(scores)))

# In[29]:

# transforming data
modelx = articles[['article', 'subj', 'difficult_words', 'pol', 'readability', 'unique_words']]
modely = articles['political leaning']
modely = encoder.fit_transform(modely)
pipeline.fit(modelx)
model_x = pipeline.transform(modelx)

# In[45]:

# this is where we take in the user input
# user = input('input article you are trying to judge: ')
d = {'article': [0]}
test = pandas.DataFrame(data=d)


#default port number is 5000
#URL for flask: localhost:5000
#

@app.route('/example', methods=['POST'])
def get_names():
   if request.method == 'POST':
       data = request.form['Article']

       # user = input('Input the article you are trying to judge: ')
       #ser = Bringing in Mr. Starr will also invariably supercharge the discussion over Mr. Trump’s impeachment by reopening the long-running debate over Mr. Clinton’s case. Mr. Starr remains a polarizing figure from that era and every point he makes in favor of Mr. Trump’s innocence will invite comparisons to the approach he took to Mr. Clinton.
       #But Mr. Trump evidently sees Mr. Starr as an important validating presence who could endorse the view that the president’s impeachment was illegitimate and unfair. The prosecutor whose investigation triggered the last presidential impeachment will now stand up on the floor of the Senate to declare that this impeachment is invalid. And he will explain why, in the view of someone who has been there, these charges do not add up to high crimes.
       #“President Trump has done nothing wrong and is confident that this team will defend him, the voters, and our democracy from this baseless, illegitimate impeachment,” the White House said in a statement on Friday night, confirming earlier news reports.
       #For some Republicans who admire Mr. Starr, his participation may carry weight. “I was encouraged by it,” Senator Kevin Cramer, Republican of North Dakota, said of the newly constituted legal team.
       #But Mr. Trump’s built-out team — which will be led by the White House counsel, Pat A. Cipollone, and the president’s personal lawyer Jay Sekulow — faces the dual challenge of preserving the president’s support among Republican senators and presenting his case to the wider public watching on television during an election year.
       #Bringing in Mr. Starr will also invariably supercharge the discussion over Mr. Trump’s impeachment by reopening the long-running debate over Mr. Clinton’s case. Mr. Starr remains a polarizing figure from that era and every point he makes in favor of Mr. Trump’s innocence will invite comparisons to the approach he took to Mr. Clinton.
       #But Mr. Trump evidently sees Mr. Starr as an important validating presence who could endorse the view that the president’s impeachment was illegitimate and unfair. The prosecutor whose investigation triggered the last presidential impeachment will now stand up on the floor of the Senate to declare that this impeachment is invalid. And he will explain why, in the view of someone who has been there, these charges do not add up to high crimes.
       #“President Trump has done nothing wrong and is confident that this team will defend him, the voters, and our democracy from this baseless, illegitimate impeachment,” the White House said in a statement on Friday night, confirming earlier news reports.
       #For some Republicans who admire Mr. Starr, his participation may carry weight. “I was encouraged by it,” Senator Kevin Cramer, Republican of North Dakota, said of the newly constituted legal team.
       #But Mr. Trump’s built-out team — which will be led by the White House counsel, Pat A. Cipollone, and the president’s personal lawyer Jay Sekulow — faces the dual challenge of preserving the president’s support among Republican senators and presenting his case to the wider public watching on television during an election year.

       test['article'][0] = data

       # transforming user input
       test['subj'] = test['article'].apply(subj_txt)
       test['pol'] = test['article'].apply(polarity_txt)
       test['difficult_words'] = test['article'].apply(hard_words)
       test['readability'] = test['article'].apply(readability)
       test['unique_words'] = test['article'].apply(unique_words)
       testx = test[['article', 'subj', 'difficult_words', 'pol', 'readability', 'unique_words']]
       pipeline.fit(testx)
       testx = pipeline.transform(testx)

       # In[46]:

       ##Predicting political lean of user input
       def trans(result):
           if result[0] == 1:
               return 'liberal'
           else:
               return 'conservative'

       # Random Forest
       forest = RandomForestClassifier(n_estimators=800)
       forest = forest.fit(model_x, modely)
       RFresult = forest.predict(testx)
       RFresult = trans(RFresult)
       # clf_sv = LinearSVC(C=1, class_weight='balanced', multi_class='ovr', random_state=40, max_iter=10000)
       # clf_nb = naive_bayes.MultinomialNB()
       # clf_lr = LogisticRegression()
       # clf_rf = RandomForestClassifier(n_estimators = 800)
       # Naive Bayes
       NBmodel = naive_bayes.MultinomialNB()
       NBmodel = NBmodel.fit(model_x, modely)
       NBresult = NBmodel.predict(testx)
       NBresult = trans(NBresult)

       # Linear SVC
       SVmodel = LinearSVC(C=1, class_weight='balanced', multi_class='ovr', random_state=40, max_iter=10000)
       SVmodel = SVmodel.fit(model_x, modely)
       SVresult = SVmodel.predict(testx)
       SVresult = trans(SVresult)

       # Logistic Regression
       LRmodel = LogisticRegression()
       LRmodel = LRmodel.fit(model_x, modely)
       LRresult = LRmodel.predict(testx)
       LRresult = trans(LRresult)

       print(f'This is the political lean according to these models\nRandomForest: {RFresult}\nNaive Bayes: {NBresult}\nLinear SVC: {SVresult}\nLinear Regression: {LRresult}')
       returnString=f'This is the political lean according to these models\nRandomForest: {RFresult}\nNaive Bayes: {NBresult}\nLinear SVC: {SVresult}\nLinear Regression: {LRresult}'

       return render_template("result.html", value=f'{RFresult}')


       # In[ ]:

       # In[ ]:

       #return f'<h1>This is the political lean according to these models\n</h1>' \
        #      f'<h1>RandomForest: {RFresult}</h1>' \
         #     f'<h1>Naive Bayes: {NBresult}</h1>' \
 #    f'<h1>Linear SVC: {SVresult}</h1>' \
#   f'<h1>Linear Regression: {LRresult}</h1>'


@app.route('/graph', methods=['GET'])
def graph():
    # !/usr/bin/env python
    # coding: utf-8

    # In[1]:

    import pandas as pd
    import numpy as np

    from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn import decomposition, ensemble
    import pandas, xgboost, numpy, textblob, string
    from keras.preprocessing import text, sequence
    from keras import layers, models, optimizers
    import tensorflow
    tensorflow.keras.__version__

    # In[2]:

    apolitcal = pd.read_csv('articles_copy.csv', encoding='cp1252')

    apolitcal.head()

    # In[3]:

    apolitcal.shape

    # In[4]:

    test = apolitcal.iloc[3]
    test

    j = apolitcal['article']
    j[1]

    # In[5]:

    # download nltk stuff
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # methods to clean up articles
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    from nltk import word_tokenize
    import string
    import re

    # In[6]:

    stop_words_ = set(stopwords.words('english'))
    stop_words_.update(('mr', 'one', 'm ', 'x', 'xa', 'n n', 'm', 'said', 'u', 'and', 'I', 'A', 'And', 'So', 'arnt',
                        'This', 'When', 'It', 'many', 'Many', 'so', 'cant', 'Yes', 'yes', 'No', 'no', 'These', 'these'))

    wn = WordNetLemmatizer()

    def accept_words(token):
        return token not in stop_words_ and token not in list(string.punctuation)

    def clean_txt(text):
        clean_text = []
        text = re.sub("'", "", text)
        text = re.sub("[\d\W]+", " ", text)
        clean_text = [wn.lemmatize(word) for word in word_tokenize(text.lower()) if accept_words(word)]
        return " ".join(clean_text)

    # In[7]:

    stop_words_

    # In[ ]:

    # In[8]:

    # example of how articles get cleaned
    apolitcal['article'][0]

    # In[9]:

    clean_txt(apolitcal['article'][0])

    # In[10]:

    # additional features (subjectivity and polarity)
    # however, have not used polarity since naive bayes can't take negative values
    from textblob import TextBlob
    from sklearn.feature_extraction import DictVectorizer
    import textstat

    def hard_words(text):
        total_words = len(text.split())
        return textstat.difficult_words(text) / total_words

    def subj_txt(text):
        return TextBlob(text).sentiment[1]

    def polarity_txt(text):
        return (TextBlob(text).sentiment[0] + 1) / 2

    def readability(text):
        return textstat.automated_readability_index(text)

    def unique_words(text):
        return len(set(clean_txt(text).split())) / len(text.split())

    # In[11]:

    apolitcal['parsed'] = apolitcal['article']

    # In[12]:

    # run thr every column for cleaned text into parsed column
    count = 0
    while (count < len(apolitcal)):
        apolitcal['parsed'][count] = clean_txt(apolitcal['article'][count])
        count = count + 1
        print(count)
    else:
        print("Done")

    # In[ ]:

    # In[ ]:

    # In[ ]:

    # In[ ]:

    # In[ ]:

    # In[ ]:

    # In[ ]:

    # In[13]:

    apolitcal['subj'] = apolitcal['parsed'].apply(subj_txt)
    apolitcal['pol'] = apolitcal['parsed'].apply(polarity_txt)
    apolitcal['difficult_words'] = apolitcal['parsed'].apply(hard_words)
    apolitcal['readability'] = apolitcal['parsed'].apply(readability)
    apolitcal['unique_words'] = apolitcal['parsed'].apply(unique_words)

    # In[14]:

    apolitcal['pol'].sort_values(ascending=True)

    # In[15]:

    apolitcal.head()

    # In[ ]:

    # In[ ]:

    # In[16]:

    # Custom class for feature union
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import FeatureUnion
    from sklearn.feature_extraction import DictVectorizer
    class item_select(BaseEstimator, TransformerMixin):
        def __init__(self, key):
            self.key = key

        def fit(self, x, y=None):
            return self

        def transform(self, data_dict):
            return data_dict[self.key]

    class text_data(BaseEstimator, TransformerMixin):
        """Extract features from each document for DictVectorizer"""

        def fit(self, x, y=None):
            return self

        def transform(self, data):
            return [{'sub': row['subj'], 'difficult_words': row['difficult_words'],
                     'pol': row['pol'], 'readability': row['readability'],
                     'unique_words': row['unique_words']} \
                    for _, row in data.iterrows()]

    # In[17]:

    # custom pipeline
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[

                #             Pipeline for pulling features from the text
                ('parsed', Pipeline([
                    ('selector', item_select(key='parsed')),
                    ('tfidf', TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                                              ngram_range=(1, 3), max_features=100,
                                              preprocessor=clean_txt)),
                ])),

                #             Pipeline for pulling metadata features
                ('stats', Pipeline([
                    ('selector', item_select(key=['subj', 'difficult_words', 'pol', 'readability', 'unique_words'])),
                    ('stats', text_data()),  # returns a list of dicts
                    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ])),

            ],
        ))
    ])

    # In[18]:

    # features and labels
    X = apolitcal[['parsed', 'subj', 'difficult_words', 'pol', 'readability', 'unique_words']]
    y = apolitcal['political leaning']
    print(y)

    # In[19]:

    # split training and testing samples
    seed = 12
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(y)
    print(y)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)

    # In[20]:

    # fit pipeline
    pipeline.fit(x_train)

    # In[21]:

    train_vec = pipeline.transform(x_train)
    test_vec = pipeline.transform(x_test)

    # In[22]:

    # models to test
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    clf_sv = LinearSVC(C=1, class_weight='balanced', multi_class='ovr', random_state=40, max_iter=10000)
    clf_nb = naive_bayes.MultinomialNB()
    clf_lr = LogisticRegression()
    clf_rf = RandomForestClassifier(n_estimators=800)

    # In[23]:

    # cross val score of models
    from sklearn.model_selection import cross_val_score
    clfs = {'SVC': clf_sv, 'NB': clf_nb, 'LR': clf_lr, 'RF': clf_rf}
    cv = 3
    for name, clf in clfs.items():
        scores = cross_val_score(clf, test_vec, y_test, cv=cv, scoring="accuracy")
        print(name)
        print(scores)
        print(("Mean score: {0:.3f} (+/-{1:.3f})").format(
            numpy.mean(scores), numpy.std(scores)))
        print('---------------------------')

    # In[24]:

    apolitcal.groupby('political leaning')['unique_words'].describe()

    # In[25]:

    apolitcal.groupby('political leaning')['difficult_words'].describe()

    # In[26]:

    apolitcal.groupby('political leaning')['pol'].describe()

    # In[27]:

    apolitcal.groupby('political leaning')['readability'].describe()

    # In[28]:

    apolitcal.groupby('political leaning')['subj'].describe()

    # In[29]:

    apolitcal_sub = apolitcal.groupby('political leaning')['subj'].aggregate('mean')
    apolitcal_sub

    df_apolitcal_sub = pd.DataFrame(apolitcal_sub)
    df_apolitcal_sub

    # In[30]:

    df_sub_c = df_apolitcal_sub.iloc[0, 0]
    df_sub_l = df_apolitcal_sub.iloc[1, 0]

    # In[31]:

    apolitcal_pol = apolitcal.groupby('political leaning')['pol']
    apolitcal_pol.aggregate('mean')

    df_apolitcal_pol = pd.DataFrame(apolitcal_pol.aggregate('mean'))
    df_apolitcal_pol

    # In[32]:

    df_pol_c = df_apolitcal_pol.iloc[0, 0]
    df_pol_l = df_apolitcal_pol.iloc[1, 0]

    # In[33]:

    apolitcal_dw = apolitcal.groupby('political leaning')['difficult_words']
    apolitcal_dw.aggregate('mean')
    df_apolitcal_dw = pd.DataFrame(apolitcal_dw.aggregate('mean'))
    df_apolitcal_dw

    # In[34]:

    df_dw_c = df_apolitcal_dw.iloc[0, 0]
    df_dw_c

    df_dw_l = df_apolitcal_dw.iloc[1, 0]
    df_dw_l

    # In[35]:

    print("Document Count")
    print(apolitcal.groupby('political leaning')['parsed'].count())

    print("Word Count")
    apolitcal.groupby('political leaning').apply(lambda x: x.parsed.apply(lambda x: len(x.split())).sum())

    # In[36]:

    df = apolitcal["political leaning"]
    df

    # In[38]:

    # plotting the bar chart in contrast for conservative vs Liberal
    # Summary - Convservatives uses more difficult words

    import plotly.graph_objects as go
    text_bar = ['Difficult words', 'Subjective Text', 'Polarized Text']

    fig = go.Figure(data=[
        go.Bar(name='Liberal', x=text_bar, y=[df_dw_l, df_sub_l, df_pol_l]),
        go.Bar(name='Conservative', x=text_bar, y=[df_dw_c, df_sub_c, df_pol_c])
    ])
    # Change the bar mode

    fig.update_layout(barmode='stack')

    # Change the bar mode
    import json

    data = [fig]
    import plotly
    graphJson = json.dump(data,cls = plotly.utils.PlotlyJsonEncoder)
    return render_template("result.html", plot=graphJson)


# Create database classes
@app.before_first_request
def setup():
    # Recreate database each time for demo
    # db.drop_all()
    print('here')

@app.route("/")
def home():
    """Render Home Page."""

    return render_template("index.html")


@app.route("/index.html")
def index():
    """Render Home Page."""

    return render_template("index.html")




if __name__ == "__main__":
    app.run(debug=True)





