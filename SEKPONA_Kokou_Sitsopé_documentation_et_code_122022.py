"""
-Dataset Url: https://www.kaggle.com/competitions/nlp-getting-started/data
-Kaggle notebook url: https://www.kaggle.com/code/kokousitsope/disasters-tweet-competition
- Pylint score: 7.78/10
"""
"""import dependences"""
import re
import string
import warnings
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pandarallel import pandarallel
from nltk.tokenize import word_tokenize

# Gensim for doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Sklearn dependences
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Tensorflow dependences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_hub as hub
import spacy

# Initialisation and download
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
pandarallel.initialize(progress_bar=True, nb_workers=6)

"""Load Dataset"""


df_train = pd.read_csv("/content/drive/MyDrive/tweet_data/train.csv")
df_test = pd.read_csv("/content/drive/MyDrive/tweet_data/test.csv")
df_submit = pd.read_csv(
    '/content/drive/MyDrive/tweet_data/sample_submission.csv')

# Dataset informations
print("train set informations: ", df_train.info())

print("test set informations: ", df_test.info())

# Check if dataset have Nan values
print("Train set NaN: ", df_train.isnull().sum())
print("Test set Nan: ", df_test.isnull().sum())

# Plot graph

df_train.dtypes.value_counts().plot(kind="pie")
df_test.dtypes.value_counts().plot(kind="pie")

# Check the duplicated data
print('duplicated lines: ', df_train.duplicated().sum())
print('duplicated lines: ', df_train[[
      'location', 'text', 'target']].duplicated().sum())

# Drop duplicated rows
print(df_train.shape)
df_train.drop_duplicates(subset=['location', 'text', 'target'], inplace=True)
print(df_train.shape)


"""Plot the distribution of dataset"""


# set the figure size
plt.figure(figsize=(9, 5))

# set the style
plt.style.use('seaborn-darkgrid')

# set the colors
colors = ['lightskyblue', 'lightcoral']

# generate a pie plot
plt.pie(df_train['target'].value_counts(), explode=(0, 0.05),
        labels=["Non catastrophique", "catastrophique"],
        autopct="%0.2f%%", textprops={'fontsize': 14}, shadow=True,
        startangle=90, colors=colors)

# add a title
plt.title('Repartition des tweets', size=16, y=0.93)

# show the plot
plt.show()

"""Let's visualize the unprocessed text as a word cloud.
 The size of text shows the frequency that the word appears in the dataset."""

# Set random state
RANDOM_STATE = 4041

# concat all the text for each labels
non_disaster_text = [''.join(t)
                     for t in df_train[df_train['target'] == 0]['text']]
NON_DISASTER = ' '.join(map(str, non_disaster_text))
disaster_text = [''.join(t) for t in df_train[df_train['target'] == 1]['text']]
DISASTER_STRING = ' '.join(map(str, disaster_text))

# generate word clouds
non_disaster_cloud = WordCloud(
    width=800,
    height=400,
    max_words=500,
    background_color='white',
    random_state=RANDOM_STATE).generate(NON_DISASTER)
disaster_cloud = WordCloud(
    width=800,
    height=400,
    max_words=500,
    random_state=RANDOM_STATE).generate(DISASTER_STRING)

# create subplots for the generated clouds
fig, axes = plt.subplots(1, 2, figsize=(20, 20))
axes[0].imshow(non_disaster_cloud, interpolation='bilinear')
axes[1].imshow(disaster_cloud, interpolation='bilinear')

# turn the axis off
print([ax.axis('off') for ax in axes])

# add titles
axes[0].set_title('Normal tweet', fontsize=16)
axes[1].set_title('Disaster tweet', fontsize=16)

# show the figure
plt.show()


"""🧾 Text Preprocessing - Part I

Let's start preprocessing our text by removing the parts below:
- URLs
- HTML tags
- character references
- non-printable characters
- numeric values

We'll come back to the preprocessing step after creating some new features.

"""



def remove_url(text):
    """Define a function that removes URLs from the text"""
    text = re.sub(
        r'((?:https?|ftp|file)://[-\w\d+=&@#/%?~|!:;\.,]*)',
        '',
        text)
    return text


# remove URLs from the text and show the modified text in a new column
df_train['text_cleaned'] = df_train['text'].apply(remove_url)
df_test['text_cleaned'] = df_test['text'].apply(remove_url)


def remove_html(text):
    """Remove HTML tags"""
    text = re.sub(r'<.*?>', '', text)
    return text


# remove HTML tags
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_html)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_html)

""" Remove Character References"""


def remove_references(text):
    """define a function to remove character references (e.g., &lt;, &amp;, &nbsp;)"""
    text = re.sub(r'&[a-zA-Z]+;?', '', text)
    return text


# remove character references
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_references)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_references)

"""### Remove Non-printable Characters"""

# check which characters are printable (ASCII)
print('Printables characters: ', string.printable)


def remove_non_printable(text):
    """define a function that removes non-printable characters"""
    text = ''.join([word for word in text if word in string.printable])
    return text


# remove non-printable characters
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_non_printable)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_non_printable)

"""Remove Numeric Values
   Remove numeric values, including mixtures of alphabetical
   characters and numeric values such as 'M194', '5km'.
"""


def remove_num(text):
    """"define a function that removes numeric values and mixtures"""
    text = re.sub(r'\w*\d+\w*', '', text)
    return text


# remove numeric values and mixtures
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_num)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_num)

# check the results
df_train.tail()

# check the results
df_test.tail()

"""📐 Feature Engineering

Below are 10 features we're going to create:

- Number of **sentences**
- Number of **words**
- Number of **characters**
- Number of **hashtags**
- Number of **mentions**
- Number of **all caps words**
- Average **length of words**
- Number of **proper nouns (PROPN)**
- Number of **non-proper nouns (NOUN)**
- Percentage of characters that are **punctuation**


"""

"""Number of Sentences"""

# create a new feature for the number of sentences in each Tweet
df_train['sent_count'] = df_train['text'].apply(
    nltk.tokenize.sent_tokenize).apply(len)
df_test['sent_count'] = df_test['text'].apply(
    nltk.tokenize.sent_tokenize).apply(len)

"""Number of Words"""

# create a new feature for the number of words
df_train['word_count'] = df_train['text'].apply(
    nltk.tokenize.word_tokenize).apply(len)
df_test['word_count'] = df_test['text'].apply(
    nltk.tokenize.word_tokenize).apply(len)

"""Number of Characters"""

# create a new feature for the number of characters excluding white spaces
df_train['char_count'] = df_train['text'].apply(
    lambda x: len(x) - x.count(" "))
df_test['char_count'] = df_test['text'].apply(lambda x: len(x) - x.count(" "))

"""Number of Hashtags"""


def hash_count(strings):
    """define a function that returns the number of hashtags in a string"""
    words = strings.split()
    hashtags = [w for w in words if w.startswith('#')]
    return len(hashtags)


# create a new feature for the number of hashtags
df_train['hash_count'] = df_train['text'].apply(hash_count)
df_test['hash_count'] = df_test['text'].apply(hash_count)

"""Number of Mentions"""


def ment_count(strings):
    """define a function that returns the number of mentions in a string"""
    words = strings.split()
    mentions = [w for w in words if w.startswith('@')]
    return len(mentions)


# create a new feature for the number of mentions
df_train['ment_count'] = df_train['text'].apply(ment_count)
df_test['ment_count'] = df_test['text'].apply(ment_count)

"""Number of All Caps Words"""


def all_caps_count(strings):
    """define a function that returns the number of words in all CAPS"""
    words = strings.split()
    pattern = re.compile(r'\b[A-Z]+[A-Z]+\b')
    capswords = [w for w in words if w in re.findall(pattern, string)]
    return len(capswords)


# create a new feature for the number of words in all CAPS
df_train['all_caps_count'] = df_train['text'].apply(all_caps_count)
df_test['all_caps_count'] = df_test['text'].apply(all_caps_count)

"""Average Length of words"""


def avg_word_len(strings):
    """define a function that returns the average length of words"""
    words = strings.split()
    total_len = sum([len(words[i]) for i in range(len(words))])
    avg_len = round(total_len / len(words), 2)
    return avg_len


# create a new feature for the average length of words
df_train['avg_word_len'] = df_train['text'].apply(avg_word_len)
df_test['avg_word_len'] = df_test['text'].apply(avg_word_len)

""" Number of Proper Nouns (PROPN)
    It is known that fake news tends to use more proper nouns
    than real news ([this article](https://arxiv.org/pdf/1703.09398.pdf)
    is a great resource to learn about how NLP helps us detect the fake news).
    Would the number of proper nouns in Tweets tell us anything about whether
    a given Tweet is an actual disaster-related Tweet or not? Let's try it out.
"""


def propn_count_nltk(text):
    """define a function using nltk that returns the number of proper nouns in the text"""
    tokens = nltk.word_tokenize(text)
    tagged = [token for token in nltk.pos_tag(tokens)]
    propn_count = len(
        [token for (token, tag) in tagged if tag == 'NNP' or tag == 'NNPS'])
    return propn_count


# create a new feature for the number of proper nouns
df_train['propn_count_nltk'] = df_train['text'].apply(propn_count_nltk)
df_test['propn_count_nltk'] = df_test['text'].apply(propn_count_nltk)

# check the results
df_train[['id', 'text', 'text_cleaned', 'propn_count_nltk']].head()

"""Looking at the results, we can easily tell **nltk** did not do a good job
detecting proper nouns here. The first text, "Our Deeds are the Reason of
 this #earthquake May ALLAH Forgive us all" doesn't seem to have 4 proper nouns.
  Let's check which tokens were tagged as proper nouns."""

# test how nltk worked with the first text
STRING = "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"
print([(token, tag) for (token, tag) in nltk.pos_tag(
    nltk.word_tokenize(STRING)) if tag == 'NNP'])

"""Non-proper nouns beginning with a capital letter were tagged as proper nouns!
   Would it have been tagged correctly if the string first had been converted to
   lowercase?"""

# test how nltk works with the first text after lowercasing it
STRING = "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"
print([(token, tag) for (token, tag) in nltk.pos_tag(
    nltk.word_tokenize(STRING.lower())) if tag == 'NNP'])

"""No, now with the lowercased text, nltk does not tag "allah" as a proper noun.
   Let's try with **spaCy** this time."""

# load the model
nlp = spacy.load('en_core_web_sm')

# check the same string with spaCy
STRING = "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"
print([(token.text, token.pos_)
      for token in nlp(STRING) if token.pos_ == 'PROPN'])

"""SpaCy correctly picked up the proper noun from the string. Let's create
   the feature of the number of proper nouns in the text with spaCy and remove
   the one we previously created with nltk."""


def propn_count(text, my_model=nlp):
    """define a function that returns number of proper nouns with spaCy"""
    doc = my_model(text)
    pos = [token.pos_ for token in doc]
    return pos.count('PROPN')


# create a new feature for numbers of proper nouns
df_train['propn_count'] = df_train['text'].apply(propn_count)
df_test['propn_count'] = df_test['text'].apply(propn_count)

# remove 'propn_count_nltk' columns
df_train = df_train.drop(['propn_count_nltk'], axis=1)
df_test = df_test.drop(['propn_count_nltk'], axis=1)

# check the results
df_train[['id', 'text', 'text_cleaned', 'propn_count']].head()

"""SpaCy is not perfect, either - "La Ronge" in the second
   text (id:4) is one proper noun not two, but it is clear
   that spaCy still performs better than nltk on this specific task.
    Let's use spaCy for the next feature as well.

### Number of Non-proper Nouns (NOUN)
"""



def noun_count(text, my_model=nlp):
    """define a function that returns number of non-proper nouns"""
    doc = my_model(text)
    pos = [token.pos_ for token in doc]
    return pos.count('NOUN')


# create a new feature for numbers of non-proper nouns
df_train['noun_count'] = df_train['text'].parallel_apply(noun_count)
df_test['noun_count'] = df_test['text'].parallel_apply(noun_count)

""" Percentage of Characters that are Punctuation"""


def punc_per(text):
    """define a function that returns the percentage of punctuation"""
    total_count = len(text) - text.count(" ")
    punc_count = sum([1 for c in text if c in string.punctuation])
    if punc_count != 0 and total_count != 0:
        return round(punc_count / total_count * 100, 2)
    else:
        return 0


# create a new feature for the percentage of punctuation in text
df_train['punc_per'] = df_train['text'].apply(punc_per)
df_test['punc_per'] = df_test['text'].apply(punc_per)

# check the results
df_train.tail()

# check the results
df_test.tail()

"""🧾 Text Preprocessing - Part II
    Let's resume our text preprocessing and **lemmatize**
    the text and make it **lowercase**. We'll also **remove
    repeated characters in elongated words, as well as mentions,
    stopwords, and punctuation**. We'll keep hashtags as they may
    provide valuable insights in this particular project.


"""


# lemmatize the text
df_train['text_cleaned'] = df_train['text_cleaned'].parallel_apply(
    lambda x: ' '.join([t.lemma_ for t in nlp(x)]))
df_test['text_cleaned'] = df_test['text_cleaned'].parallel_apply(
    lambda x: ' '.join([t.lemma_ for t in nlp(x)]))

"""Convert the Text to Lowercase"""

# lowercase the text
df_train['text_cleaned'] = [t.lower() for t in df_train['text_cleaned']]
df_test['text_cleaned'] = [t.lower() for t in df_test['text_cleaned']]

"""Remove Repeated Charcters in Elongated Words"""


def remove_repeated(text):
    """define a function that removes repeated characters in elongated words"""

    elongated = re.compile(r'(\S*?)([a-z])\2{2,}')
    text = elongated.sub(r'\1\2', text)
    return text


# remove repeated characters in elongated words
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_repeated)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_repeated)

"""Remove Mentions"""


def remove_mention(text):
    """define a function that removes mentions"""
    text = re.sub(r'@\w+', '', text)
    return text


# remove mentions
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_mention)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_mention)

"""Remove Stopwords"""


def remove_stopwords(text):
    """define a function that removes stopwords"""
    stopwords = nlp.Defaults.stop_words
    text_nostop = ' '.join(
        [token for token in text.split() if token not in stopwords])
    return text_nostop


# remove stopwords
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_stopwords)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_stopwords)

"""Remove Punctuation"""


def remove_punct(text):
    """define a function to remove punctuation"""
    punct = string.punctuation
    text_nospunct = ' '.join(
        [token for token in text.split() if token not in punct])
    return text_nospunct


# remove punctuation
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_punct)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_punct)

# check the results
df_train[['id', 'text', 'text_cleaned']].head()

# check the results
df_test[['id', 'text', 'text_cleaned']].head()

"""Raw vs. Preprocessed Text with Word Clouds

Let's generate word clouds of preprocessed text.
"""

RANDOM_STATE = 123
# concat all the preprocessed text for both labels
non_disaster_processed = [
    ''.join(t) for t in df_train[df_train['target'] == 0]['text_cleaned']]
non_disaster_processed_s = ' '.join(map(str, non_disaster_processed))
disaster_processed = [
    ''.join(t) for t in df_train[df_train['target'] == 1]['text_cleaned']]
DISASTER_PROCESSED = ' '.join(map(str, disaster_processed))

# generate word clouds of the preprocessed text
non_disaster_processed_wc = WordCloud(
    width=800,
    height=400,
    max_words=500,
    background_color='white',
    random_state=RANDOM_STATE).generate(non_disaster_processed_s)
disaster_processed_wc = WordCloud(
    width=800,
    height=400,
    max_words=500,
    random_state=RANDOM_STATE).generate(DISASTER_PROCESSED)

# create subplots for the generated clouds
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
axes[0, 0].imshow(non_disaster_cloud, interpolation='bilinear')
axes[0, 1].imshow(disaster_cloud, interpolation='bilinear')
axes[1, 0].imshow(non_disaster_processed_wc, interpolation='bilinear')
axes[1, 1].imshow(disaster_processed_wc, interpolation='bilinear')

# turn the axis off
print([ax.axis('off') for ax in axes.ravel()])

# add titles
axes[0, 0].set_title('Non-disaster Tweets (raw)', fontsize=16)
axes[0, 1].set_title('Disaster Tweets (raw)', fontsize=16)
axes[1, 0].set_title('Non-disaster Tweets (preprocessed)', fontsize=16)
axes[1, 1].set_title('Disaster Tweets (preprocessed)', fontsize=16)

# show the figure
plt.show()

"""Now it's easier to see the frequently used words that actually are meaningful.
   It also seems like **more disaster-related words are showing on the word cloud
   of real disaster Tweets**.

   ### 📊 Visualizing Differences
   Let's visualize some of the features we've created and see if there are easy-to-tell
   differences between disaster and non-disaster Tweets in our training dataset.
"""

# store the features and their names in variables
features = [
    'sent_count',
    'word_count',
    'char_count',
    'hash_count',
    'ment_count',
    'all_caps_count',
    'avg_word_len',
    'propn_count',
    'noun_count',
    'punc_per']

# create the figure
fig = plt.figure(figsize=(20, 20))

# adjust the height of the padding between subplots to avoid overlapping
plt.subplots_adjust(hspace=0.3)

# add a centered suptitle to the figure
plt.suptitle(
    "Difference in Features, Disaster vs. Non-disaster",
    fontsize=20,
    y=0.91)

# generate the histograms in a for loop
for i, feature in enumerate(features):
    # add a new subplot iteratively
    ax = plt.subplot(4, 3, i + 1)
    ax = df_train[df_train['target'] == 0][feature].hist(
        alpha=0.5, label='Non-disaster', bins=40, color='royalblue', density=True)
    ax = df_train[df_train['target'] == 1][feature].hist(
        alpha=0.5, label='Disaster', bins=40, color='lightcoral', density=True)

    # set x_label, y_label, and legend
    ax.set_xlabel(features[i], fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.legend(loc='upper right', fontsize=14)

# shot the figure
plt.show()

"""- We'll use the four features, **word_count, char_count,
    avg_word_len, and punc_per**, for our models as they show bigger
    differences in distributions than other features we've created.

    - Note: The y-axis in the plots above is probability density,
    not # of Tweets due to the different size of disaster/non-disaster Tweets.

    Now let's move on and start building our models!

"""


"""Features extraction
    Vectorizing Text:
       -TfidfVectorizer
       -Doc2Vec
"""


# instantiate the vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=1)

# fit and transform
tif = tfidf.fit(df_train['text_cleaned'])
X_tfidf = tif.transform(df_train['text_cleaned'])

# create a dataframe from the sparse matrix
X_tfidf_df = pd.DataFrame(X_tfidf.toarray())

# check the dataframe
X_tfidf_df.head()

vocabulary_size = tfidf.get_feature_names_out().shape[0]

# get the feature names from our stored vectorizer and assign them to X_tfidf_df
# to avoid getting 'FutureWarning: Feature names only support names that
# are strings.'
X_tfidf_df.columns = tfidf.get_feature_names_out()

# check the column names
print("tfif columns: ", X_tfidf_df.columns)

print("tfidf shape: ", X_tfidf_df.shape)

X_tfidf_df[['word_count', 'char_count', 'avg_word_len', 'punc_per']
           ] = df_train[['word_count', 'char_count', 'avg_word_len', 'punc_per']]


# check the shape
X_features = X_tfidf_df
print(X_features.shape)

# check the dataframe
print(X_features.head())

x_train_tf, x_val_tf, y_train_tf, y_val_tf = train_test_split(
    X_features, df_train['target'], test_size=0.1, random_state=123)

# Doc2vec

data = df_train['text_cleaned'].tolist()
tagged_data = [
    TaggedDocument(
        words=word_tokenize(d),
        tags=[
            str(i)]) for i,
    d in enumerate(data)]

print(tagged_data[0])

MAX_EPOCHS = 100

ALPHA = 0.025

my_model = Doc2Vec(alpha=ALPHA,
                   min_alpha=0.00025,
                   min_count=1,
                   dm=0
                   )

my_model.build_vocab(tagged_data)

for epoch in range(MAX_EPOCHS):
    print('iteration {0}'.format(epoch))
    my_model.train(tagged_data,
                   total_examples=my_model.corpus_count,
                   epochs=1)
    # decrease the learning rate
    my_model.alpha -= 0.0002
    # fix the learning rate, no decay
    my_model.min_alpha = my_model.alpha

my_model.save("d2v.model")
print("Model Saved")

doc_data = []
for i in range(len(tagged_data)):
    doc_data.append(my_model.infer_vector(tagged_data[i].words))
doc_df = pd.DataFrame(doc_data)
doc_df.head()

print(doc_df.shape)

doc_data = []
for i in range(len(df_test['text_cleaned'].tolist())):
    doc_data.append(
        my_model.infer_vector(
            df_test['text_cleaned'].tolist()[i].split(' ')))
x_test_doc = pd.DataFrame(doc_data)
x_test_doc.head()

x_train_doc, x_val_doc, y_train_doc, y_val_doc = train_test_split(
    doc_df, df_train['target'], test_size=0.1, random_state=123)


"""Modélisation

## Logistic Regressor
"""

# Model:

l_model = LogisticRegression()


def clean_dataset(data, label):
    """This function convert in the dataset infinite values to float"""
    data['label'] = label.values
    assert isinstance(data, pd.DataFrame), "df needs to be a pd.DataFrame"
    data.dropna(inplace=True)
    indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
    data = data[indices_to_keep].astype(np.float64)
    label = data['label']
    data.drop(columns=['label'], inplace=True)
    return data, label


def clean_test_dataset(data):
    """This function convert in the dataset infinite values to float"""
    assert isinstance(data, pd.DataFrame), "df needs to be a pd.DataFrame"
    data.dropna(inplace=True)
    indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
    data = data[indices_to_keep].astype(np.float64)
    return data


def logistic_regressor(data_train, label_train):
    """Fonction of logistic regressor model"""
    n_jobs = [-1]

    hyperparameters = {"C": [0.5, 1],
                       "l1_ratio": [0.5, 1],
                       "max_iter": [100, 300, 1000],
                       'n_jobs': n_jobs,

                       }
    print(
        GridSearchCV(
            l_model,
            param_grid={},
            scoring='accuracy').get_params())
    data_train, label_train = clean_dataset(data_train, label_train)
    grid = GridSearchCV(
        l_model,
        hyperparameters,
        cv=KFold(
            4,
            shuffle=True,
            random_state=123),
        scoring='accuracy',
        verbose=2)
    grid.fit(data_train, label_train)
    return grid


# Tfidf

model_tf = logistic_regressor(x_train_tf, y_train_tf)

x_val_tf, y_val_tf = clean_dataset(x_val_tf, y_val_tf)
score = model_tf.score(x_val_tf, y_val_tf)
print("validation_score: ", score)

x_test_tf = tif.transform(df_test['text_cleaned']).toarray()
x_test_tf = pd.DataFrame(x_test_tf)
x_test_tf[['word_count', 'char_count', 'avg_word_len', 'punc_per']] = df_test[
    ['word_count', 'char_count', 'avg_word_len', 'punc_per']]
x_test_tf = clean_test_dataset(x_test_tf)
pred = model_tf.predict(x_test_tf)
print("accuracy: ", accuracy_score(model_tf.predict(x_val_tf), y_val_tf))

pred = [int(i) for i in pred]

df_test['target'] = pred
df_submit['target'] = pred
df_submit.head()

print(df_test[['text_cleaned', "target"]])

print(df_test.target.describe())

df_submit.to_csv('sample_submission.csv', index=False)

# Doc2vec with adding df  columns


model_doc = logistic_regressor(x_train_doc, y_train_doc)

score = model_doc.score(x_val_doc, y_val_doc)
print(score)

print("accuracy: ", accuracy_score(model_doc.predict(x_val_doc), y_val_doc))
pred_doc = model_doc.predict(x_test_doc)

pred_doc = [int(i) for i in pred_doc]
np.unique(pred_doc)

df_test['target'] = pred_doc
df_submit['target'] = pred_doc
df_submit.head()
df_submit.to_csv('sample_submission.csv', index=False)

# Transformers


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"),
             Dense(embed_dim), ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocabulary_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(
            input_dim=vocabulary_size,
            output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, val):
        maxlen = tf.shape(val)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        final = self.token_emb(val)
        return final + positions


tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train['text_cleaned'])
vocabulary_size = len(tokenizer.word_index) + 1
print(vocabulary_size)


def transformer_model(data_train, label_train, data_val, y_val):
    """Transformer model
    :arg
      -data_train: the train-set
      -label_train: the label of train set
      -data_val: the validation set
      -y_val: label of validation set
    """

    maxlenth = data_train.shape[1]
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    vocabulary_size = data_train.shape[1]  # Only consider the top 20k words
    maxlen = data_train.shape[1]
    inputs = Input(shape=(maxlenth,))
    embedding_layer = TokenAndPositionEmbedding(
        maxlen, vocabulary_size, embed_dim)
    embed = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    transform = transformer_block(embed)
    pool = GlobalAveragePooling1D()(transform)
    drop = Dropout(0.1)(pool)
    out = Dense(20, activation="relu")(drop)
    out = Dropout(0.1)(out)
    outputs = Dense(2, activation="softmax")(out)

    my_model = Model(inputs=inputs, outputs=outputs)

    my_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    model_history = my_model.fit(data_train, label_train,
                                 batch_size=64, epochs=5,
                                 validation_data=(data_val, y_val)
                                 )
    results = my_model.evaluate(data_val, y_val, verbose=2)

    for name, value in zip(my_model.metrics_names, results):
        print(f"{name}, {value}")
    return my_model, model_history


# Tfidf Vectorizer

# instantiate the vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=2000)

# fit and transform
tif = tfidf.fit(df_train['text_cleaned'])
X_tfidf = tif.transform(df_train['text_cleaned'])
dftf = pd.DataFrame(X_tfidf.toarray())
x_train_tf, x_val_tf, y_train_tf, y_val_tf = train_test_split(
    dftf, df_train['target'], test_size=0.2)

x_test_tf = pd.DataFrame(tfidf.transform(df_test['text_cleaned']).toarray())

data_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train_tf.values, maxlen=2000)
data_val = tf.keras.preprocessing.sequence.pad_sequences(
    x_val_tf.values, maxlen=2000)
y_train_tf = np.array(y_train_tf)
y_val_df = np.array(y_val_tf)

model_tf1, model_history = transformer_model(
    data_train, y_train_tf, data_val, y_val_tf)

pred = model_tf1.predict(x_test_tf)
pred = [round(i[0]) for i in pred]

df_submit['target'] = pred
df_submit.to_csv('sample_submission.csv', index=False)

# Doc2vec

x_train_ = tf.keras.preprocessing.sequence.pad_sequences(
    x_train_doc.values, maxlen=100)
data_val = tf.keras.preprocessing.sequence.pad_sequences(
    x_val_doc.values, maxlen=100)
label_train = np.array(y_train_doc)
label_val = np.array(y_val_doc)

model_doc2, model_history = transformer_model(
    x_train_, label_train, data_val, label_val)

pred_doc = model_doc2.predict(x_test_doc)
pred_doc = [round(i[0]) for i in pred_doc]

df_submit['target'] = pred_doc
df_submit.to_csv('sample_submission.csv', index=False)

# Model Pré-entrainé

EMBEDDING = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(EMBEDDING, input_shape=[],
                           dtype=tf.string, trainable=True)

my_model = Sequential()
my_model.add(hub_layer)
my_model.add(Dense(16, activation='relu'))
my_model.add(Dense(1, activation='sigmoid'))

my_model.summary()

my_model.compile(optimizer='adam',
                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 metrics=['accuracy'])

model_history = my_model.fit(df_train['text_cleaned'], df_train['target'],
                             epochs=5,
                             validation_split=0.1,
                             verbose=1)

pred = my_model.predict(df_test['text_cleaned'])

pred = [round(i[0]) for i in pred]
print(pred)

df_test['target'] = pred

df_submit['target'] = pred
df_submit.head()

df_submit.to_csv('sample_submission.csv', index=False)

print(df_test[df_test['target'] == 0]['text_cleaned'])
