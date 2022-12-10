# -*- coding: utf-8 -*-
"""P8_.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GSMFHKwAij0prw8cxKVKU8uin94DniT-

# Bibliothèques et base de données
"""

#!pip install pandarallel bs4 transformers

# Commented out IPython magic to ensure Python compatibility.
# import the libraries
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re, string
import nltk
import spacy
from pandarallel import pandarallel
import seaborn as sns
pandarallel.initialize(progress_bar=True, nb_workers=6)

#Chargement des jeux de données
df_train=pd.read_csv("/content/drive/MyDrive/tweet_data/train.csv")
df_test=pd.read_csv("/content/drive/MyDrive/tweet_data/test.csv")
df_submit=pd.read_csv('/content/drive/MyDrive/tweet_data/sample_submission.csv')

#Informations sur les bases de données:
df_train.info()

df_test.head()

df_test.info()

# Valeurs manquantes train
df_train.isnull().sum()

# Valeurs manquantes test
df_test.isnull().sum()

df_submit.head()

df_train.dtypes.value_counts().plot(kind="pie")

df_test.dtypes.value_counts().plot(kind="pie")

# check the duplicated data
df_train.duplicated().sum()

df_train[['location', 'text', 'target']].duplicated().sum()

"""Nous avons au total 70 Doublons dans notre dataset. Nous allons les supprimer"""

print(df_train.shape)
df_train.drop_duplicates(subset=['location', 'text', 'target'], inplace=True)
print(df_train.shape)

"""## Visualisation de l'équilibre de la dataset"""

# set the figure size
plt.figure(figsize=(9, 5))

# set the style
plt.style.use('seaborn-darkgrid')

# set the colors
colors = ['lightskyblue', 'lightcoral']

# generate a pie plot
plt.pie(df_train['target'].value_counts(), explode=(0, 0.05), labels=["Non catastrophique", "catastrophique"], 
        autopct="%0.2f%%", textprops={'fontsize': 14}, shadow=True, startangle=90, colors=colors)

# add a title
plt.title('Repartition des tweets', size=16, y=0.93)

# show the plot
plt.show()

"""Donc la dataset est presque équilibré.

### Word Clouds  Avec le test brut

Let's visualize the unprocessed text as a word cloud. The size of text shows the frequency that the word appears in the dataset.
"""

# Définition du random state
random_state = 4041

# bibliothèque
from wordcloud import WordCloud

# concat all the text for each labels
non_disaster_text = [''.join(t) for t in df_train[df_train['target']==0]['text']]
non_disaster_strings = ' '.join(map(str, non_disaster_text))
disaster_text = [''.join(t) for t in df_train[df_train['target']==1]['text']]
disaster_strings = ' '.join(map(str, disaster_text))

# generate word clouds
non_disaster_cloud = WordCloud(width=800, height=400, max_words=500, background_color='white', random_state=random_state).generate(non_disaster_strings)
disaster_cloud = WordCloud(width=800, height=400, max_words=500, random_state=random_state).generate(disaster_strings)

# create subplots for the generated clouds
fig, axes = plt.subplots(1, 2, figsize = (20,20))
axes[0].imshow(non_disaster_cloud, interpolation='bilinear')
axes[1].imshow(disaster_cloud, interpolation='bilinear')

# turn the axis off
[ax.axis('off') for ax in axes]

# add titles
axes[0].set_title('Tweets non catastrophiques', fontsize=16)
axes[1].set_title('Tweets Catastrophiques', fontsize=16)

# Affichage de la figure
plt.show()

"""Nous pouvons reconnaître certains mots liés à une catastrophe (par exemple, « incendie », « inondation » et « tempête »), mais la différence entre les Tweets de catastrophe et non liés à une catastrophe n'est pas facile à faire. Des mots comme "t" et "co" (reflétant des liens raccourcis sur Twitter) dominent les nuages de mots, mais ne fournissent pas beaucoup de sens utile. **Comparons-les plus tard avec des nuages ​​de mots de données prétraitées.**

# Traitement du text
"""

df_train.columns

"""***

## <a id="preprocess1">🧾 Text Preprocessing - Part I</a>

Let's start preprocessing our text by removing the parts below: 
- URLs
- HTML tags
- character references
- non-printable characters
- numeric values

We'll come back to the preprocessing step after creating some new features.

### Remove URLs
"""

# define a function that removes URLs from the text
def remove_url(text):
    text = re.sub(r'((?:https?|ftp|file)://[-\w\d+=&@#/%?~|!:;\.,]*)', '', text)
    return text

# remove URLs from the text and show the modified text in a new column
df_train['text_cleaned'] = df_train['text'].apply(remove_url)
df_test['text_cleaned'] = df_test['text'].apply(remove_url)

"""### Remove HTML tags"""

# define a function that removes HTML tags
def remove_HTML(text):
    text = re.sub(r'<.*?>', '', text)
    return text

# remove HTML tags
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_HTML)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_HTML)

"""### Remove Character References"""

# define a function to remove character references (e.g., &lt;, &amp;, &nbsp;)
def remove_references(text):
    text = re.sub(r'&[a-zA-Z]+;?', '', text)
    return text

# remove character references
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_references)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_references)

"""### Remove Non-printable Characters"""

# check which characters are printable (ASCII)
string.printable

# define a function that removes non-printable characters
def remove_non_printable(text):
    text = ''.join([word for word in text if word in string.printable])
    return text

# remove non-printable characters
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_non_printable)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_non_printable)

"""### Remove Numeric Values
Remove numeric values, including mixtures of alphabetical characters and numeric values such as 'M194', '5km'.
"""

# define a function that removes numeric values and mixtures
def remove_num(text):
    text = re.sub(r'\w*\d+\w*', '', text)
    return text

# remove numeric values and mixtures
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_num)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_num)

# check the results
df_train.tail()

# check the results
df_test.tail()

"""## <a id="engineer">📐 Feature Engineering</a>

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

### Number of Sentences
"""

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# create a new feature for the number of sentences in each Tweet
df_train['sent_count'] = df_train['text'].apply(nltk.tokenize.sent_tokenize).apply(len)
df_test['sent_count'] = df_test['text'].apply(nltk.tokenize.sent_tokenize).apply(len)

"""### Number of Words"""

# create a new feature for the number of words
df_train['word_count'] = df_train['text'].apply(nltk.tokenize.word_tokenize).apply(len)
df_test['word_count'] = df_test['text'].apply(nltk.tokenize.word_tokenize).apply(len)

"""### Number of Characters"""

# create a new feature for the number of characters excluding white spaces
df_train['char_count'] = df_train['text'].apply(lambda x: len(x) - x.count(" "))
df_test['char_count'] = df_test['text'].apply(lambda x: len(x) - x.count(" "))

"""### Number of Hashtags"""

# define a function that returns the number of hashtags in a string
def hash_count(string):
    words = string.split()
    hashtags = [w for w in words if w.startswith('#')]
    return len(hashtags)

# create a new feature for the number of hashtags
df_train['hash_count'] = df_train['text'].apply(hash_count)
df_test['hash_count'] = df_test['text'].apply(hash_count)

"""### Number of Mentions"""

# define a function that returns the number of mentions in a string
def ment_count(string):
    words = string.split()
    mentions = [w for w in words if w.startswith('@')]
    return len(mentions)

# create a new feature for the number of mentions
df_train['ment_count'] = df_train['text'].apply(ment_count)
df_test['ment_count'] = df_test['text'].apply(ment_count)

"""### Number of All Caps Words"""

# define a function that returns the number of words in all CAPS
def all_caps_count(string):
    words = string.split()
    pattern = re.compile(r'\b[A-Z]+[A-Z]+\b')
    capsWords = [w for w in words if w in re.findall(pattern, string)]
    return len(capsWords)

# create a new feature for the number of words in all CAPS
df_train['all_caps_count'] = df_train['text'].apply(all_caps_count)
df_test['all_caps_count'] = df_test['text'].apply(all_caps_count)

"""### Average Length of words"""

# define a function that returns the average length of words
def avg_word_len(string):
    words = string.split()
    total_len = sum([len(words[i]) for i in range(len(words))])
    avg_len = round(total_len / len(words), 2)
    return avg_len

# create a new feature for the average length of words
df_train['avg_word_len'] = df_train['text'].apply(avg_word_len)
df_test['avg_word_len'] = df_test['text'].apply(avg_word_len)

"""### Number of Proper Nouns (PROPN)
It is known that fake news tends to use more proper nouns than real news ([this article](https://arxiv.org/pdf/1703.09398.pdf) is a great resource to learn about how NLP helps us detect the fake news). Would the number of proper nouns in Tweets tell us anything about whether a given Tweet is an actual disaster-related Tweet or not? Let's try it out.
"""

# define a function using nltk that returns the number of proper nouns in the text
def propn_count_nltk(text):    
    tokens = nltk.word_tokenize(text)
    tagged = [token for token in nltk.pos_tag(tokens)]
    propn_count = len([token for (token, tag) in tagged if tag == 'NNP' or tag == 'NNPS'])
    return propn_count

# create a new feature for the number of proper nouns
df_train['propn_count_nltk'] = df_train['text'].apply(propn_count_nltk)
df_test['propn_count_nltk'] = df_test['text'].apply(propn_count_nltk)

# check the results
df_train[['id', 'text', 'text_cleaned', 'propn_count_nltk']].head()

"""Looking at the results, we can easily tell **nltk** did not do a good job detecting proper nouns here. The first text, "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all" doesn't seem to have 4 proper nouns. Let's check which tokens were tagged as proper nouns."""

# test how nltk worked with the first text
string = "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"
print([(token, tag) for (token, tag) in nltk.pos_tag(nltk.word_tokenize(string)) if tag == 'NNP'])

"""Les noms non propres commençant par une majuscule ont été étiquetés comme noms propres ! Aurait-il été correctement étiqueté si la chaîne avait d'abord été convertie en minuscules ?"""

# test how nltk works with the first text after lowercasing it
string = "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"
print([(token, tag) for (token, tag) in nltk.pos_tag(nltk.word_tokenize(string.lower())) if tag == 'NNP'])

"""No, now with the lowercased text, nltk does not tag "allah" as a proper noun. Let's try with **spaCy** this time."""

# load the model
nlp = spacy.load('en_core_web_sm')

# check the same string with spaCy
string = "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"
print([(token.text, token.pos_) for token in nlp(string) if token.pos_=='PROPN'])

"""SpaCy correctly picked up the proper noun from the string. Let's create the feature of the number of proper nouns in the text with spaCy and remove the one we previously created with nltk."""

# define a function that returns number of proper nouns with spaCy
def propn_count(text, model=nlp):
    doc = model(text)
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

"""SpaCy is not perfect, either - "La Ronge" in the second text (id:4) is one proper noun not two, but it is clear that spaCy still performs better than nltk on this specific task. Let's use spaCy for the next feature as well.

### Number of Non-proper Nouns (NOUN)
"""

# define a function that returns number of non-proper nouns
def noun_count(text, model=nlp):
    doc = model(text)
    pos = [token.pos_ for token in doc]
    return pos.count('NOUN')

# create a new feature for numbers of non-proper nouns
df_train['noun_count'] = df_train['text'].parallel_apply(noun_count)
df_test['noun_count'] = df_test['text'].parallel_apply(noun_count)

"""### Percentage of Characters that are Punctuation"""

import string

# define a function that returns the percentage of punctuation
def punc_per(text):
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

"""***

***

## <a id="preprocess2">🧾 Text Preprocessing - Part II</a>
Let's resume our text preprocessing and **lemmatize** the text and make it **lowercase**. We'll also **remove repeated characters in elongated words, as well as mentions, stopwords, and punctuation**. We'll keep hashtags as they may provide valuable insights in this particular project.

### Lemmatization
"""

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=6)

# lemmatize the text
df_train['text_cleaned'] = df_train['text_cleaned'].parallel_apply(lambda x:' '.join([t.lemma_ for t in nlp(x)]))
df_test['text_cleaned'] = df_test['text_cleaned'].parallel_apply(lambda x:' '.join([t.lemma_ for t in nlp(x)]))

"""### Convert the Text to Lowercase"""

# lowercase the text
df_train['text_cleaned'] = [t.lower() for t in df_train['text_cleaned']]
df_test['text_cleaned'] = [t.lower() for t in df_test['text_cleaned']]

"""### Remove Repeated Charcters in Elongated Words"""

# define a function that removes repeated characters in elongated words
def remove_repeated(text):
    elongated = re.compile(r'(\S*?)([a-z])\2{2,}')
    text = elongated.sub(r'\1\2', text)
    return text

# remove repeated characters in elongated words
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_repeated)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_repeated)

"""### Remove Mentions"""

# define a function that removes mentions
def remove_mention(text):
    text = re.sub(r'@\w+', '', text)
    return text

# remove mentions
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_mention)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_mention)

"""### Remove Stopwords"""

# define a function that removes stopwords
def remove_stopwords(text):
    stopwords = nlp.Defaults.stop_words
    text_nostop = ' '.join([token for token in text.split() if token not in stopwords])
    return text_nostop

# remove stopwords
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_stopwords)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_stopwords)

"""### Remove Punctuation"""

# define a function to remove punctuation
def remove_punct(text):
    punct = string.punctuation
    text_nospunct = ' '.join([token for token in text.split() if token not in punct])
    return text_nospunct

# remove punctuation
df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_punct)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_punct)

# check the results
df_train[['id', 'text', 'text_cleaned']].head()

# check the results
df_test[['id', 'text', 'text_cleaned']].head()

"""### Raw vs. Preprocessed Text with Word Clouds

Let's generate word clouds of preprocessed text.
"""

# concat all the preprocessed text for both labels
non_disaster_processed = [''.join(t) for t in df_train[df_train['target']==0]['text_cleaned']]
non_disaster_processed_s = ' '.join(map(str, non_disaster_processed))
disaster_processed = [''.join(t) for t in df_train[df_train['target']==1]['text_cleaned']]
disaster_processed_s = ' '.join(map(str, disaster_processed))

# generate word clouds of the preprocessed text
non_disaster_processed_wc = WordCloud(width=800, height=400, max_words=500, background_color='white', random_state=random_state).generate(non_disaster_processed_s)
disaster_processed_wc = WordCloud(width=800, height=400, max_words=500, random_state=random_state).generate(disaster_processed_s)

# create subplots for the generated clouds
fig, axes = plt.subplots(2, 2, figsize = (20,10))
axes[0,0].imshow(non_disaster_cloud, interpolation='bilinear')
axes[0,1].imshow(disaster_cloud, interpolation='bilinear')
axes[1,0].imshow(non_disaster_processed_wc, interpolation='bilinear')
axes[1,1].imshow(disaster_processed_wc, interpolation='bilinear')

# turn the axis off
[ax.axis('off') for ax in axes.ravel()]

# add titles
axes[0,0].set_title('Non-disaster Tweets (raw)', fontsize=16)
axes[0,1].set_title('Disaster Tweets (raw)', fontsize=16)
axes[1,0].set_title('Non-disaster Tweets (preprocessed)', fontsize=16)
axes[1,1].set_title('Disaster Tweets (preprocessed)', fontsize=16)

# show the figure
plt.show()

"""Now it's easier to see the frequently used words that actually are meaningful. It also seems like **more disaster-related words are showing on the word cloud of real disaster Tweets**.

### 📊 Visualizing Differences
Let's visualize some of the features we've created and see if there are easy-to-tell differences between disaster and non-disaster Tweets in our training dataset.
"""

# store the features and their names in variables
features = ['sent_count', 'word_count', 'char_count', 'hash_count', 'ment_count', 'all_caps_count', 
            'avg_word_len', 'propn_count', 'noun_count', 'punc_per']

# create the figure
fig = plt.figure(figsize=(20, 20))

# adjust the height of the padding between subplots to avoid overlapping
plt.subplots_adjust(hspace=0.3)

# add a centered suptitle to the figure
plt.suptitle("Difference in Features, Disaster vs. Non-disaster", fontsize=20, y=0.91)

# generate the histograms in a for loop
for i, feature in enumerate(features):
    
    # add a new subplot iteratively
    ax = plt.subplot(4, 3, i+1)
    ax = df_train[df_train['target']==0][feature].hist(alpha=0.5, label='Non-disaster', bins=40, color='royalblue', density=True)
    ax = df_train[df_train['target']==1][feature].hist(alpha=0.5, label='Disaster', bins=40, color='lightcoral', density=True)
    
    # set x_label, y_label, and legend
    ax.set_xlabel(features[i], fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.legend(loc='upper right', fontsize=14)
    

# shot the figure
plt.show()

"""- We'll use the four features, **word_count, char_count, avg_word_len, and punc_per**, for our models as they show bigger differences in distributions than other features we've created.
- Note: The y-axis in the plots above is probability density, not # of Tweets due to the different size of disaster/non-disaster Tweets.

Now let's move on and start building our models!

***

# <a id=""> Features extraction</a>

## Vectorizing Text
"""

# import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# instantiate the vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=1)

# fit and transform
tif = tfidf.fit(df_train['text_cleaned'])
X_tfidf=tif.transform(df_train['text_cleaned'])

# create a dataframe from the sparse matrix
X_tfidf_df = pd.DataFrame(X_tfidf.toarray())

# check the dataframe
X_tfidf_df.head()

vocab_size=tfidf.get_feature_names_out().shape[0]

# get the feature names from our stored vectorizer and assign them to X_tfidf_df
# to avoid getting 'FutureWarning: Feature names only support names that are strings.'
X_tfidf_df.columns = tfidf.get_feature_names_out()

# check the column names
X_tfidf_df.columns

X_tfidf_df.shape, df_train.shape

X_tfidf_df[['word_count', 'char_count', 'avg_word_len', 'punc_per']]=df_train[['word_count', 'char_count', 'avg_word_len', 'punc_per']]

# create the new dataframe, X_features
#= pd.concat([df_train[['word_count', 'char_count', 'avg_word_len', 'punc_per']], X_tfidf_df], axis=1)

# check the shape
X_features= X_tfidf_df
X_features.shape

# check the dataframe
X_features.head()

from sklearn.model_selection import train_test_split
x_train_tf, x_val_tf, y_train_tf, y_val_tf=train_test_split(X_features,df_train['target'],  test_size=0.1, random_state=123)

"""##  Doc2vec"""

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

data=df_train['text_cleaned'].tolist()
tagged_data = [TaggedDocument(words=word_tokenize(d), tags=[str(i)]) for i, d in enumerate(data)]

tagged_data[0]

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(#size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm=0
                )
  
vocab=model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=1)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")

doc_data=[]
for i in range(len(tagged_data)):
  doc_data.append(model.infer_vector(tagged_data[i].words))
doc_df=pd.DataFrame(doc_data)
doc_df.head()

doc_df.shape

doc_data=[]
for i in range(len(df_test['text_cleaned'].tolist())):
   doc_data.append(model.infer_vector(df_test['text_cleaned'].tolist()[i].split(' ')))
x_test_doc=pd.DataFrame(doc_data)
x_test_doc.head()

from sklearn.model_selection import train_test_split
x_train_doc, x_val_doc, y_train_doc, y_val_doc=train_test_split(doc_df,df_train['target'],  test_size=0.1, random_state=123)

'''x_train_doc = tf.keras.preprocessing.sequence.pad_sequences(x_train_doc)
y_train_doc=np.array(y_train_doc.tolist())
y_val_doc=np.array(y_val_doc.tolist())'''

"""# Modélisation

## Logistic Regressor
"""

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore')

#Model:
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
l_model=LogisticRegression()
#l_model = MultiOutputClassifier(l_model)
def clean_dataset(df, y):
    df['y']=y.values
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df=df[indices_to_keep].astype(np.float64)
    y=df['y']
    df.drop(columns=['y'], inplace=True)
    return df, y

def clean_test_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df=df[indices_to_keep].astype(np.float64)
    return df

def logistic_regressor(x_train, y_train):
    n_jobs=[-1]

    hyperparameters = {"C": [0.5, 1], 
        "l1_ratio": [0.5, 1],
        "max_iter": [100,300, 1000], 
                          'n_jobs': n_jobs,
              
                          }
    print(GridSearchCV(l_model, param_grid={}, scoring='accuracy').get_params())
    x_train, y_train=clean_dataset(x_train, y_train)
    grid = GridSearchCV(l_model, hyperparameters, cv=KFold(4,shuffle=True, random_state=123), scoring='accuracy', verbose=2)
    grid.fit(x_train, y_train)
    return grid

"""### Tfidf"""

from sklearn.metrics import accuracy_score
model_tf=logistic_regressor(x_train_tf, y_train_tf)

x_val_tf, y_val_tf=clean_dataset(x_val_tf, y_val_tf)
score=model_tf.score(x_val_tf, y_val_tf)
print("validation_score: ", score)

x_test_tf=tif.transform(df_test['text_cleaned']).toarray()
x_test_tf=pd.DataFrame(x_test_tf)
x_test_tf[['word_count', 'char_count', 'avg_word_len', 'punc_per']]=df_test[['word_count', 'char_count', 'avg_word_len', 'punc_per']]
x_test_tf=clean_test_dataset(x_test_tf)
pred=model_tf.predict(x_test_tf)
print("accuracy: ", accuracy_score(model_tf.predict(x_val_tf), y_val_tf))

pred=[int(i) for i in pred]

df_test['target']=pred
df_submit['target']=pred
df_submit.head()

df_test[['text_cleaned', "target"]]

df_test.target.describe()

df_submit.to_csv('sample_submission.csv', index=False)