import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression



df = pd.read_csv("datas/training.1600000.processed.noemoticon.csv", encoding="latin-1", nrows=100)
df.columns = ["target","id","date","flag","user","text"]
print(df.target.unique())
df = df.replace(4,1)

# 0: NEGATIVE, 1: POSITIVE
#print(df.text)
print(df.target.unique())
#sns.countplot(df.target)
#plt.show()

data = []
X = df.text.tolist()
y = df.target.tolist()

for X,y in zip(X,y):
    data.append((X,y))

#for sentence, target in data:
 #   print(sentence)

# DATA PREPROCESSING
def cleanedSentence(data):
    final_words = []
    for i in range(len(data)):
        word = data[i][0]
        word = word.lower()
        cleaned_text = word.translate(str.maketrans("","",string.punctuation))
        final_words.append((cleaned_text,data[i][1]))
    return final_words

def wordSplit(data):
    total_words = []
    final_words = cleanedSentence(data)
    tokenizer = TweetTokenizer()
    for sentence,target in final_words:
        tokenize_text = tokenizer.tokenize(sentence)
        newTokenizeText = []
        [newTokenizeText.append(tokenize_text[i]) for i in range(len(tokenize_text)) if tokenize_text[i] not in stopwords.words("english")]
        total_words.append((newTokenizeText,target))
    return total_words

#total_words = wordSplit(data)
#print(total_words)
def lemmatizerWords(data):
    lemmaWords = []
    lemmatizer = WordNetLemmatizer()
    total_words = wordSplit(data)
    for words, target in total_words:
        newLemmaWords = []
        for word in words:
            lemmatize_word = lemmatizer.lemmatize(word)
            newLemmaWords.append(lemmatize_word)
            if word != lemmatize_word:
                print(f"{word}:{lemmatize_word}")
        lemmaWords.append((newLemmaWords,target))
    return lemmaWords
#print(lemmatizerWords(data)[0][0])

def posTag(data):
    posTagList = []
    wordsList = []
    lemmaWords = lemmatizerWords(data)
    #[wordsList.append(words) for words, target in lemmaWords]
    for wordsList, target in lemmaWords:
        #print(pos_tag(wordsList))
        newWordsList = []
        for i in range(len(pos_tag(wordsList))):
            word = pos_tag(wordsList)[i][0]
            tag = pos_tag(wordsList)[i][1]
            if tag.startswith("NN") or tag.startswith("VB"): # noun or verb
               #print(f"{word}:{tag}")
               newWordsList.append(word)
        posTagList.append((newWordsList,target))
    return posTagList
#posTagList = posTag(data)
#print(posTagList)
#print(len(posTagList))

# DATA VISUALIZATION
def negativePositiveWords(data):
    positiveWords, negativeWords = [],[]
    posTagList = posTag(data)
    for i in range(len(posTagList)):
        if posTagList[i][1] == 0:
            negativeWords.append(posTagList[i][0])
        else:
            positiveWords.append(posTagList[i][0])
    return negativeWords,positiveWords

def wordCloud(dataList, imagePath, color = "black"):
    plt.figure(figsize=(13,10), dpi=80)
    print(f"dataList: {dataList}")
    cloud = WordCloud(background_color=color,
                      width=2500,
                      height=2000).generate(''.join(dataList))
    plt.imshow(cloud,interpolation='bilinear')
    plt.title("Twitter Word Cloud")
    plt.axis("off")
    plt.savefig(imagePath)
    plt.show()

#negativeWords,positiveWords = negativePositiveWords(data)
#wordCloud(negativeWords, "img/twitterNegativeWords.png")
#wordCloud(positiveWords, "img/twitterPositiveWords.png", color = "white")
posTagList = posTag(data)

# MODEL CREATE
def modelCreate(posTagList):
    newdf = pd.DataFrame(posTagList, columns=["text","target"])
    newdf.text = newdf.text.apply(lambda X:" ".join([w for w in X]))
    X = newdf.text.values
    y = newdf.target.values
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=42)
    return X_train,X_test,y_train,y_test

def vectorizer(posTagList):
    X_train,X_test,y_train,y_test = modelCreate(posTagList)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    print(len(vectorizer.get_feature_names()))
    return X_train,X_test,y_train,y_test

def modelEvaluate(model, image = False):
    X_train, X_test, y_train, y_test = vectorizer(posTagList)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    #print(f"y_pred: {y_pred}\ty_test: {y_test}")
    print(classification_report(y_test, y_pred))
    if image == True:
        plt.figure(figsize=(12,6))
        sns.heatmap(confusion_matrix(y_test, y_pred))
        plt.show()


lr = LogisticRegression()
modelEvaluate(lr)

'''
feature_names = vectorizer.get_feature_names()
dense = tfidf_matrix.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names, index=characters)
s = pd.Series(df.loc['Adam'])
s[s > 0].sort_values(ascending=False)[:10]
'''
