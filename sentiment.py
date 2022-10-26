import string,re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt


text = open("datas/read.txt", encoding="utf-8").read()
text = text.lower()
print(type(text)) # <class 'str'>

cleaned_text = text.translate(str.maketrans("","",string.punctuation))
#print(cleaned_text)
tokenized_words = word_tokenize(cleaned_text) # cümleyi kelimelere ayırma

def stopwordsThrow(tokenized_words):
    finalWords = []
    for word  in tokenized_words:
        if word not in stopwords.words("english"):
            finalWords.append(word)
    return finalWords

def emotionListCreate():
    emotionList = []
    with open("datas/emotions.txt", "r") as file:
        for line in file:
            line = line.replace("\n","").replace(",","").replace("'","").strip() # fazladan boslugu, whitespace karakterleri kaldırma
            word,emotion = line.split(":")
            finalWords = stopwordsThrow(tokenized_words)
            if word in finalWords:
                emotionList.append(emotion)
    return emotionList
emotionList = emotionListCreate()
emotionDict = Counter(emotionList)
print(emotionDict)

def visualization(dictionary):
    fig, axes = plt.subplots()
    axes.bar(dictionary.keys(), dictionary.values(), color = "brown")
    fig.autofmt_xdate()
    plt.title("Sentiment Analysis")
    plt.xlabel("Emotion")
    plt.ylabel("Percent")
    #plt.savefig("graphEmotion.png")
    plt.show()
visualization(emotionDict)
