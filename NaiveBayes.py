import pandas as pd
import urllib.request as request
import json
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB   #import MultinomialNB model
from sklearn.metrics import accuracy_score, precision_score, recall_score   #evaluate module
from sklearn.metrics import accuracy_score

df = pd.read_csv("sinhala-hate-speech-dataset.csv",encoding='utf8')
X_train, X_test, y_train, y_test = train_test_split(df.comment,df.label,test_size=0.3)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train,y_train)


predictions = model.predict(X_test)
print('Accuracy: ', accuracy_score(y_test.astype(str),predictions.astype(str))*100)
#print(precision_score(y_test,predictions,pos_label='nonhate')*100)
#print('Recall: ', recall_score(y_test.astype(str),predictions.astype(str),pos_label='nonhate')*100)

def classify_hate(arr):
  return model.predict(vectorizer.transform(arr))

with request.urlopen('YoutubeAPI') as response:
    source = response.read()
    data = json.loads(source)
    count = 0
    totalComments = 0
    nba_teams = [team for team in data['items']]

    for x in nba_teams:
        abc = (x['snippet']['topLevelComment']['snippet']['textDisplay'])
        xyz = (x['snippet']['topLevelComment']['snippet']['likeCount'])

        regexp = re.compile(r'[a-zA-Z]+')
        if not regexp.search(abc):  # Check whether the comment contains alphabet
            value = 1 + xyz
            totalComments += value

            def remove_punctuation(text):
                translator = str.maketrans('', '', string.punctuation)  # Pre-processing (Remove Punctuation marks)
                return text.translate(translator)

            x = remove_punctuation(abc)

            def remove_stopwords(text):
                sw_array = []
                with open('sw.txt',encoding="UTF-8") as stopwords_file:  # Remove Stopwords
                    for line in stopwords_file:
                        sw_array.append(line.rstrip('\n'))

                    querywords = text.split()
                    resultwords = [word for word in querywords if word not in sw_array]
                    result = ' '.join(resultwords)
                return result

            y = remove_stopwords(x)

            def remove_numbers(text):
                result = re.sub(r'\d+', '', text)  # Pre processing (Remove Numbers)
                return result

            z = remove_numbers(y)
            messages = [z]
            for index_instance, instance in enumerate(classify_hate(messages)):
                print(messages[index_instance], ' - ', instance)
                if instance == "hate":
                    val = 1 + xyz
                    count += val

percentage = (count / totalComments) * 100
print('Hate comment percentage: ' + str(percentage) + '%')
