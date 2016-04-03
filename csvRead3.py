import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
train_df_raw = pd.read_csv('/home/project/Documents/Project/textdata/twitter_sentiment_data.csv',header=None, names=['label','tweet'],encoding='ISO-8859-1')
test_df_raw = pd.read_csv('/home/project/Documents/Project/test.csv',header=None, names=['label','tweet'],encoding='ISO-8859-1')
train_df_raw =  train_df_raw[train_df_raw['tweet'].notnull()]
test_df_raw =  test_df_raw[test_df_raw['tweet'].notnull()]
#test_df_raw =  test_df_raw[test_df_raw['label']]

y_train = [x if x==0 else 1 if x==2 else 2 for x in train_df_raw['label'].tolist()]
y_test = [x if x==0 else 1 if x==2 else 2 for x in test_df_raw['label'].tolist()]
X_train = train_df_raw['tweet'].tolist()
X_test = test_df_raw['tweet'].tolist()
print dir(test_df_raw['tweet'])
print X_test[1]
print dir(X_test)
print('At vectorizer')
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
print('At vectorizer for test data')
X_test = vectorizer.transform(X_test)

print('at Classifier')
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print 'Accuracy:', accuracy_score(y_test, predictions)
textclassifier = open('text.learn', 'wb')
pickle.dump(classifier,textclassifier)
textclassifier.close()
textclassifier = open('text.vector','wb')
pickle.dump(vectorizer,textclassifier)
textclassifier.close()
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)