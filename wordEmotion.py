import speech_recognition
import pickle
import pyaudio
from sklearn.feature_extraction.text import TfidfVectorizer
class WordEmotion():
    def __init__(self):
        self.speechrecognizer = speech_recognition.Recognizer()
        self.learnfile = open('text.learn','rb')
        self.classifier = pickle.load(self.learnfile)
        self.learnfile.close()
        self.learnfile = open('text.vector','rb')
        self.vectorize = pickle.load(self.learnfile)
        self.learnfile.close()

    def wordEmotionRecognition(self):
        with speech_recognition.Microphone() as source:
            #TODO add word emotion configuration  #Main control for this class
            audio = self.speechrecognizer.listen(source)
            try:
                speechText = self.speechrecognizer.recognize_google(audio)
                print speechText
                speechTextList = [speechText]
                speechTextTrain = self.vectorize.transform(speechTextList)
                return self.classifier.predict(speechTextTrain)
            except speech_recognition.UnknownValueError:
                return -1
            except speech_recognition.RequestError:
                return -2

    def wordRelationalSignificance(self):
        #TODO use twitter database for this
        #TODO write a script to push data in database
        return

    def wordVAD(self):
        #TODO use database to get VAD values

        return

    def VADRegressionToEmotionClassification(self):
        #TODO figure out relationship between VAD and classes
        return


if __name__=="__main__":
    wordemotion = WordEmotion()
    wordemotion.wordEmotionRecognition()