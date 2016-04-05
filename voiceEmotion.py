import neuralNetwork
import trainingData
import pickle
class VoiceEmotion():
    def __init__(self, weightFile = None):
        self.neuralnetwork = {}
        self.neuralnetwork['W'] = neuralNetwork.NeuralNetwork()
        self.neuralnetwork['L'] = neuralNetwork.NeuralNetwork()
        self.neuralnetwork['E'] = neuralNetwork.NeuralNetwork()
        self.neuralnetwork['A'] = neuralNetwork.NeuralNetwork()
        self.neuralnetwork['F'] = neuralNetwork.NeuralNetwork()
        self.neuralnetwork['T'] = neuralNetwork.NeuralNetwork()
        self.neuralnetwork['N'] = neuralNetwork.NeuralNetwork()
        print self.neuralnetwork
        self.emotionlist = []
        self.filebuffer = open("emotion.normal","wb")

        #self.trainer = neuralNetwork.Trainer(self.neuralnetwork)
        self.emotionconversion = {'W':['Anger'],'L':['Boredom'],'E':['Disgust'],'A':['Anxiety'],'F':['Happiness'],'T':['Sadness'],'N':['Neutral']}
        self.trainednetwork = trainingData.Training(self.neuralnetwork)
        
    def training(self):
        features_train = self.trainednetwork.feature_normalisation()
        print "Starting training"

        self.neuralnetwork = self.trainednetwork.emo_db_complete_processing(features_train)
        self.trainednetwork = trainingData.Training(self.neuralnetwork)
        self.trainednetwork.features_train = features_train
        self.neuralnetwork = self.trainednetwork.emo_db_complete_processing(features_train)

    def voiceEmotionRecognition(self,x):
        x = x.T
        emotion = {}
        for netemo,net in self.neuralnetwork.items():
            y = net.forward(x)
            emotion[self.emotionconversion[netemo][0]] = y[0][0]

        return emotion

    def testing(self):
        #self.trainednetwork.feature_normalisation()
        self.neuralnetwork = self.trainednetwork.test_emo_db_complete()

if __name__ == "__main__":
    print "Setup a input configuration for voice recognition"


