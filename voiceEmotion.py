import neuralNetwork
import trainingData
class VoiceEmotion():
    def __init__(self, weightFile = None):
        self.neuralnetwork = neuralNetwork.NeuralNetwork(weightFile)
        self.trainer = neuralNetwork.Trainer(self.neuralnetwork)
        self.emotionconversion = {'W':['Anger'],'L':['Boredom'],'E':['Disgust'],'A':['Anxiety'],'F':['Happiness'],'T':['Sadness'],'N':['Neutral']}
        self.trainednetwork = trainingData.Training(self.neuralnetwork)
        
    def Training(self):
        self.trainingdata.feature_normalisation()
        for key,value in self.emotionconversion.items():
            self.neuralnetwork = self.trainednetwork.emo_db_complete_processing(key)

    def voiceEmotionRecognition(self,x):
        y = self.neuralnetwork.forward(x)
        return y

if __name__ == "__main__":
    print "Setup a input configuration for voice recognition"


