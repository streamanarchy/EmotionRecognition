import voiceInput
import speechRecognition
import voiceFeature
import voiceEmotion
import wordEmotion
import threading

class controlPanel():
    def __init__(self):
        self.voiceInput = voiceInput.VoiceInput()
        self.speechRecognition = speechRecognition.SpeechRecognition()
        self.voiceFeatures = voiceFeature.VoiceFeature()
        self.voiceEmotion = voiceEmotion.VoiceEmotion()
        self.wordEmotion = wordEmotion.WordEmotion()
        self.voiceemotioncoefficient = []
        self.speechemotioncoefficient = []

        #self.emotionCoefficient = emotionCoefficient()
        #self.emotionDatabase = emotionDatabase()
        #self.wordDatabase = wordDatabase()

    def recognize(self):
        voicethread = threading.Thread(name='voicethread',target=self.voice)
        speechthread = threading.Thread(name='speechthread',target=self.speech)
        voicethread.start()
        speechthread.start()
        voicethread.join()
        speechthread.join(8)
        return self.voiceemotioncoefficient,self.avgarray,self.speechemotioncoefficient

    def voice(self):
        x = self.voiceInput.voiceInput(6)
        feature = self.voiceFeatures.FeatureExtraction(x,x.shape[0],x.shape[0])
        self.voiceemotioncoefficient,self.avgarray = self.voiceEmotion.voiceEmotionRecognition(feature)

    def speech(self):
        self.speechemotioncoefficient = self.wordEmotion.wordEmotionRecognition()


    def display(self):
        #TODO call main interface here
        while 1:
            print("Select your options:\n1.Train and Test \n2.Input and Analyze\n3.Voice analysis\nInput:")
            optionInput = int(raw_input())
            if optionInput == 1:
                self.voiceEmotion.training()
            if optionInput == 2:
                x = self.voiceInput.voiceInput(4)
                print x
                feature = self.voiceFeatures.FeatureExtraction(x,x.shape[0],x.shape[0])
                emotion = self.voiceEmotion.voiceEmotionRecognition(feature)
                print emotion
            if optionInput == 3:
                self.voiceEmotion.testing()
            if optionInput == 4:
                exit()

if __name__ == "__main__":
    controller = controlPanel()
    controller.display()