import voiceInput
import speechRecognition
import voiceFeature
import voiceEmotion

class controlPanel():
    def __init__(self):
        self.voiceInput = voiceInput.VoiceInput()
        self.speechRecognition = speechRecognition.SpeechRecognition()
        self.voiceFeatures = voiceFeature.VoiceFeature()
        self.voiceEmotion = voiceEmotion.VoiceEmotion()
        self.wordEmotion = wordEmotion()
        self.emotionCoefficient = emotionCoefficient()
        self.emotionDatabase = emotionDatabase()
        self.wordDatabase = wordDatabase()

    def display(self):
        #TODO call main interface here
        while 1:
            print("Select your options:\n1.Train and Test \n2.Input and Analyze\n3.Voice analysis\nInput:")
            optionInput = int(raw_input())
            if optionInput == 1:
                train()
            if optionInput == 2:
              main.control()
            if optionInput == 3:
                recordAudioSegments("",4)

if __name__ == "__main__":
    controller = controlPanel()
    controller.display()