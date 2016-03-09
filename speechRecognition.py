import speech_recognition

class SpeechRecognition(object):

    def __init__(self):
        self.recognizer = speech_recognition.Recognizer()
        self.microphone = speech_recognition.Microphone()

    def noiseAdjustment(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def backgroundListen(self):
        self.recognizer.listen_in_background(self.microphone, self.recognition)

    def recognition(self,recognizer, audio):
        try:
            speechText = recognizer.recognize_google(audio)
            #TODO redirect speech Text
            print speechText
        except speech_recognition.UnknownValueError:
            print "Can't understand"
        except speech_recognition.RequestError:
            print "Can't Connect"

if __name__ == "__main__":
    speechRecognitionEnglish = SpeechRecognition()
    speechRecognitionEnglish.noiseAdjustment()
    speechRecognitionEnglish.backgroundListen()
    while 1:
        pass