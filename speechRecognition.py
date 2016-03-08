import speech_recognition

class speechRecognition(object):

    def __init__(self):
        self.recognizer = speech_recognition.Recognizer()
        self.microphone = speech_recognition.Microphone()

    def noiseAdjustment(self):
        return self.recognizer.adjust_for_ambient_noise(self.microphone)

    def backgroundListen(self):
        with self.recognizer as source:
            self.recognizer.listen_in_background(source, self.recognition)

    def recognition(self, audio):
        try:
            speechText = self.recognizer.recognize_google(audio,key=None,language="en-US",show_all=True)
            #TODO redirect speech Text
            print speechText
        except speech_recognition.UnknownValueError:
            print "Can't understand"
        except speech_recognition.RequestError:
            print "Can't Connect"

if __name__ == "__main__":
    speechRecognition = speechRecognition()
    speechRecognition.noiseAdjustment()
    speechRecognition.backgroundListen()