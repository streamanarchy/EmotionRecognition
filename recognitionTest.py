import speech_recognition as srecog
import pyaudio
def recog():
	"""rec = srecog.Recognizer()
	with srecog.WavFile() as source:
		print "Now listening:"
		voice = rec.listen(source)
	print voice
	print rec.recognize_google(voice)"""
	while 1:
		rec = srecog.Recognizer()
		with srecog.Microphone() as source:
			print "Now listening:"
			voice = rec.listen(source)
		print rec.recognize_google(voice)
				
recog()

