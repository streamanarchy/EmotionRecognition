from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import emotionRecognition

recognizer = emotionRecognition.controlPanel()
def index(request):
	template = loader.get_template('templates/index.html')
	emotion = emotionplot()
	print emotion
	context = {'emotion': emotion, }

	return HttpResponse(template.render(context,request))

def emotionplot():
	raw_emotion = recognizer.recognize()
	return raw_emotion


