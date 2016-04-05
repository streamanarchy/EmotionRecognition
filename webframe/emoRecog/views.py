from django.shortcuts import render_to_response
from django.http import HttpResponse
from django.template import loader
from django.template import RequestContext
import json
import emotionRecognition

recognizer = emotionRecognition.controlPanel()
def index(request):
	template = loader.get_template('templates/index.html')
	emotion = emotionplot()
	context = {'emotion': emotion, }

	return HttpResponse(template.render(context,request))

def replot(request):
	emotion = emotionplot()
	context = {'emotion': emotion,}
	return HttpResponse(json.dumps(context),content_type='application/json')

def emotionplot():
	raw_emotion = recognizer.recognize()
	return raw_emotion


