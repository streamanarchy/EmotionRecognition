from django.shortcuts import render_to_response
from django.http import HttpResponse
from django.template import loader
from django.template import RequestContext
import json
import emotionRecognition

recognizer = emotionRecognition.controlPanel()
def index(request):
	template = loader.get_template('templates/index.html')
	emotion,avgemotion,mode = emotionplot()
	print "Avg emotion",avgemotion
	context = {'emotion': emotion, 'avgarray':avgemotion.tolist(),'mode':mode}

	return HttpResponse(template.render(context,request))

def replot(request):
	emotion,avgemotion,mode = emotionplot()
	context = {'emotion': emotion, 'avgarray':avgemotion.tolist(),'mode':mode}
	return HttpResponse(json.dumps(context),content_type='application/json')

def emotionplot():
	raw_emotion,avgemotion ,mode= recognizer.recognize()
	return raw_emotion,avgemotion,mode


