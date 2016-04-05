from django.conf.urls import url
from . import views

urlpatterns = [url(r'^emoRecog',views.index,name='index'),url(r'^replot',views.replot,name ='replot')]
