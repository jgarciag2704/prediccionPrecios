from django.shortcuts import render,redirect
from .models import historicoPrecios
from django.http import HttpResponse,JsonResponse
# Create your views here.

def index(request):
    title="Prediccion de verduras"
    return render(request,"index.html",{
        'title':title
    })
    
def about(request):
    username='Jose Manuel'
    return render(request,"about.html",{
        'username':username
    })