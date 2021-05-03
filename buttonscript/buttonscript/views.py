from django.shortcuts import render
import requests
from subprocess import run, PIPE
import sys


def button(request):
    return render(request, 'index.html')


def output(request):
    data = requests.get('https://www.google.co.in/')
    print(data.text)
    data = data.text
    return render(request, 'index.html', {'data': data})


def external_text(request):
    inp = request.POST.get('param')
    out = run([sys.executable, 'G://NIT//1.PROJECT WORK NIT//SentimentAnalysisWebApp//app.py',
              inp], shell=False, stdout=PIPE)
    print(out)
    return render(request, 'index.html', {'data1': out.stdout.decode("utf-8")})


def external_audio(request):
    inp = request.POST.get('avatar')
    out = run([sys.executable, 'G://NIT//1.PROJECT WORK NIT//SentimentAnalysisWebApp//app1.py',
              inp], shell=False, stdout=PIPE)
    print(out)
    return render(request, 'index.html', {'data2': out.stdout.decode("utf-8")})
