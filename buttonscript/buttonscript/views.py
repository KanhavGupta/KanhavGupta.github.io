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
    out = run([sys.executable, 'KanhavGupta.github.io//buttonscript//app.py',
              inp], shell=False, stdout=PIPE)
    print(out)
    return render(request, 'https://kanhavgupta.github.io/buttonscript/templates/', {'data1': out.stdout.decode("utf-8")})


def external_audio(request):
    inp = request.POST.get('avatar')
    out = run([sys.executable, 'KanhavGupta.github.io//buttonscript//app1.py',
              inp], shell=False, stdout=PIPE)
    print(out)
    return render(request, 'https://kanhavgupta.github.io/buttonscript/templates/', {'data2': out.stdout.decode("utf-8")})
