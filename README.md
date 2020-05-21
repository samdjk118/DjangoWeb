# DjangoWeb
## notice : when you can not install `python3-pip`
 $sudo add-apt-repository universe
#and then do
  4 $ apt-get update
  5 $ apt-get python3-pip
  6 #install and use virtualenv
  7 $ virtualenv (your virtualenv name)
  8 $ source (your virtualenv name)/bin/activate
  9 #install django and setup project
 10 $ pip install Django
 11 $ django-admin.py startproject mysite
 12 $ cd mysite
 13 #install uwsgi
 14 $ pip install uwsgi
 15 # create a file to test uwsgi
 16 # test.py
 17 def application(env, response):
 18     start_response('200 OK',[('Content-Type':'text/html')])
 19     return [b"Hello World"]
 20 # run uwsgi
 21 $ uwsgi --http :8000 --wsgi-file test.pt
 22 # in linux test
 23 curl http://127.0.0.1:8000
