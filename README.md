# DjangoWeb
## 1.set up envirment
notice : when you can not install **python3-pip**<br>
``` shell
$sudo add-apt-repository universe
```
if not got any error message and then do<br>
```shell
$ apt-get update 
$ apt-get python3-pip
```
### set up virtualenv
install and use virtualenv
```shell
$ virtualenv (your virtualenv name)
$ source (your virtualenv name)/bin/activate
```
### Django
install django and setup project
```shell
$ pip install Django
$ django-admin.py startproject mysite
$ cd mysite
```
### Install uwsgi
install uwsgi
```shell
$ pip install uwsgi
```
### Basic test
create a file to test uwsgi<br>
test.py
```python
def application(env, response):
    start_response('200 OK',[('Content-Type':'text/html')])
    return [b"Hello World"]
``` 
run uwsgi
```shell
$ uwsgi --http :8000 --wsgi-file test.pt
``` 
in linux test
 ```shell
$ curl http://127.0.0.1:8000
```
### Test Django Project
```shell
$ python3 manage.py runserver 0.0.0.0:8000
```
