# DjangoWeb Deploy on nginx web server
This project is talk about how to setup django on nginx and with uwsgi
## 1.Setting up uWSGI
### set up envirment
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
create test.py 
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
if you can see successfully!,run it use uwsgi
```shell
$ uwsgi --http :8000 --module mysite.wsgi
```
if use uwsgi aslo see successfully and follow next step
## 2.Basic nginx
### Install nginx
```shell
$ sudo apt-get install nginx
$ sudo /etc/init.d/nginx start
$ sudo systemctl enable nginx //the service will start auto 
```
### Configure nginx for your site
Create a file called mysite_nginx.conf in the /etc/nginx/sites-availabe/ directory<br>
and paste this in it
```shell
# mysite_nginx.conf

# the upstream component nginx needs to connect to
upstream django {
    server unix:///path/to/your/mysite/mysite.sock; # for a file socket
    # server 127.0.0.1:8001; # for a web port socket (we'll use this first)
}

# configuration of the server
server {
    # the port your site will be served on
    listen      8000;
    # the domain name it will serve for
    server_name example.com; # substitute your machine's IP address or FQDN
    charset     utf-8;

    # max upload size
    client_max_body_size 75M;   # adjust to taste

    # Django media
    location /media  {
        alias /path/to/your/mysite/media;  # your Django project's media files - amend as required
    }

    location /static {
        alias /path/to/your/mysite/static; # your Django project's static files - amend as required
    }

    # Finally, send all non-media requests to the Django server.
    location / {
        uwsgi_pass  django;
        include     /path/to/your/mysite/uwsgi_params; # the uwsgi_params file you installed
    }
}
```
Symlink to this file from /etc/nginx/sites-enabled so nginx can see it:
```shell
$ ln -s mysite_nginx.conf /etc/nginx/sites-enabled/
```
### Deploying static files
Before running nginx,you have to collect all Django static files in the static folder.<br>
first of all we have to edit mysite/settings.py padding:
```shell
STATIC_ROOT = os.path.join(BASE_DIR, "static/")
```
and then run 
```shell
$ python manage.py collectstatic
```
### Basic nginx test 
Restart nginx
```shell
$ sudo /etc/init.d/nginx restart
```
and then go browser to check nginx<br>http://127.0.0.1<br>
### Running the Django application with uwsgi and nginx
```
$ uwsgi --socket mysite.sock --module mysite.wsgi --chmod-socket=664
```
--- 
# reference document<br>
[uwsgi-docs](https://uwsgi-docs.readthedocs.io/en/latest/tutorials/Django_and_nginx.html)
