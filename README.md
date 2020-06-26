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
# test.py
def application(env, start_response):
    start_response('200 OK', [('Content-Type','text/html')])
    return [b"Hello World"] # python3
    #return ["Hello World"] # python2
``` 
run uwsgi
```shell
$ uwsgi --http :8000 --wsgi-file test.py
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
if receive Invalid HTTP_HOST header: '192.168.50.240:8000'. You may need to add '192.168.50.240' to ALLOWED_HOSTS.
edit settings.py
```
ALLOWED_HOSTS = ['*']
```
if use uwsgi aslo see successfully and follow next step
### put uwsgi_params in your django project
[uwsgi_params_example](https://github.com/samdjk118/DjangoWeb/blob/master/uwsgi_params)
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
$ sudo ln -s /etc/nginx/sites-available/mysite_nginx.conf /etc/nginx/sites-enabled/
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
check your broswer can connect to http://127.0.0.1:8000<br>
else you can check log from /var/log/nginx/error.log
### Configuring uWSGI to run with a .ini file
```
# mysite_uwsgi.ini file
[uwsgi]

# Django-related settings
# the base directory (full path)
chdir           = /path/to/your/project
# Django's wsgi file
module          = project.wsgi
# the virtualenv (full path)
home            = /path/to/virtualenv

# process-related settings
# master
master          = true
# maximum number of worker processes
processes       = 10
# the socket (use the full path to be safe
socket          = /path/to/your/project/mysite.sock
# ... with appropriate permissions - may be needed
chmod-socket    = 666
# clear environment on exit
vacuum          = true
```
And run uwsgi using this file:
```
$ uwsgi --ini mysite_uwsgi.ini
```
also check your htttp://127.0.0.1:8000 can connect 
### Install uWSGI system-wide
Deactivate your virtualenv:
```
$ deactivate
```
and install uWSGI system-wide:
```
$ sudo pip install uwsgi

# Or install LTS (long term support).
$ pip install https://projects.unbit.it/downloads/uwsgi-lts.tar.gz
```
Check again that you can still run uWSGI just like you did before:
```
$ uwsgi --ini mysite_uwsgi.ini
```
## 3.Emperor mode
uWSGI can run in ‘emperor’ mode. In this mode it keeps an eye on a directory of uWSGI config files, and will spawn instances (‘vassals’) for each one it finds.<br>
Whenever a config file is amended, the emperor will automatically restart the vassal.
```
create a directory for the vassals
$ sudo mkdir -p /etc/uwsgi/vassals
symlink from the default config directory to your config file
$ sudo ln -s /path/to/your/mysite/mysite_uwsgi.ini /etc/uwsgi/vassals/
give your user Permission to use www-data
$ usermod www-data -aG your-user
logout your account to make access usermod and login your account again
run the emperor
$ /path/to/VENV/bin/uwsgi --emperor /etc/uwsgi/vassals --uid user-name --gid user-name
```
check site.And it should be running.
## Set startup uwsgi when system boots
debain/ubuntu:<br>
create a systemd system unit 
/etc/systemd/system/uwsgi.service
```
Description=Django server by uWSGI
After=syslog.target

[Service]
ExecStart=/path/to/VENV/bin/uwsgi --ini /etc/uwsgi/vassals/mysite.ini 
Restart=always
KillSignal=SIGQUIT
Type=notify
StandardError=syslog
NotifyAccess=all

[Install]
WantedBy=multi-user.target
```
Here it is set to restart automatically (when there is an error), and lead stderr to syslog.<br> Then, it is necessary to start the uwsgi.service service:
```
$ sudo systemctl enable uwsgi.service
$ sudo systemctl status uwsgi.service
```
reboot your system to make sure it work!

--- 
# reference document<br>
[uwsgi-docs](https://uwsgi-docs.readthedocs.io/en/latest/tutorials/Django_and_nginx.html)
