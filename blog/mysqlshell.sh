#!/bin/bash
mysql -u root -p -e "CREATE DATABASE blog;"
mysql -u root -p -e "CREATE USER 'bloguser'@'localhost' IDENTIFIED BY 'blogpassword';"
mysql -u root -p -e "GRANT ALL PRIVILEGES ON blog.* TO 'bloguser'@'localhost';"
