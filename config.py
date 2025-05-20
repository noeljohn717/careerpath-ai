import os
basedir = os.path.abspath(os.path.dirname(__file__))

# app/__init__.py or config.py
from dotenv import load_dotenv
import os

basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)
  
# General Config
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///careerpath.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Mail Config
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USE_TLS = True
MAIL_USERNAME = os.getenv('MAIL_USERNAME', 'your-email@gmail.com')
MAIL_PASSWORD = os.getenv('MAIL_PASSWORD', 'your-email-password')

SESSION_COOKIE_SECURE = True
REMEMBER_COOKIE_SECURE = True

