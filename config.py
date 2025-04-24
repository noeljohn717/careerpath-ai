import os
basedir = os.path.abspath(os.path.dirname(__file__))

SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'database', 'careerpath.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = 'your-secret-key'
