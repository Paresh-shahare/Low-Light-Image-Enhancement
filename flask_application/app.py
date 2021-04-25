from flask import Flask

UPLOAD_FOLDER = 'C:/Users/SPARK/Downloads/flask_files/static/uploads'

app = Flask(__name__,template_folder='C:/Users/SPARK/Downloads/flask_files/templates',static_folder='C:/Users/SPARK/Downloads/flask_files/static')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
