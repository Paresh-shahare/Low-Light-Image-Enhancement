from flask import Flask, render_template, request, redirect, flash, url_for
import main
import urllib.request
from app import app
from werkzeug.utils import secure_filename
from main import getPrediction
import os
from matplotlib import pyplot as plt

@app.route('/')
def index():
    return render_template('webUI.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            label=getPrediction(filename)
            
            plt.figure()
            plt.imshow(label[1])
            plt.axis('off')
            plt.savefig('C:/Users/SPARK/Downloads/flask_files/static/rock/'+filename, dpi=72, bbox_inches='tight')
            #plt.show()
            
            plt.figure()
            plt.imshow(label[0])
            plt.axis('off')
            plt.savefig('C:/Users/SPARK/Downloads/flask_files/static/download/'+filename, dpi=72, bbox_inches='tight')
            #plt.show()    
            return render_template('webUI.html',filename=filename)

@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static',filename='download/' + filename), code=301)
@app.route('/original/<filename>')
def display_o_image(filename):
    print('display_o_image filename: ' + filename)
    return redirect(url_for('static',filename='rock/' + filename), code=302)



if __name__ == "__main__":
    app.run()
