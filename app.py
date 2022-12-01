from flask import Flask, redirect, url_for,render_template,send_from_directory
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from generator import PDF
import pandas as pd
import os

UPLOAD_FOLDER = 'uploads'
TMP_FOLDER = 'tmp'
ALLOWED_EXTENSIONS = {'csv'}

def save_data(name):
    print(name)
    df = pd.read_csv(os.path.join(app.config['TMP_FOLDER'], name),names=['date','open','high','low','close','volume'],parse_dates=['date'],header=0)
    new_name = name.split(".")[0]+ '.parquet'
    df.to_parquet('data/'+new_name)
    flash("Loaded")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TMP_FOLDER'] = TMP_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            if 'data' not in request.files:
                return render_template('index.html')
        else:
            file = request.files['file']
            data = request.files['data']
            if file.filename == '':
                if data.filename == '':
                    render_template('index.html')
                else:
                    if data and allowed_file(data.filename):
                        data_filename = secure_filename(data.filename) # type: ignore
                        if not os.path.exists(app.config['TMP_FOLDER']):
                            os.makedirs(app.config['TMP_FOLDER'])
                            data.save(os.path.join(app.config['TMP_FOLDER'], data_filename))
                            save_data(data_filename)

                        else:
                            data.save(os.path.join(app.config['TMP_FOLDER'], data_filename))
                            save_data(data_filename)

            if file and allowed_file(file.filename):
                if data and allowed_file(data.filename):
                    data_filename = secure_filename(data.filename) # type: ignore
                    if not os.path.exists(app.config['TMP_FOLDER']):
                        os.makedirs(app.config['TMP_FOLDER'])
                        data.save(os.path.join(app.config['TMP_FOLDER'], data_filename))
                        save_data(data_filename)
                    else:
                        data.save(os.path.join(app.config['TMP_FOLDER'], data_filename))
                        save_data(data_filename)

                filename = secure_filename(file.filename)  # type: ignore
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('download_file', name=filename))
    return render_template('index.html')
@app.route('/uploads/<name>')
def download_file(name):
    new_path = None
    pdf = PDF(name)
    try:
        new_path = pdf.generate_pdf() + '.pdf'
    except:
        pass
    new_path = name[:-4] +".pdf"
    return send_from_directory(app.config["UPLOAD_FOLDER"], new_path)  # type: ignore
if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0')
