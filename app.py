from flask import Flask, redirect, url_for,render_template,send_from_directory
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from generator import PDF

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
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
    app.run(host='0.0.0.0',debug=True)