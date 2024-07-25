from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap
import os
import psycopg2 # type: ignore
import torch
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from torchvision.models import resnet50
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required

app = Flask(_name_)
app.secret_key = 'scrypt:32768:8:1$roCjGTFYRckzsQGi$aa2d675ed3fabcc531d2199debb387144d4f457c01c34f8d3f95d8c1aa81b63e3673c4f60856e151ffff11f49416c57ba2a2feaae8041d2e2e191fe1a32c4c61'  # Replace with a secure secret key
Bootstrap(app)

# PostgreSQL configuration
mydb = psycopg2.connect(
    host="103.27.206.29",
    user="postgres",
    password="admin",
    database="ttd"
)

mycursor = mydb.cursor()

# Create tables if not exist
mycursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    timestamps TIMESTAMP NOT NULL,
    prediction VARCHAR(50) NOT NULL,
    prediction_type VARCHAR(25) NOT NULL
);
""")

mycursor.execute("""
CREATE TABLE IF NOT EXISTS personality_predictions (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    prediction VARCHAR(50) NOT NULL
);
""")

mycursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);
""")

# Folder to store uploaded files
UPLOAD_FOLDER = './static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize PyTorch models and load state_dict for signature and personality
signature_model_path = 'resnet50_signature_model.pth'
signature_model = resnet50(pretrained=False)
signature_model.fc = torch.nn.Linear(signature_model.fc.in_features, 2)
signature_model.load_state_dict(torch.load(signature_model_path, map_location=torch.device('cpu')))
signature_model.eval()

personality_model_path = 'resnet50_personality_model.pth'
personality_model = resnet50(pretrained=False)
personality_model.fc = torch.nn.Linear(personality_model.fc.in_features, 2)
personality_model.load_state_dict(torch.load(personality_model_path, map_location=torch.device('cpu')))
personality_model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Login manager configuration
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    pass

@login_manager.user_loader
def load_user(user_id):
    mycursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
    user_data = mycursor.fetchone()
    if user_data:
        user = User()
        user.id = user_data[0]
        return user
    return None

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        mycursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user_data = mycursor.fetchone()
        
        if user_data and check_password_hash(user_data[2], password):
            user = User()
            user.id = user_data[0]
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        hashed_password = generate_password_hash(password)
        
        try:
            sql = 'INSERT INTO users (username, password) VALUES (%s, %s)'
            val = (username, hashed_password)
            mycursor.execute(sql, val)
            mydb.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Registration failed: {str(e)}', 'error')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload_graduation', methods=['POST'])
@login_required
def upload_graduation():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0)
            with torch.no_grad():
                outputs = signature_model(img)
                _, signature_predicted = torch.max(outputs, 1)
                signature_predicted_class = "Will Graduate On Time" if signature_predicted.item() == 0 else "Will Graduate Not On Time"

            sql = "INSERT INTO predictions (filename, timestamps, prediction, prediction_type) VALUES (%s, %s, %s, %s)"
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            val = (filename, current_timestamp, signature_predicted_class, "graduation")
            mycursor.execute(sql, val)
            mydb.commit()

            return render_template('predict.html', 
                                   filename_graduation=filename, 
                                   signature_predicted_class=signature_predicted_class)
    return redirect(url_for('predict'))

@app.route('/upload_personality', methods=['POST'])
@login_required
def upload_personality():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0)
            with torch.no_grad():
                outputs = personality_model(img)
                _, personality_predicted = torch.max(outputs, 1)
                personality_predicted_class = "Have Ordinary Intrinsic Motivation" if personality_predicted.item() == 0 else "Have Strong Intrinsic Motivation"

            sql = "INSERT INTO predictions (filename, timestamps, prediction, prediction_type) VALUES (%s, %s, %s, %s)"
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            val = (filename, current_timestamp, personality_predicted_class, "personality")
            mycursor.execute(sql, val)
            mydb.commit()

            return render_template('predict.html', 
                                   filename_personality=filename, 
                                   personality_predicted_class=personality_predicted_class)
    return redirect(url_for('predict'))
            
@app.route('/reset_graduation', methods=['POST'])
@login_required
def reset_graduation():
    sql = "DELETE FROM predictions"
    mycursor.execute(sql)
    mydb.commit()

    return redirect(url_for('prediction'))

@app.route('/reset_personality', methods=['POST'])
@login_required
def reset_personality():
    sql = "DELETE FROM personality_predictions"
    mycursor.execute(sql)
    mydb.commit()

    return redirect(url_for('prediction'))

# Route for showing prediction results
@app.route('/prediction')
@login_required
def prediction():
    # Retrieve prediction data from database
    sql = "SELECT * FROM predictions"
    mycursor.execute(sql)
    predictions = mycursor.fetchall()

    return render_template('predict.html', predictions=predictions)

@app.route('/report')
@login_required
def report():
    # Retrieve prediction data from database
    sql = "SELECT * FROM predictions"
    mycursor.execute(sql)
    report = mycursor.fetchall()

    return render_template('report.html', report=report)

if _name_ == '_main_':
    app.run(debug=True)