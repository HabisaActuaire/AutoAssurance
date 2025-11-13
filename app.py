from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'votre_cle_secrete_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///insurance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# Modèles de données
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ContractRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')
    
    # Caractéristiques du client
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    driving_experience = db.Column(db.Integer, nullable=False)
    annual_mileage = db.Column(db.Integer, nullable=False)
    vehicle_age = db.Column(db.Integer, nullable=False)
    vehicle_power = db.Column(db.Integer, nullable=False)
    vehicle_type = db.Column(db.String(50), nullable=False)
    vehicle_category = db.Column(db.String(50), nullable=False)
    previous_claims = db.Column(db.Integer, default=0)
    credit_score = db.Column(db.Integer, nullable=False)
    coverage_type = db.Column(db.String(20), nullable=False)
    
    # Documents
    document_filename = db.Column(db.String(255))
    
    # Prédictions et prime
    predicted_claims = db.Column(db.Float)
    claim_probability = db.Column(db.Float)
    predicted_cost = db.Column(db.Float)
    insurance_premium = db.Column(db.Float)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    client = db.relationship('User', backref=db.backref('contracts', lazy=True))

class PremiumProposal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    contract_id = db.Column(db.Integer, db.ForeignKey('contract_request.id'), nullable=False)
    premium_amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    contract = db.relationship('ContractRequest', backref=db.backref('premium_proposals', lazy=True))

# Types de véhicules au Burundi
VEHICLE_TYPES = {
    'voiture': ['Berline', '4x4', 'Minibus', 'Bus', 'Camionnette'],
    'moto': ['Moto 125cc', 'Moto 250cc', 'Moto >250cc', 'Scooter'],
    'camion': ['Camion léger', 'Camion lourd', 'Porteur'],
    'special': ['Tracteur', 'Engin BTP', 'Remorque']
}

# Initialisation du modèle ML
class ClaimPredictor:
    def __init__(self):
        self.frequency_model = None
        self.severity_model = None
        self.load_or_train_models()
    
    def load_or_train_models(self):
        try:
            self.frequency_model = joblib.load('frequency_model.pkl')
            self.severity_model = joblib.load('severity_model.pkl')
        except:
            self.train_models()
    
    def train_models(self):
        np.random.seed(42)
        n_samples = 10000
        
        X = np.column_stack([
            np.random.randint(18, 80, n_samples),
            np.random.randint(0, 2, n_samples),
            np.random.randint(1, 50, n_samples),
            np.random.randint(5000, 50000, n_samples),
            np.random.randint(0, 20, n_samples),
            np.random.randint(50, 300, n_samples),
            np.random.randint(0, 4, n_samples),
            np.random.randint(0, 5, n_samples),
            np.random.randint(300, 850, n_samples),
            np.random.randint(0, 3, n_samples)
        ])
        
        claims_frequency = np.random.poisson(
            np.exp(-3 + 
                   0.02 * X[:,0] -
                   0.1 * X[:,1] +
                   0.01 * X[:,2] - 
                   0.0001 * X[:,3] +
                   0.05 * X[:,4] +
                   0.002 * X[:,5] +
                   0.1 * X[:,6] +
                   0.3 * X[:,7] +
                   0.001 * X[:,8] -
                   0.2 * X[:,9]
            )
        )
        
        claims_severity = np.random.gamma(
            shape=2,
            scale=500000 * np.exp(
                0.01 * X[:,0] +
                0.05 * X[:,1] +
                0.005 * X[:,2] -
                0.00005 * X[:,3] +
                0.02 * X[:,4] +
                0.003 * X[:,5] +
                0.08 * X[:,6] +
                0.1 * X[:,7] +
                0.0005 * X[:,8] -
                0.15 * X[:,9]
            )
        )
        
        self.frequency_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.frequency_model.fit(X, claims_frequency)
        
        self.severity_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.severity_model.fit(X, claims_severity)
        
        joblib.dump(self.frequency_model, 'frequency_model.pkl')
        joblib.dump(self.severity_model, 'severity_model.pkl')
    
    def predict_claims(self, features):
        vehicle_type_map = {
            'Berline': 0, '4x4': 1, 'Minibus': 2, 'Bus': 3, 'Camionnette': 4,
            'Moto 125cc': 0, 'Moto 250cc': 1, 'Moto >250cc': 2, 'Scooter': 3,
            'Camion léger': 0, 'Camion lourd': 1, 'Porteur': 2,
            'Tracteur': 0, 'Engin BTP': 1, 'Remorque': 2
        }
        
        coverage_map = {'basique': 0, 'intermediaire': 1, 'premium': 2}
        
        feature_array = np.array([[
            features['age'],
            1 if features['gender'] == 'male' else 0,
            features['driving_experience'],
            features['annual_mileage'],
            features['vehicle_age'],
            features['vehicle_power'],
            vehicle_type_map.get(features['vehicle_type'], 0),
            features['previous_claims'],
            features['credit_score'],
            coverage_map.get(features['coverage_type'], 0)
        ]])
        
        predicted_frequency = max(0, self.frequency_model.predict(feature_array)[0])
        predicted_severity = max(0, self.severity_model.predict(feature_array)[0])
        
        claim_probability = 1 - np.exp(-predicted_frequency)
        
        # Calcul de la prime (en BIF)
        base_premium = predicted_severity * predicted_frequency
        risk_factor = 1 + claim_probability
        insurance_premium = base_premium * risk_factor * 1.2
        
        return {
            'predicted_claims': round(predicted_frequency, 2),
            'claim_probability': round(claim_probability, 4),
            'predicted_cost': round(predicted_severity, 2),
            'insurance_premium': round(insurance_premium, 2)
        }

predictor = ClaimPredictor()

# Routes principales
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            flash('Connexion réussie!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Identifiants incorrects', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        phone = request.form['phone']
        role = request.form['role']
        
        if User.query.filter_by(username=username).first():
            flash('Nom d\'utilisateur déjà utilisé', 'error')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(
            username=username,
            password=hashed_password,
            email=email,
            phone=phone,
            role=role
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Compte créé avec succès!', 'success')
        return redirect(url_for('login'))
    
    return render_template('login.html', register=True)

@app.route('/admin/create_user', methods=['GET', 'POST'])
def create_user():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        phone = request.form['phone']
        role = request.form['role']
        
        if User.query.filter_by(username=username).first():
            flash('Nom d\'utilisateur déjà utilisé', 'error')
            return redirect(url_for('create_user'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(
            username=username,
            password=hashed_password,
            email=email,
            phone=phone,
            role=role
        )
        
        db.session.add(new_user)
        db.session.commit()
        flash(f'Compte {role} créé avec succès!', 'success')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('create_user.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if session['role'] == 'client':
        my_contracts = ContractRequest.query.filter_by(client_id=session['user_id']).all()
        return render_template('client_dashboard.html', contracts=my_contracts)
    elif session['role'] == 'agent':
        pending_requests = ContractRequest.query.filter_by(status='pending').all()
        return render_template('agent_dashboard.html', requests=pending_requests, vehicle_types=VEHICLE_TYPES)
    elif session['role'] == 'admin':
        return redirect(url_for('admin_dashboard'))
    
    return redirect(url_for('login'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    users = User.query.all()
    all_requests = ContractRequest.query.all()
    return render_template('admin_dashboard.html', users=users, requests=all_requests)

@app.route('/submit_contract', methods=['POST'])
def submit_contract():
    if 'user_id' not in session or session['role'] != 'client':
        return redirect(url_for('login'))
    
    # Gestion du fichier uploadé
    document = request.files['document']
    filename = None
    if document and document.filename != '':
        filename = secure_filename(f"{session['user_id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{document.filename}")
        document.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    features = {
        'age': int(request.form['age']),
        'gender': request.form['gender'],
        'driving_experience': int(request.form['driving_experience']),
        'annual_mileage': int(request.form['annual_mileage']),
        'vehicle_age': int(request.form['vehicle_age']),
        'vehicle_power': int(request.form['vehicle_power']),
        'vehicle_type': request.form['vehicle_type'],
        'previous_claims': int(request.form['previous_claims']),
        'credit_score': int(request.form['credit_score']),
        'coverage_type': request.form['coverage_type']
    }
    
    # Sauvegarde sans prédictions (calculées côté agent/admin)
    new_contract = ContractRequest(
        client_id=session['user_id'],
        age=features['age'],
        gender=features['gender'],
        driving_experience=features['driving_experience'],
        annual_mileage=features['annual_mileage'],
        vehicle_age=features['vehicle_age'],
        vehicle_power=features['vehicle_power'],
        vehicle_type=features['vehicle_type'],
        vehicle_category=request.form['vehicle_category'],
        previous_claims=features['previous_claims'],
        credit_score=features['credit_score'],
        coverage_type=features['coverage_type'],
        document_filename=filename
    )
    
    db.session.add(new_contract)
    db.session.commit()
    
    flash('Demande soumise avec succès! Un agent va traiter votre demande.', 'success')
    return redirect(url_for('dashboard'))

@app.route('/agent/submit_contract', methods=['POST'])
def agent_submit_contract():
    if 'user_id' not in session or session['role'] != 'agent':
        return redirect(url_for('login'))
    
    features = {
        'age': int(request.form['age']),
        'gender': request.form['gender'],
        'driving_experience': int(request.form['driving_experience']),
        'annual_mileage': int(request.form['annual_mileage']),
        'vehicle_age': int(request.form['vehicle_age']),
        'vehicle_power': int(request.form['vehicle_power']),
        'vehicle_type': request.form['vehicle_type'],
        'previous_claims': int(request.form['previous_claims']),
        'credit_score': int(request.form['credit_score']),
        'coverage_type': request.form['coverage_type']
    }
    
    # Calcul des prédictions
    predictions = predictor.predict_claims(features)
    
    new_contract = ContractRequest(
        client_id=session['user_id'],
        age=features['age'],
        gender=features['gender'],
        driving_experience=features['driving_experience'],
        annual_mileage=features['annual_mileage'],
        vehicle_age=features['vehicle_age'],
        vehicle_power=features['vehicle_power'],
        vehicle_type=features['vehicle_type'],
        vehicle_category=request.form['vehicle_category'],
        previous_claims=features['previous_claims'],
        credit_score=features['credit_score'],
        coverage_type=features['coverage_type'],
        predicted_claims=predictions['predicted_claims'],
        claim_probability=predictions['claim_probability'],
        predicted_cost=predictions['predicted_cost'],
        insurance_premium=predictions['insurance_premium']
    )
    
    db.session.add(new_contract)
    db.session.commit()
    
    flash('Contrat créé avec succès avec prédictions calculées!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/calculate_predictions/<int:contract_id>')
def calculate_predictions(contract_id):
    if 'user_id' not in session or session['role'] not in ['agent', 'admin']:
        return redirect(url_for('login'))
    
    contract = ContractRequest.query.get_or_404(contract_id)
    
    features = {
        'age': contract.age,
        'gender': contract.gender,
        'driving_experience': contract.driving_experience,
        'annual_mileage': contract.annual_mileage,
        'vehicle_age': contract.vehicle_age,
        'vehicle_power': contract.vehicle_power,
        'vehicle_type': contract.vehicle_type,
        'previous_claims': contract.previous_claims,
        'credit_score': contract.credit_score,
        'coverage_type': contract.coverage_type
    }
    
    predictions = predictor.predict_claims(features)
    
    contract.predicted_claims = predictions['predicted_claims']
    contract.claim_probability = predictions['claim_probability']
    contract.predicted_cost = predictions['predicted_cost']
    contract.insurance_premium = predictions['insurance_premium']
    
    db.session.commit()
    
    flash('Prédictions calculées avec succès!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/send_premium/<int:contract_id>', methods=['POST'])
def send_premium(contract_id):
    if 'user_id' not in session or session['role'] not in ['agent', 'admin']:
        return redirect(url_for('login'))
    
    contract = ContractRequest.query.get_or_404(contract_id)
    premium_amount = float(request.form['premium_amount'])
    
    # Créer une proposition de prime
    premium_proposal = PremiumProposal(
        contract_id=contract_id,
        premium_amount=premium_amount
    )
    
    contract.status = 'prime_sent'
    
    db.session.add(premium_proposal)
    db.session.commit()
    
    flash('Prime envoyée au client avec succès!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/client/respond_premium/<int:proposal_id>', methods=['POST'])
def respond_premium(proposal_id):
    if 'user_id' not in session or session['role'] != 'client':
        return redirect(url_for('login'))
    
    proposal = PremiumProposal.query.get_or_404(proposal_id)
    decision = request.form['decision']
    
    if decision == 'accept':
        proposal.status = 'accepted'
        proposal.contract.status = 'approved'
        flash('Contrat accepté! Votre assurance est maintenant active.', 'success')
    else:
        proposal.status = 'rejected'
        proposal.contract.status = 'rejected'
        flash('Contrat refusé.', 'info')
    
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/update_request_status/<int:request_id>', methods=['POST'])
def update_request_status(request_id):
    if 'user_id' not in session or session['role'] not in ['agent', 'admin']:
        return redirect(url_for('login'))
    
    contract_request = ContractRequest.query.get_or_404(request_id)
    new_status = request.form['status']
    
    contract_request.status = new_status
    db.session.commit()
    
    flash('Statut mis à jour avec succès!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/download_document/<filename>')
def download_document(filename):
    if 'user_id' not in session or session['role'] not in ['agent', 'admin']:
        return redirect(url_for('login'))
    
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/logout')
def logout():
    session.clear()
    flash('Déconnexion réussie', 'success')
    return redirect(url_for('login'))

def init_db():
    with app.app_context():
        # Supprimer et recréer la base de données
        db.drop_all()
        db.create_all()
        
        # Création d'un administrateur par défaut
        admin_user = User(
            username='admin',
            password=generate_password_hash('admin123'),
            email='admin@assurance.bi',
            phone='+25761096807',
            role='admin'
        )
        db.session.add(admin_user)
        
        # Création d'un agent par défaut
        agent_user = User(
            username='agent',
            password=generate_password_hash('agent123'),
            email='agent@assurance.bi',
            phone='+25768463665',
            role='agent'
        )
        db.session.add(agent_user)
        
        # Création d'un client par défaut
        client_user = User(
            username='client',
            password=generate_password_hash('client123'),
            email='client@assurance.bi',
            phone='+25772289341',
            role='client'
        )
        db.session.add(client_user)
        
        db.session.commit()
        print("Base de données initialisée avec les utilisateurs par défaut")

if __name__ == '__main__':
    init_db()
    app.run(debug=True)