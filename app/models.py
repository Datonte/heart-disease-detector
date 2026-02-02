from . import db
from flask_login import UserMixin
from datetime import datetime
import json

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone_number = db.Column(db.String(20))
    role = db.Column(db.String(20), default='Staff')  # Admin, Doctor, Nurse, etc.
    password_hash = db.Column(db.String(128))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(10))
    dob = db.Column(db.Date)
    phone = db.Column(db.String(20))
    address = db.Column(db.String(200))
    next_of_kin = db.Column(db.String(100))
    medical_history = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='patient', lazy=True)
    appointments = db.relationship('Appointment', backref='patient', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    prediction_result = db.Column(db.String(50))  # "Heart Disease Detected" / "No Heart Disease"
    probability_score = db.Column(db.Float)
    model_used = db.Column(db.String(50))  # "Logistic Regression", "Random Forest"
    input_data = db.Column(db.Text)  # JSON string of input features
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Assigned doctor
    doctor = db.relationship('User', backref='appointments')
    appointment_date = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='Pending')  # Pending, Completed, Cancelled
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
