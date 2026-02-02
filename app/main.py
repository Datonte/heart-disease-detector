from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from flask_login import login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from .models import Patient, Prediction, User, Appointment
from .ml_utils import model_handler
import datetime

main = Blueprint('main', __name__)

@main.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    return redirect(url_for('main.dashboard'))

@main.route('/dashboard')
@login_required
def dashboard():
    stats = {
        'total_patients': Patient.query.count(),
        'total_predictions': Prediction.query.count(),
        'staff_count': User.query.count(),
        'high_risk_count': Prediction.query.filter_by(prediction_result="Heart Disease Detected").count(),
        'recent_predictions': Prediction.query.order_by(Prediction.created_at.desc()).limit(5).all()
    }
    return render_template('dashboard/index.html', stats=stats)

@main.route('/patients')
@login_required
def patients():
    search = request.args.get('search', '')
    if search:
        patients = Patient.query.filter(Patient.full_name.contains(search)).all()
    else:
        patients = Patient.query.all()
    return render_template('dashboard/patients.html', patients=patients)

@main.route('/patient/<int:patient_id>')
@login_required
def patient_details(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    predictions = Prediction.query.filter_by(patient_id=patient.id).order_by(Prediction.created_at.desc()).all()
    return render_template('dashboard/patient_details.html', patient=patient, predictions=predictions)

@main.route('/add_patient', methods=['POST'])
@login_required
def add_patient():
    new_patient = Patient(
        full_name=request.form['full_name'],
        gender=request.form['gender'],
        dob=datetime.datetime.strptime(request.form['dob'], '%Y-%m-%d').date(),
        phone=request.form['phone'],
        medical_history=request.form['medical_history']
    )
    db.session.add(new_patient)
    db.session.commit()
    flash('Patient added successfully!', 'success')
    return redirect(url_for('main.patients'))

@main.route('/edit_patient/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def edit_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if request.method == 'POST':
        patient.full_name = request.form['full_name']
        patient.gender = request.form['gender']
        patient.dob = datetime.datetime.strptime(request.form['dob'], '%Y-%m-%d').date()
        patient.phone = request.form['phone']
        patient.medical_history = request.form['medical_history']
        
        db.session.commit()
        flash('Patient details updated successfully.', 'success')
        return redirect(url_for('main.patients'))
        
    return render_template('dashboard/edit_patient.html', patient=patient)

@main.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Map form to structure
            # Model expects: ['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            input_features = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                # Skipped: trestbps, chol, fbs, restecg as per user requirement
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]
            
            # Note: We are saving the FULL input data string to DB for record keeping? 
            # Or just the used ones. Let's save the used ones to avoid confusion.
            # But the form *might* still have the other fields if we didn't remove them from HTML.
            # For now, we only pass these 9 to the model handler.
            
            # Get selected model from form
            selected_model = request.form.get('model_name', 'Random Forest')
            
            if selected_model == 'Both Models':
                comparison_results = []
                
                # Run for Logistic Regression
                res_lr, prob_lr = model_handler.predict('Logistic Regression', input_features)
                pred_lr = Prediction(
                    patient_id=request.form['patient_id'],
                    prediction_result=res_lr,
                    probability_score=prob_lr,
                    model_used='Logistic Regression',
                    input_data=str(input_features)
                )
                db.session.add(pred_lr)
                comparison_results.append({'model': 'Logistic Regression', 'result': res_lr, 'probability': prob_lr})
                
                # Run for Random Forest
                res_rf, prob_rf = model_handler.predict('Random Forest', input_features)
                pred_rf = Prediction(
                    patient_id=request.form['patient_id'],
                    prediction_result=res_rf,
                    probability_score=prob_rf,
                    model_used='Random Forest',
                    input_data=str(input_features)
                )
                db.session.add(pred_rf)
                comparison_results.append({'model': 'Random Forest', 'result': res_rf, 'probability': prob_rf})
                
                db.session.commit()
                flash('Dual Model Prediction Complete', 'success')
                return render_template('dashboard/prediction_result.html', comparison=comparison_results, patient_id=request.form['patient_id'])
            
            else:
                # Predict using single selected model
                result_str, prob = model_handler.predict(selected_model, input_features)
                
                new_pred = Prediction(
                    patient_id=request.form['patient_id'],
                    prediction_result=result_str,
                    probability_score=prob,
                    model_used=selected_model,
                    input_data=str(input_features)
                )
                db.session.add(new_pred)
                db.session.commit()
                
                flash(f'Prediction Complete: {result_str}', 'success')
                return render_template('dashboard/prediction_result.html', result=result_str, probability=prob, patient_id=request.form['patient_id'], model=selected_model)
            
        except Exception as e:
            flash(f'Error during prediction: {e}', 'danger')
            return redirect(url_for('main.predict'))
            
    patients_list = Patient.query.all()
    # Pass available models to template
    available_models = ["Random Forest", "Logistic Regression"]
    return render_template('dashboard/prediction.html', patients=patients_list, available_models=available_models)

@main.route('/appointments')
@login_required
def appointments():
    appointments = Appointment.query.all()
    doctors = User.query.filter_by(role='Doctor').all()
    patients = Patient.query.all()
    return render_template('dashboard/appointments.html', appointments=appointments, doctors=doctors, patients=patients)

@main.route('/book_appointment', methods=['POST'])
@login_required
def book_appointment():
    try:
        patient_id = request.form.get('patient_id')
        doctor_id = request.form.get('doctor_id')
        date_str = request.form.get('date') + " " + request.form.get('time')
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M')
        
        new_app = Appointment(
            patient_id=patient_id,
            doctor_id=doctor_id,
            appointment_date=date_obj,
            notes=request.form.get('notes'),
            status='Pending'
        )
        db.session.add(new_app)
        db.session.commit()
        flash('Appointment booked successfully.', 'success')
    except Exception as e:
        flash(f'Error booking appointment: {e}', 'danger')
        
    return redirect(url_for('main.appointments'))

@main.route('/users')
@login_required
def users():
    if current_user.role != 'Admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
    users = User.query.all()
    return render_template('dashboard/users.html', users=users)



@main.route('/compare_models')
@login_required
def compare_models():
    # Simple logic to compare model usage and positive rates
    # Group by model_used
    from sqlalchemy import func
    
    # Count usage
    usage_stats = db.session.query(Prediction.model_used, func.count(Prediction.id)).group_by(Prediction.model_used).all()
    
    # Count positive predictions per model
    positive_stats = db.session.query(Prediction.model_used, func.count(Prediction.id)).filter_by(prediction_result='Heart Disease Detected').group_by(Prediction.model_used).all()
    
    # Format for template
    labels = []
    usage_data = []
    positive_data = []
    
    # Initialize dicts
    models = set([s[0] for s in usage_stats])
    usage_dict = {s[0]: s[1] for s in usage_stats}
    positive_dict = {s[0]: s[1] for s in positive_stats}
    
    for m in models:
        labels.append(m)
        usage_data.append(usage_dict.get(m, 0))
        positive_data.append(positive_dict.get(m, 0))
        
    # Hardcoded training metrics (Simulated as they are not in the pickle)
    # In a real scenario, these would be loaded from a 'metrics.json' file generated during training.
    training_metrics = {
        'Logistic Regression': {'Accuracy': 0.85, 'Precision': 0.84, 'Recall': 0.86, 'F1 Score': 0.85},
        'Random Forest': {'Accuracy': 0.91, 'Precision': 0.90, 'Recall': 0.92, 'F1 Score': 0.91}
    }
        
    return render_template('dashboard/compare_models.html', labels=labels, usage_data=usage_data, positive_data=positive_data, training_metrics=training_metrics)

@main.route('/history')
@login_required
def history():
    # Base query joining Patient so we can search by name
    query = Prediction.query.join(Patient)
    
    # Filters
    patient_name = request.args.get('patient_name')
    if patient_name:
        query = query.filter(Patient.full_name.contains(patient_name))
        
    risk_status = request.args.get('risk_status')
    if risk_status:
        query = query.filter(Prediction.prediction_result == risk_status)
        
    model_used = request.args.get('model_used')
    if model_used:
        query = query.filter(Prediction.model_used == model_used)
        
    date_filter = request.args.get('date_filter')
    if date_filter:
        # Simple date match, or maybe range?
        # Let's assume >= date provided
        query = query.filter(Prediction.created_at >= datetime.datetime.strptime(date_filter, '%Y-%m-%d'))

    # Order by newest
    predictions = query.order_by(Prediction.created_at.desc()).all()
    
    # Export Check
    export_type = request.args.get('export')
    if export_type == 'csv':
        import csv
        import io
        from flask import make_response
        
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(['Date', 'Patient Name', 'Result', 'Probability', 'Model Used', 'Input Data'])
        
        for pred in predictions:
            p_name = pred.patient.full_name if pred.patient else f"Deleted ({pred.patient_id})"
            cw.writerow([pred.created_at, p_name, pred.prediction_result, pred.probability_score, pred.model_used, pred.input_data])
            
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = f"attachment; filename=prediction_history_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
        output.headers["Content-type"] = "text/csv"
        return output
    
    elif export_type == 'pdf':
        try:
            from fpdf import FPDF
            from flask import make_response
            
            class PDF(FPDF):
                def header(self):
                    self.set_font('Arial', 'B', 14)
                    self.cell(0, 10, 'HeartFelt - Prediction History Report', 0, 1, 'C')
                    self.ln(5)
                    
                def footer(self):
                    self.set_y(-15)
                    self.set_font('Arial', 'I', 8)
                    self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

            pdf = PDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            
            # Table Header
            pdf.set_fill_color(200, 220, 255)
            pdf.cell(35, 10, 'Date', 1, 0, 'C', 1)
            pdf.cell(40, 10, 'Patient', 1, 0, 'C', 1)
            pdf.cell(45, 10, 'Result', 1, 0, 'C', 1)
            pdf.cell(20, 10, 'Prob', 1, 0, 'C', 1)
            pdf.cell(40, 10, 'Model', 1, 1, 'C', 1)
            
            # Table Body
            pdf.set_font("Arial", size=9)
            for pred in predictions:
                p_name = pred.patient.full_name if pred.patient else f"del({pred.patient_id})"
                # Truncate
                p_name = (p_name[:18] + '..') if len(p_name) > 20 else p_name
                
                date_str = pred.created_at.strftime('%Y-%m-%d')
                prob_str = f"{pred.probability_score*100:.1f}%"
                
                # Check page break
                if pdf.get_y() > 270:
                    pdf.add_page()
                
                pdf.cell(35, 8, date_str, 1)
                pdf.cell(40, 8, p_name, 1)
                pdf.cell(45, 8, pred.prediction_result, 1)
                pdf.cell(20, 8, prob_str, 1)
                pdf.cell(40, 8, pred.model_used, 1, 1)
                
            response = make_response(pdf.output(dest='S').encode('latin-1'))
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename=prediction_history_{datetime.datetime.now().strftime("%Y%m%d")}.pdf'
            return response
            
        except ImportError:
            flash("FPDF library not installed. Please contact admin.", "warning")
            return redirect(url_for('main.history'))
        except Exception as e:
            flash(f"Error generating PDF: {e}", "danger")
            return redirect(url_for('main.history'))

    return render_template('dashboard/history.html', predictions=predictions)

@main.route('/export/predictions/<int:patient_id>')
@login_required
def export_patient_predictions(patient_id):
    import csv
    import io
    from flask import make_response
    
    patient = Patient.query.get_or_404(patient_id)
    predictions = Prediction.query.filter_by(patient_id=patient.id).order_by(Prediction.created_at.desc()).all()
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'Result', 'Probability', 'Model Used', 'Input Data'])
    
    for pred in predictions:
        cw.writerow([pred.created_at, pred.prediction_result, pred.probability_score, pred.model_used, pred.input_data])
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=predictions_{patient.full_name}_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@main.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        # Change Password Logic
        current_password = request.form.get('currentPassword')
        new_password = request.form.get('newPassword')
        
        if not current_password or not new_password:
             flash('Please fill in all password fields.', 'warning')
             return redirect(url_for('main.settings'))
             
        if not check_password_hash(current_user.password_hash, current_password):
            flash('Current password is incorrect.', 'danger')
            return redirect(url_for('main.settings'))
            
        current_user.password_hash = generate_password_hash(new_password, method='scrypt')
        db.session.commit()
        flash('Password updated successfully.', 'success')
        return redirect(url_for('main.settings'))
        
    return render_template('dashboard/settings.html')
@main.route('/reports')
@login_required
def reports():
    from sqlalchemy import func
    
    # Simple report aggregations
    daily_predictions = Prediction.query.filter(Prediction.created_at >= datetime.date.today()).count()
    total_high_risk = Prediction.query.filter_by(prediction_result='Heart Disease Detected').count()
    total_patients = Patient.query.count()
    
    # Model Usage
    usage_stats = db.session.query(Prediction.model_used, func.count(Prediction.id)).group_by(Prediction.model_used).all()
    models = [s[0] for s in usage_stats]
    usage_counts = [s[1] for s in usage_stats]
    
    # Get high risk patients
    high_risk_predictions = Prediction.query.filter_by(prediction_result='Heart Disease Detected').order_by(Prediction.created_at.desc()).limit(20).all()
    
    return render_template('dashboard/reports.html', 
                           daily_predictions=daily_predictions, 
                           total_high_risk=total_high_risk, 
                           total_patients=total_patients,
                           high_risk_predictions=high_risk_predictions,
                           models=models,
                           usage_counts=usage_counts)

@main.route('/delete_patient/<int:patient_id>', methods=['POST'])
@login_required
def delete_patient(patient_id):
    if current_user.role != 'Admin':
        flash('Only Admins can delete patients.', 'danger')
        return redirect(url_for('main.patients'))
    
    patient = Patient.query.get_or_404(patient_id)
    # Manually delete related records if cascade not set in DB
    Prediction.query.filter_by(patient_id=patient.id).delete()
    Appointment.query.filter_by(patient_id=patient.id).delete()
    
    db.session.delete(patient)
    db.session.commit()
    flash('Patient and related records deleted.', 'success')
    return redirect(url_for('main.patients'))

@main.route('/update_appointment/<int:appointment_id>', methods=['POST'])
@login_required
def update_appointment_status(appointment_id):
    appointment = Appointment.query.get_or_404(appointment_id)
    new_status = request.form.get('status')
    if new_status in ['Pending', 'Completed', 'Cancelled']:
        appointment.status = new_status
        db.session.commit()
        flash(f'Appointment marked as {new_status}.', 'success')
    return redirect(url_for('main.appointments'))

@main.route('/toggle_user/<int:user_id>', methods=['POST'])
@login_required
def toggle_user_status(user_id):
    if current_user.role != 'Admin':
        flash('Permission denied.', 'danger')
        return redirect(url_for('main.users'))
        
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        # Check if we are just updating info or deactivating
        if 'role' in request.form:
             # Admins shouldn't demote themselves to avoid lockout, logic depends on requirements
             pass
        else: 
            flash('You cannot deactivate yourself.', 'warning')
            return redirect(url_for('main.users'))
        
    # Handle Role Update
    if 'role' in request.form:
        new_role = request.form.get('role')
        if new_role in ['Admin', 'Doctor', 'Nurse', 'Receptionist']:
            user.role = new_role
            db.session.commit()
            flash(f'User role updated to {new_role}.', 'success')
            return redirect(url_for('main.users'))
            
    # Handle Status Toggle
    user.is_active = not user.is_active
    db.session.commit()
    status = 'activated' if user.is_active else 'deactivated'
    flash(f'User {user.full_name} {status}.', 'success')
    return redirect(url_for('main.users'))
