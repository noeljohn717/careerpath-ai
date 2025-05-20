from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.models import User
from flask_login import login_user, logout_user, login_required
from app import db
from app.forms import RegisterForm
from app.forms import LoginForm  
from werkzeug.security import generate_password_hash, check_password_hash

auth = Blueprint('auth', __name__)

from werkzeug.security import generate_password_hash, check_password_hash

@auth.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        # Check if the email already exists
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('Email already registered. Please log in.', 'danger')
            return redirect(url_for('auth.login'))

        # Create a new user with the correct hashing method
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registered successfully! Please login.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register.html', form=form)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():  # Validates the form on POST
        # Check if the user exists
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):  # Corrected line
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('main.dashboard'))  # Updated to correct endpoint
        else:
            flash('Invalid email or password.', 'danger')

    # If form validation fails, render the form with error messages
    return render_template('login.html', form=form)

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')
