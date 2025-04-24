from app import app
from flask import render_template, request
from app.ai_module import get_career_recommendation

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/suggest', methods=['POST'])
def suggest():
    user_input = request.form['skills']
    suggestions = get_career_recommendation(user_input)
    return render_template('dashboard.html', suggestions=suggestions)
