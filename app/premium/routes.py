from flask import Blueprint, render_template
from flask_login import login_required, current_user

premium = Blueprint('premium', __name__)

@premium.route('/premium')
@login_required
def premium_dashboard():
    if not current_user.is_premium:
        return "You must be a Premium User to access this page.", 403
    return render_template('premium_dashboard.html')
