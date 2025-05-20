from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from app.models import User
from app import db

admin = Blueprint('admin', __name__)

@admin.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Unauthorized access.')
        return redirect(url_for('main.dashboard'))

    users = User.query.all()
    return render_template('admin.html', users=users)

@admin.route('/make-premium/<int:user_id>', methods=['POST'])
@login_required
def make_premium(user_id):
    if not current_user.is_admin:
        flash('Unauthorized')
        return redirect(url_for('main.dashboard'))

    user = User.query.get_or_404(user_id)
    user.is_premium = True
    db.session.commit()
    flash(f'{user.username} is now Premium!')
    return redirect(url_for('admin.admin_dashboard'))
