from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import json
import os
import sqlite3
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize database and ensure schema is compatible
def init_db():
    conn = sqlite3.connect("news.db")
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news TEXT,
            result TEXT,
            user_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    cursor.execute("PRAGMA table_info(predictions)")
    columns = [row[1] for row in cursor.fetchall()]
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")
    if "timestamp" not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN timestamp DATETIME")

    conn.commit()
    conn.close()

init_db()


def load_model_metrics():
    metrics_path = "model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

MODEL_METRICS = load_model_metrics()

@app.context_processor
def inject_user():
    return {
        "logged_in": session.get("user_id") is not None,
        "username": session.get("username")
    }

def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped_view

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect')
@login_required
def detect():
    return render_template('detect.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        conn = sqlite3.connect('news.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        existing = cursor.fetchone()
        if existing:
            flash('That username is already taken.', 'error')
            conn.close()
            return render_template('register.html')

        password_hash = generate_password_hash(password)
        cursor.execute(
            'INSERT INTO users (username, password_hash) VALUES (?, ?)',
            (username, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        session['user_id'] = user_id
        session['username'] = username
        flash('Account created successfully. You are now logged in.', 'success')
        return redirect(url_for('home'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        conn = sqlite3.connect('news.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user is None or not check_password_hash(user[1], password):
            flash('Invalid username or password.', 'error')
            return render_template('login.html')

        session['user_id'] = user[0]
        session['username'] = username
        flash('Logged in successfully.', 'success')

        next_page = request.args.get('next')
        if next_page and next_page.startswith('/'):
            return redirect(next_page)
        return redirect(url_for('home'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    news = request.form['news']
    data = vectorizer.transform([news])

    prediction = model.predict(data)
    proba = model.predict_proba(data)[0]

    result = "Real News" if prediction[0] == 1 else "Fake News"
    confidence_real = round(proba[1] * 100, 1)
    confidence_fake = round(proba[0] * 100, 1)

    conn = sqlite3.connect('news.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO predictions (news, result, user_id, timestamp) VALUES (?, ?, ?, CURRENT_TIMESTAMP)',
        (news, result, session['user_id'])
    )
    conn.commit()
    conn.close()

    return render_template(
        'detect.html',
        prediction=result,
        confidence_real=confidence_real,
        confidence_fake=confidence_fake
    )

@app.route('/history')
@login_required
def history():
    conn = sqlite3.connect('news.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, news, result FROM predictions WHERE user_id = ? ORDER BY id DESC',
        (session['user_id'],)
    )
    data = cursor.fetchall()
    conn.close()
    return render_template('history.html', data=data)

@app.route('/dashboard')
@login_required
def dashboard():
    conn = sqlite3.connect('news.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM predictions WHERE user_id = ?', (session['user_id'],))
    total = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM predictions WHERE user_id = ? AND result = ?', (session['user_id'], 'Real News'))
    real_news = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM predictions WHERE user_id = ? AND result = ?', (session['user_id'], 'Fake News'))
    fake_news = cursor.fetchone()[0]

    cursor.execute(
        "SELECT DATE(timestamp), COUNT(*) FROM predictions WHERE user_id = ? GROUP BY DATE(timestamp) ORDER BY DATE(timestamp)",
        (session['user_id'],)
    )
    date_rows = cursor.fetchall()
    prediction_dates = [row[0] for row in date_rows]
    prediction_counts = [row[1] for row in date_rows]

    conn.close()

    model_accuracy = MODEL_METRICS.get('accuracy')
    training_samples = MODEL_METRICS.get('training_samples')
    real_samples = MODEL_METRICS.get('real_samples')
    fake_samples = MODEL_METRICS.get('fake_samples')
    tn = MODEL_METRICS.get('tn')
    fp = MODEL_METRICS.get('fp')
    fn = MODEL_METRICS.get('fn')
    tp = MODEL_METRICS.get('tp')

    return render_template(
        'dashboard.html',
        total=total,
        real=real_news,
        fake=fake_news,
        accuracy=model_accuracy,
        training_samples=training_samples,
        real_samples=real_samples,
        fake_samples=fake_samples,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        prediction_dates=prediction_dates,
        prediction_counts=prediction_counts
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
