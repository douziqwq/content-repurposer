import os
import json
import time
import uuid
import threading
import requests
from datetime import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash, session
)
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['DATABASE'] = os.path.join(app.instance_path, 'content_repurposer.db')

# Ensure instance folder exists
os.makedirs(app.instance_path, exist_ok=True)

# ===== Database Setup (SQLite) =====
import sqlite3

def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database tables."""
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_pro INTEGER DEFAULT 0,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            action TEXT NOT NULL DEFAULT 'repurpose',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE INDEX IF NOT EXISTS idx_usage_user_date ON usage_logs(user_id, created_at);
    ''')
    conn.commit()
    conn.close()

init_db()

# ===== Flask-Login Setup =====
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

class User(UserMixin):
    def __init__(self, user_row):
        self.id = user_row['id']
        self.email = user_row['email']
        self.password_hash = user_row['password_hash']
        self.is_pro = bool(user_row['is_pro'])
        self.stripe_customer_id = user_row['stripe_customer_id']
        self.stripe_subscription_id = user_row['stripe_subscription_id']
        self.created_at = user_row['created_at']

    def get_monthly_usage(self):
        """Get the number of uses this month."""
        conn = get_db()
        now = datetime.now()
        row = conn.execute(
            '''SELECT COUNT(*) as count FROM usage_logs
               WHERE user_id = ? AND action = 'repurpose'
               AND strftime('%Y-%m', created_at) = ?''',
            (self.id, now.strftime('%Y-%m'))
        ).fetchone()
        conn.close()
        return row['count']

    def can_use(self):
        """Check if user can still use the service this month."""
        if self.is_pro:
            return True
        return self.get_monthly_usage() < 3

    def record_usage(self):
        """Record a usage event."""
        conn = get_db()
        conn.execute(
            'INSERT INTO usage_logs (user_id, action) VALUES (?, ?)',
            (self.id, 'repurpose')
        )
        conn.commit()
        conn.close()

@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    user_row = conn.execute('SELECT * FROM users WHERE id = ?', (int(user_id),)).fetchone()
    conn.close()
    if user_row:
        return User(user_row)
    return None

# ===== API Configuration =====
API_KEY = "sk-ogykqjysrqxvkzhkhlngacbjniyqjbmfqaxbrwcvpgqfkllg"
API_BASE_URL = "https://api.siliconflow.cn/v1"
TRANSCRIPTION_MODEL = "FunAudioLLM/SenseVoiceSmall"
REWRITE_MODEL = "deepseek-ai/DeepSeek-V3"

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# In-memory task storage
tasks = {}

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ===== Transcription & Content Generation =====

def transcribe_audio(file_path, task_id):
    """Transcribe audio file using SenseVoiceSmall model."""
    try:
        tasks[task_id]['status'] = 'transcribing'
        tasks[task_id]['progress'] = 20
        tasks[task_id]['message'] = 'Transcribing audio...'

        with open(file_path, 'rb') as audio_file:
            files = {'file': (os.path.basename(file_path), audio_file, 'audio/mpeg')}
            data = {
                'model': TRANSCRIPTION_MODEL,
            }
            headers = {
                'Authorization': f'Bearer {API_KEY}',
            }

            response = requests.post(
                f"{API_BASE_URL}/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
                timeout=300
            )

        if response.status_code == 200:
            result = response.json()
            text = result.get('text', '')
            tasks[task_id]['transcription'] = text
            tasks[task_id]['progress'] = 50
            tasks[task_id]['message'] = 'Transcription complete. Generating content...'
            return text
        else:
            error_msg = f"Transcription failed: {response.status_code} - {response.text}"
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['message'] = error_msg
            return None
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = f'Transcription error: {str(e)}'
        return None


def generate_content(text, formats, task_id):
    """Generate repurposed content using DeepSeek-V3."""
    try:
        tasks[task_id]['status'] = 'generating'
        tasks[task_id]['progress'] = 60
        tasks[task_id]['message'] = 'Generating platform-specific content...'

        format_instructions = []
        if 'twitter' in formats:
            format_instructions.append("1. Twitter/X: 5 tweets, each under 280 characters, with strong hooks. Use line breaks for readability.")
        if 'linkedin' in formats:
            format_instructions.append("2. LinkedIn: 1 professional post, 800-1200 words, with insights and takeaways. Use a conversational but authoritative tone.")
        if 'xiaohongshu' in formats:
            format_instructions.append("3. Xiaohongshu (RED): 1 post, casual tone with emojis, 500-800 characters. Include a catchy title and hashtags.")
        if 'blog' in formats:
            format_instructions.append("4. Blog Article: 1 SEO-optimized article, 1500-2000 words, with H2/H3 headings and a compelling introduction.")
        if 'video' in formats:
            format_instructions.append("5. Short Video Scripts: 3 scripts, each 30-60 seconds, with scene descriptions and dialogue.")

        prompt = f"""You are a professional content marketer. Repurpose the following content into platform-specific posts.

For each platform, match the tone, style, and format that performs best:

{chr(10).join(format_instructions)}

IMPORTANT: Return your response as a valid JSON object with these exact keys:
- "twitter": array of 5 tweet strings (only if Twitter was requested)
- "linkedin": string of the LinkedIn post (only if LinkedIn was requested)
- "xiaohongshu": string of the Xiaohongshu post (only if Xiaohongshu was requested)
- "blog": string of the blog article in markdown (only if Blog was requested)
- "video": array of 3 script strings (only if Video Scripts were requested)

Only include keys for the formats that were requested. Make sure the JSON is valid and complete.

Original content:
{text}"""

        tasks[task_id]['progress'] = 70

        response = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional content marketer who excels at repurposing content for different social media platforms. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=8000,
            stream=False
        )

        tasks[task_id]['progress'] = 90
        tasks[task_id]['message'] = 'Processing results...'

        raw_content = response.choices[0].message.content.strip()

        json_str = raw_content
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0].strip()

        try:
            results = json.loads(json_str)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            if json_match:
                results = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse model output as JSON")

        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['message'] = 'Done!'
        tasks[task_id]['results'] = results

    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = f'Generation error: {str(e)}'
        import traceback
        traceback.print_exc()


def process_task(file_path, text_content, formats, task_id, user_id):
    """Main task processing pipeline."""
    try:
        if file_path:
            text = transcribe_audio(file_path, task_id)
            if not text:
                return
        else:
            text = text_content
            tasks[task_id]['progress'] = 50
            tasks[task_id]['message'] = 'Generating content...'

        generate_content(text, formats, task_id)

        # Record usage on success
        if tasks[task_id]['status'] == 'completed' and user_id:
            conn = get_db()
            conn.execute(
                'INSERT INTO usage_logs (user_id, action) VALUES (?, ?)',
                (user_id, 'repurpose')
            )
            conn.commit()
            conn.close()

    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = f'Unexpected error: {str(e)}'
        import traceback
        traceback.print_exc()
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass


# ===== Routes: Auth =====

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        if not email or not password:
            flash('Email and password are required.', 'error')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return redirect(url_for('register'))

        if password != confirm:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))

        conn = get_db()
        existing = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        if existing:
            conn.close()
            flash('An account with this email already exists.', 'error')
            return redirect(url_for('register'))

        password_hash = generate_password_hash(password)
        cursor = conn.execute(
            'INSERT INTO users (email, password_hash) VALUES (?, ?)',
            (email, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        user = load_user(user_id)
        login_user(user)
        flash('Account created successfully! Welcome aboard.', 'success')
        return redirect(url_for('dashboard'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Email and password are required.', 'error')
            return redirect(url_for('login'))

        conn = get_db()
        user_row = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        if not user_row or not check_password_hash(user_row['password_hash'], password):
            flash('Invalid email or password.', 'error')
            return redirect(url_for('login'))

        user = User(user_row)
        login_user(user)
        flash('Welcome back!', 'success')
        next_page = request.args.get('next')
        return redirect(next_page or url_for('dashboard'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing'))


# ===== Routes: Pages =====

@app.route('/')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html')


@app.route('/pricing')
def pricing():
    return render_template('landing.html', section='pricing')


# ===== Routes: Stripe (Mock Mode) =====

@app.route('/upgrade')
@login_required
def upgrade():
    """
    Mock Stripe Checkout.
    In production, this would redirect to a real Stripe Checkout session.
    """
    flash('Stripe integration is pending configuration. Payment processing will be available soon.', 'info')
    return redirect(url_for('pricing'))


@app.route('/webhook/stripe', methods=['POST'])
def stripe_webhook():
    """
    Mock Stripe webhook endpoint.
    In production, this would verify the Stripe signature and process events.
    """
    # Mock: accept a JSON body with { "user_email": "...", "action": "upgrade" }
    data = request.get_json(silent=True) or {}

    if data.get('action') == 'upgrade':
        email = data.get('user_email', '').strip().lower()
        if email:
            conn = get_db()
            conn.execute(
                'UPDATE users SET is_pro = 1 WHERE email = ?',
                (email,)
            )
            conn.commit()
            conn.close()
            return jsonify({'status': 'success', 'message': f'User {email} upgraded to Pro.'}), 200

    return jsonify({'status': 'ignored'}), 200


# ===== Routes: API (protected) =====

@app.route('/api/upload', methods=['POST'])
@login_required
def upload():
    if not current_user.can_use():
        if current_user.is_pro:
            return jsonify({'error': 'Something went wrong. Please contact support.'}), 403
        return jsonify({
            'error': f'You have reached your monthly limit of 3 uses. Upgrade to Pro for unlimited access.',
            'upgrade_url': url_for('upgrade')
        }), 403

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format. Please upload mp3, wav, or m4a files.'}), 400

    formats = request.form.getlist('formats')
    if not formats:
        return jsonify({'error': 'Please select at least one output format.'}), 400

    task_id = str(uuid.uuid4())

    upload_dir = os.path.join(app.instance_path, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{task_id}_{file.filename}")
    file.save(file_path)

    tasks[task_id] = {
        'status': 'uploaded',
        'progress': 10,
        'message': 'File uploaded. Starting transcription...',
        'results': None,
        'transcription': None,
    }

    thread = threading.Thread(
        target=process_task,
        args=(file_path, None, formats, task_id, current_user.id)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/api/text', methods=['POST'])
@login_required
def process_text():
    if not current_user.can_use():
        if current_user.is_pro:
            return jsonify({'error': 'Something went wrong. Please contact support.'}), 403
        return jsonify({
            'error': f'You have reached your monthly limit of 3 uses. Upgrade to Pro for unlimited access.',
            'upgrade_url': url_for('upgrade')
        }), 403

    data = request.get_json()
    text_content = data.get('text', '').strip()

    if not text_content:
        return jsonify({'error': 'Please enter some text content.'}), 400

    formats = data.get('formats', [])
    if not formats:
        return jsonify({'error': 'Please select at least one output format.'}), 400

    task_id = str(uuid.uuid4())

    tasks[task_id] = {
        'status': 'processing',
        'progress': 50,
        'message': 'Generating content...',
        'results': None,
        'transcription': None,
    }

    thread = threading.Thread(
        target=process_task,
        args=(None, text_content, formats, task_id, current_user.id)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/api/status/<task_id>')
@login_required
def task_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = tasks[task_id]
    response = {
        'status': task['status'],
        'progress': task['progress'],
        'message': task['message'],
    }

    if task.get('transcription'):
        response['transcription'] = task['transcription']

    if task['status'] == 'completed' and task.get('results'):
        response['results'] = task['results']

    return jsonify(response)


@app.route('/api/usage')
@login_required
def usage_info():
    """Return current user's usage information."""
    return jsonify({
        'is_pro': current_user.is_pro,
        'monthly_usage': current_user.get_monthly_usage(),
        'monthly_limit': None if current_user.is_pro else 3,
        'can_use': current_user.can_use(),
    })


# ===== Template Helpers =====

@app.context_processor
def inject_user():
    return {
        'current_user': current_user,
        'is_authenticated': current_user.is_authenticated,
    }


# ===== Error Handlers =====

@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('landing'))


if __name__ == '__main__':
    os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
