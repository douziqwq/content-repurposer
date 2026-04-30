import os
import json
import time
import uuid
import random
import threading
import requests
import resend
import stripe
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
# ===== Database Setup (PostgreSQL) =====
import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ.get('DATABASE_URL', '')

def get_db():
    """Get a database connection."""
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def init_db():
    """Initialize the database tables."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_pro BOOLEAN DEFAULT FALSE,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS usage_logs (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            action TEXT NOT NULL DEFAULT 'repurpose',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_usage_user_date ON usage_logs(user_id, created_at);
    ''')
    conn.commit()
    cur.close()
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
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            '''SELECT COUNT(*) as count FROM usage_logs
               WHERE user_id = %s AND action = 'repurpose'
               AND TO_CHAR(created_at, 'YYYY-MM') = %s''',
            (self.id, now.strftime('%Y-%m'))
        )
        row = cur.fetchone()
        cur.close()
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
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO usage_logs (user_id, action) VALUES (%s, %s)',
            (self.id, 'repurpose')
        )
        conn.commit()
        cur.close()
        conn.close()

@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute('SELECT * FROM users WHERE id = %s', (int(user_id),))
    user_row = cur.fetchone()
    cur.close()
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

# ===== Email Configuration (Resend) =====
RESEND_API_KEY = os.environ.get('RESEND_API_KEY', '')
resend.api_key = RESEND_API_KEY
FROM_EMAIL = "ContentRepurposer <onboarding@resend.dev>"

# ===== Stripe Configuration =====
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
stripe.api_key = STRIPE_SECRET_KEY

# Pro Plan price ID (will be set after creating the product in Stripe)
# For now we use a hardcoded test price - in production this should be from env
STRIPE_PRICE_ID = os.environ.get('STRIPE_PRICE_ID', 'price_1TRrYT3Mr56JhSrBmKjE3YqP')

# In-memory verification code storage: {email: {'code': '123456', 'expires': timestamp}}
verification_codes = {}

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
            cur = conn.cursor()
            cur.execute(
                'INSERT INTO usage_logs (user_id, action) VALUES (%s, %s)',
                (user_id, 'repurpose')
            )
            conn.commit()
            cur.close()
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

@app.route('/api/send-code', methods=['POST'])
def send_verification_code():
    """Send a 6-digit verification code to the user's email."""
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    
    if not email:
        return jsonify({'error': 'Email is required.'}), 400
    
    # Check if email already registered
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute('SELECT id FROM users WHERE email = %s', (email,))
    existing = cur.fetchone()
    cur.close()
    conn.close()
    if existing:
        return jsonify({'error': 'An account with this email already exists.'}), 400
    
    # Rate limit: prevent sending too many codes
    if email in verification_codes:
        last_sent = verification_codes[email].get('last_sent', 0)
        if time.time() - last_sent < 60:
            return jsonify({'error': 'Please wait 60 seconds before requesting a new code.'}), 429
    
    # Generate 6-digit code
    code = str(random.randint(100000, 999999))
    verification_codes[email] = {
        'code': code,
        'expires': time.time() + 600,  # 10 minutes
        'last_sent': time.time()
    }
    
    # Send email via Resend
    try:
        params = {
            "from": FROM_EMAIL,
            "to": [email],
            "subject": "Your ContentRepurposer Verification Code",
            "html": f"""
            <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #3b82f6;">ContentRepurposer</h2>
                <p>Hi there!</p>
                <p>Your verification code is:</p>
                <div style="background: #f3f4f6; padding: 16px; border-radius: 8px; text-align: center; margin: 20px 0;">
                    <span style="font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #1f2937;">{code}</span>
                </div>
                <p>This code expires in 10 minutes. If you didn't request this, you can safely ignore this email.</p>
                <p>Best regards,<br>The ContentRepurposer Team</p>
            </div>
            """
        }
        resend.Emails.send(params)
        return jsonify({'success': True, 'message': 'Verification code sent!'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to send verification email. Please try again.'}), 500


@app.route('/api/verify-code', methods=['POST'])
def verify_code():
    """Verify the 6-digit code entered by the user."""
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    code = data.get('code', '').strip()
    
    if not email or not code:
        return jsonify({'error': 'Email and code are required.'}), 400
    
    if email not in verification_codes:
        return jsonify({'error': 'No verification code found. Please request a new one.'}), 400
    
    stored = verification_codes[email]
    
    if time.time() > stored['expires']:
        del verification_codes[email]
        return jsonify({'error': 'Verification code has expired. Please request a new one.'}), 400
    
    if stored['code'] != code:
        return jsonify({'error': 'Invalid verification code.'}), 400
    
    # Mark as verified
    verification_codes[email]['verified'] = True
    return jsonify({'success': True, 'message': 'Email verified!'}), 200


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

        # Check email verification
        if email not in verification_codes or not verification_codes[email].get('verified'):
            flash('Please verify your email first.', 'error')
            return redirect(url_for('register'))

        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('SELECT id FROM users WHERE email = %s', (email,))
        existing = cur.fetchone()
        if existing:
            cur.close()
            conn.close()
            flash('An account with this email already exists.', 'error')
            return redirect(url_for('register'))

        password_hash = generate_password_hash(password)
        cur.execute(
            'INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id',
            (email, password_hash)
        )
        user_id = cur.fetchone()['id']
        conn.commit()
        cur.close()
        conn.close()

        # Clean up verification code
        if email in verification_codes:
            del verification_codes[email]

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
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('SELECT * FROM users WHERE email = %s', (email,))
        user_row = cur.fetchone()
        cur.close()
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


# ===== Routes: Stripe =====

@app.route('/upgrade')
@login_required
def upgrade():
    """Create a Stripe Checkout session for Pro upgrade."""
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        flash('Payment system is being configured. Please try again later.', 'info')
        return redirect(url_for('pricing'))

    try:
        # Get or create Stripe customer
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute('SELECT * FROM users WHERE id = %s', (current_user.id,))
        user_row = cur.fetchone()
        cur.close()
        conn.close()

        customer_id = user_row['stripe_customer_id']
        if not customer_id:
            customer = stripe.Customer.create(
                email=current_user.email,
                metadata={'user_id': current_user.id}
            )
            customer_id = customer.id
            conn = get_db()
            cur = conn.cursor()
            cur.execute('UPDATE users SET stripe_customer_id = %s WHERE id = %s', (customer_id, current_user.id))
            conn.commit()
            cur.close()
            conn.close()

        # Create Checkout session
        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode='subscription',
            line_items=[{'price': STRIPE_PRICE_ID, 'quantity': 1}],
            success_url=url_for('upgrade_success', _external=True),
            cancel_url=url_for('pricing', _external=True),
            subscription_data={
                'metadata': {'user_id': current_user.id}
            }
        )
        return redirect(session.url)
    except stripe.error.StripeError as e:
        flash(f'Payment error: {str(e)}', 'error')
        return redirect(url_for('pricing'))


@app.route('/upgrade/success')
@login_required
def upgrade_success():
    """Handle successful payment return."""
    flash('Welcome to Pro! You now have unlimited access.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/webhook/stripe', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events."""
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature', '')
    webhook_secret = os.environ.get('STRIPE_WEBHOOK_SECRET', '')

    if webhook_secret:
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        except (stripe.error.SignatureVerificationError, ValueError):
            return jsonify({'error': 'Invalid signature'}), 400
    else:
        event = json.loads(payload)

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_id = session.get('metadata', {}).get('user_id')
        customer_id = session.get('customer')

        if user_id:
            conn = get_db()
            cur = conn.cursor()
            cur.execute('UPDATE users SET is_pro = TRUE, stripe_customer_id = %s WHERE id = %s', (customer_id, int(user_id)))
            conn.commit()
            cur.close()
            conn.close()

    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        customer_id = subscription.get('customer')

        if customer_id:
            conn = get_db()
            cur = conn.cursor()
            cur.execute('UPDATE users SET is_pro = FALSE WHERE stripe_customer_id = %s', (customer_id,))
            conn.commit()
            cur.close()
            conn.close()

    return jsonify({'status': 'success'}), 200


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
        'stripe_publishable_key': STRIPE_PUBLISHABLE_KEY,
    }


# ===== Error Handlers =====

@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('landing'))


if __name__ == '__main__':
    os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
