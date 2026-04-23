# THIS IS CLAUDE CODE

import os
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
import nltk
from textblob import TextBlob
from newspaper import Article
from datetime import datetime, timedelta
from urllib.parse import urlparse
import validators
import requests

from sqlalchemy import func

# AUTH IMPORTS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()
nltk.download('punkt')

app = Flask(__name__)

# =========================
# CONFIG
# =========================

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback_dev_key_change_in_prod')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')

# API KEYS — store these in your .env file
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')          # https://newsapi.org (free tier)
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', '')          # https://huggingface.co/settings/tokens (free)

# Hugging Face model endpoint for summarization
HF_SUMMARIZATION_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# =========================
# DATABASE MODELS
# =========================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    preferred_category = db.Column(db.String(50), default='general')  # NEW: user preference

    summaries = db.relationship('Summary', backref='author', lazy=True)


class Summary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500))
    summary_text = db.Column(db.Text)
    sentiment = db.Column(db.String(50))
    category = db.Column(db.String(100))          # NEW: topic/category
    source_url = db.Column(db.String(1000))       # NEW: original article URL
    keywords = db.Column(db.String(500))          # NEW: extracted keywords
    reading_time_saved = db.Column(db.String(50)) # NEW: reading time saved
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# =========================
# HELPER FUNCTIONS
# =========================

def get_website_name(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def huggingface_summarize(text):
    """
    Calls Hugging Face Inference API (facebook/bart-large-cnn) to summarize text.
    Falls back to sentence-slicing if API call fails or token is missing.
    """
    # Truncate input — BART handles up to ~1024 tokens (~3000 chars is safe)
    truncated_text = text[:3000]

    if not HF_API_TOKEN:
        # Fallback: basic extractive summarization
        sentences = truncated_text.split('.')
        return '.'.join(sentences[:5]).strip() + '.'

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": truncated_text,
        "parameters": {
            "max_length": 180,
            "min_length": 60,
            "do_sample": False
        }
    }

    try:
        response = requests.post(HF_SUMMARIZATION_URL, headers=headers, json=payload, timeout=30)

        # Model may be loading on first call — handle gracefully
        if response.status_code == 503:
            return "Summarization model is loading. Please try again in ~20 seconds."

        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) > 0:
            return result[0].get('summary_text', 'Summary not available.')
        return 'Summary not available.'

    except requests.RequestException as e:
        # Fallback to sentence slicing on any API failure
        sentences = truncated_text.split('.')
        return '.'.join(sentences[:5]).strip() + '.'


def calculate_reading_time(original_text, summary_text):
    """
    Estimates reading time at 200 words per minute.
    Returns a human-readable string like 'Saved ~6 min'.
    """
    original_words = len(original_text.split())
    summary_words = len(summary_text.split())
    original_mins = round(original_words / 200)
    summary_mins = max(1, round(summary_words / 200))
    saved = max(0, original_mins - summary_mins)
    return f"~{saved} min saved (orig: ~{original_mins} min)"


def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'happy 😊'
    elif polarity < 0:
        return 'sad 😟'
    else:
        return 'neutral 😐'


# =========================
# AUTH ROUTES
# =========================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = generate_password_hash(request.form.get('password'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists!')
            return redirect(url_for('register'))

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully! Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))

        flash('Invalid username or password.')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.')
    return redirect(url_for('login'))


# =========================
# LIVE NEWS FEED (NEW)
# =========================

NEWS_CATEGORIES = ['general', 'technology', 'business', 'sports', 'science', 'health', 'entertainment']

@app.route('/live-news')
@login_required
def live_news():
    """
    Fetches top headlines from NewsAPI and displays them as cards.
    Users can click 'Summarize' on any article to process it directly.
    """
    category = request.args.get('category', current_user.preferred_category or 'general')

    if not NEWS_API_KEY:
        flash('NewsAPI key not configured. Add NEWS_API_KEY to your .env file.')
        return render_template('live_news.html', articles=[], categories=NEWS_CATEGORIES, selected_category=category)

    try:
        api_url = "https://newsapi.org/v2/top-headlines"
        params = {
            'apiKey': NEWS_API_KEY,
            'category': category,
            'country': 'us',
            'pageSize': 20
        }
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])

        # Filter out articles with [Removed] content (NewsAPI free tier quirk)
        articles = [a for a in articles if a.get('title') and a['title'] != '[Removed]']

    except requests.RequestException:
        flash('Failed to fetch live news. Please try again later.')
        articles = []

    return render_template(
        'live_news.html',
        articles=articles,
        categories=NEWS_CATEGORIES,
        selected_category=category
    )


@app.route('/update-preference', methods=['POST'])
@login_required
def update_preference():
    """Saves the user's preferred news category."""
    category = request.form.get('category', 'general')
    if category in NEWS_CATEGORIES:
        current_user.preferred_category = category
        db.session.commit()
        flash(f'News preference updated to: {category.capitalize()}')
    return redirect(url_for('live_news', category=category))


# =========================
# HISTORY ROUTE
# =========================

@app.route('/history')
@login_required
def history():
    category_filter = request.args.get('category', None)

    query = Summary.query.filter_by(user_id=current_user.id)
    if category_filter:
        query = query.filter_by(category=category_filter)

    user_summaries = query.order_by(Summary.date_created.desc()).all()

    # Get distinct categories for the filter dropdown
    categories = db.session.query(Summary.category).filter_by(
        user_id=current_user.id
    ).distinct().all()
    categories = [c[0] for c in categories if c[0]]

    return render_template(
        'history.html',
        summaries=user_summaries,
        categories=categories,
        selected_category=category_filter
    )


# =========================
# DASHBOARD
# =========================

@app.route('/dashboard')
@login_required
def dashboard():
    total_articles = Summary.query.filter_by(user_id=current_user.id).count()

    sentiment_data = db.session.query(
        Summary.sentiment,
        func.count(Summary.sentiment)
    ).filter(Summary.user_id == current_user.id).group_by(Summary.sentiment).all()

    sentiment_labels = [s for s, _ in sentiment_data]
    sentiment_counts = [c for _, c in sentiment_data]

    last_7_days = []
    article_counts = []
    for i in range(6, -1, -1):
        day = datetime.utcnow().date() - timedelta(days=i)
        count = Summary.query.filter(
            Summary.user_id == current_user.id,
            func.date(Summary.date_created) == day
        ).count()
        last_7_days.append(day.strftime("%d %b"))
        article_counts.append(count)

    # NEW: Category breakdown for dashboard chart
    category_data = db.session.query(
        Summary.category,
        func.count(Summary.category)
    ).filter(Summary.user_id == current_user.id).group_by(Summary.category).all()

    category_labels = [c for c, _ in category_data if c]
    category_counts = [n for _, n in category_data if _]

    return render_template(
        'dashboard.html',
        total_articles=total_articles,
        sentiment_labels=sentiment_labels,
        sentiment_counts=sentiment_counts,
        last_7_days=last_7_days,
        article_counts=article_counts,
        category_labels=category_labels,      # NEW
        category_counts=category_counts       # NEW
    )


# =========================
# MAIN SUMMARIZER
# =========================

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        url = request.form['url']
        category = request.form.get('category', 'general')  # optional: let user tag category

        if not validators.url(url):
            flash('Please enter a valid URL.')
            return redirect(url_for('index'))

        # Parse article
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
        except Exception:
            flash('Failed to parse the article. The site may be blocking scrapers.')
            return redirect(url_for('index'))

        article_text = article.text
        if not article_text or len(article_text.strip()) < 100:
            flash('Could not extract enough content from this URL. Try a different article.')
            return redirect(url_for('index'))

        title = article.title
        authors = ', '.join(article.authors) or get_website_name(url)
        publish_date = article.publish_date.strftime('%B %d, %Y') if article.publish_date else "N/A"
        top_image = article.top_image

        # --- Hugging Face AI Summarization ---
        summary = huggingface_summarize(article_text)

        # --- Sentiment Analysis ---
        sentiment = get_sentiment(article_text)

        # --- Keyword Extraction (newspaper3k already does this) ---
        keywords = ', '.join(article.keywords[:8]) if article.keywords else 'N/A'

        # --- Reading Time ---
        reading_time_saved = calculate_reading_time(article_text, summary)

        # Save to DB
        new_summary = Summary(
            title=title,
            summary_text=summary,
            sentiment=sentiment,
            category=category,
            source_url=url,
            keywords=keywords,
            reading_time_saved=reading_time_saved,
            user_id=current_user.id
        )
        db.session.add(new_summary)
        db.session.commit()

        return render_template(
            'index.html',
            title=title,
            authors=authors,
            publish_date=publish_date,
            summary=summary,
            top_image=top_image,
            sentiment=sentiment,
            keywords=keywords,              # NEW
            reading_time_saved=reading_time_saved,  # NEW
            source_url=url                  # NEW
        )

    return render_template('index.html', categories=NEWS_CATEGORIES)


# =========================
# API ENDPOINT (bonus — for future mobile app)
# =========================

@app.route('/api/summarize', methods=['POST'])
@login_required
def api_summarize():
    """
    JSON API endpoint. POST { "url": "https://..." }
    Returns summary, sentiment, keywords as JSON.
    Useful if you later build a mobile app or browser extension on top.
    """
    data = request.get_json()
    url = data.get('url', '')

    if not validators.url(url):
        return jsonify({'error': 'Invalid URL'}), 400

    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
    except Exception as e:
        return jsonify({'error': 'Failed to parse article', 'detail': str(e)}), 500

    summary = huggingface_summarize(article.text)
    sentiment = get_sentiment(article.text)
    keywords = article.keywords[:8] if article.keywords else []

    return jsonify({
        'title': article.title,
        'summary': summary,
        'sentiment': sentiment,
        'keywords': keywords,
        'reading_time_saved': calculate_reading_time(article.text, summary)
    })


# =========================
# RUN APP
# =========================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)









# THIS IS THE WORKING CODE

# from flask import Flask, request, render_template, flash, redirect, url_for
# import nltk
# from textblob import TextBlob
# from newspaper import Article
# from datetime import datetime
# from urllib.parse import urlparse
# import validators
# import requests

# from sqlalchemy import func
# from datetime import timedelta

# # AUTH IMPORTS
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
# from werkzeug.security import generate_password_hash, check_password_hash

# nltk.download('punkt')

# app = Flask(__name__)

# # CONFIG
# app.config['SECRET_KEY'] = 'your_secret_key'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

# db = SQLAlchemy(app)

# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'


# # =========================
# # DATABASE MODELS
# # =========================

# class User(UserMixin, db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(100), unique=True, nullable=False)
#     password = db.Column(db.String(200), nullable=False)

#     summaries = db.relationship('Summary', backref='author', lazy=True)


# class Summary(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(500))
#     summary_text = db.Column(db.Text)
#     sentiment = db.Column(db.String(50))
#     date_created = db.Column(db.DateTime, default=datetime.utcnow)

#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))


# # =========================
# # HELPER FUNCTION
# # =========================

# def get_website_name(url):
#     parsed_url = urlparse(url)
#     domain = parsed_url.netloc
#     if domain.startswith("www."):
#         domain = domain[4:]
#     return domain


# # =========================
# # AUTH ROUTES
# # =========================

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = generate_password_hash(request.form.get('password'))

#         if User.query.filter_by(username=username).first():
#             flash('Username already exists!')
#             return redirect(url_for('register'))

#         new_user = User(username=username, password=password)
#         db.session.add(new_user)
#         db.session.commit()

#         flash('Account created successfully! Please login.')
#         return redirect(url_for('login'))

#     return render_template('register.html')




# @app.route('/login', methods=['GET', 'POST'])
# def login():

#     if current_user.is_authenticated:
#         return redirect(url_for('index'))

#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')

#         user = User.query.filter_by(username=username).first()

#         if user and check_password_hash(user.password, password):
#             login_user(user)
#             return redirect(url_for('index'))

#         flash('Invalid username or password.')

#     return render_template('login.html')


# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     flash('Logged out successfully.')
#     return redirect(url_for('login'))


# # =========================
# # HISTORY ROUTE
# # =========================

# @app.route('/history')
# @login_required
# def history():
#     user_summaries = Summary.query.filter_by(
#         user_id=current_user.id
#     ).order_by(Summary.date_created.desc()).all()

#     return render_template('history.html', summaries=user_summaries)



# @app.route('/dashboard')
# @login_required
# def dashboard():

#     total_articles = Summary.query.filter_by(
#         user_id=current_user.id
#     ).count()

#     sentiment_data = db.session.query(
#         Summary.sentiment,
#         func.count(Summary.sentiment)
#     ).filter(
#         Summary.user_id == current_user.id
#     ).group_by(Summary.sentiment).all()

#     sentiment_labels = []
#     sentiment_counts = []

#     for sentiment, count in sentiment_data:
#         sentiment_labels.append(sentiment)
#         sentiment_counts.append(count)

#     last_7_days = []
#     article_counts = []

#     for i in range(6, -1, -1):
#         day = datetime.utcnow().date() - timedelta(days=i)
#         count = Summary.query.filter(
#             Summary.user_id == current_user.id,
#             func.date(Summary.date_created) == day
#         ).count()

#         last_7_days.append(day.strftime("%d %b"))
#         article_counts.append(count)

#     return render_template(
#         'dashboard.html',
#         total_articles=total_articles,
#         sentiment_labels=sentiment_labels,
#         sentiment_counts=sentiment_counts,
#         last_7_days=last_7_days,
#         article_counts=article_counts
#     )


# # =========================
# # MAIN SUMMARIZER
# # =========================

# @app.route('/', methods=['GET', 'POST'])
# @login_required
# def index():

#     if request.method == 'POST':
#         url = request.form['url']

#         if not validators.url(url):
#             flash('Please enter a valid URL.')
#             return redirect(url_for('index'))

#         try:
#             response = requests.get(url)
#             response.raise_for_status()
#         except requests.RequestException:
#             flash('Failed to download the content of the URL.')
#             return redirect(url_for('index'))

#         article = Article(url)
#         article.download()
#         article.parse()
#         article.nlp()

#         title = article.title
#         authors = ', '.join(article.authors)
#         if not authors:
#             authors = get_website_name(url)

#         publish_date = article.publish_date.strftime('%B %d, %Y') if article.publish_date else "N/A"

#         article_text = article.text
#         sentences = article_text.split('.')
#         max_summarized_sentences = 5
#         summary = '.'.join(sentences[:max_summarized_sentences])

#         top_image = article.top_image

#         analysis = TextBlob(article.text)
#         polarity = analysis.sentiment.polarity

#         if polarity > 0:
#             sentiment = 'happy 😊'
#         elif polarity < 0:
#             sentiment = 'sad 😟'
#         else:
#             sentiment = 'neutral 😐'

#         # SAVE TO DATABASE
#         new_summary = Summary(
#             title=title,
#             summary_text=summary,
#             sentiment=sentiment,
#             user_id=current_user.id
#         )

#         db.session.add(new_summary)
#         db.session.commit()

#         return render_template(
#             'index.html',
#             title=title,
#             authors=authors,
#             publish_date=publish_date,
#             summary=summary,
#             top_image=top_image,
#             sentiment=sentiment
#         )

#     return render_template('index.html')


# # =========================
# # RUN APP
# # =========================

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True)



# from flask import Flask, request, render_template, flash, redirect, url_for  # Import flash
# import nltk
# from textblob import TextBlob
# from newspaper import Article
# from datetime import datetime
# from urllib.parse import urlparse
# import validators
# import requests

# nltk.download('punkt')

# app = Flask(__name__)

# def get_website_name(url):
#     # Extract the website name from the URL
#     parsed_url = urlparse(url)
#     domain = parsed_url.netloc
#     if domain.startswith("www."):
#         domain = domain[4:]
#     return domain

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         url = request.form['url']
#         # Check if the input is a valid URL

#         if not validators.url(url):
#             flash('Please enter a valid URL.')
#             return redirect(url_for('index'))
        
#         try:
#             response = requests.get(url)
#             response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
#         except requests.RequestException:
#             flash('Failed to download the content of the URL.')
#             return redirect(url_for('index'))
        
#         article = Article(url)
#         article.download()
#         article.parse()
#         article.nlp()  # Perform natural language processing

#         title = article.title
#         authors = ', '.join(article.authors)
#         if not authors:
#             authors = get_website_name(url)  # Set the author field to the website name
#         publish_date = article.publish_date.strftime('%B %d, %Y') if article.publish_date else "N/A"

#         # Manually adjust the summary length by selecting a certain number of sentences
#         article_text = article.text
#         sentences = article_text.split('.')
#         max_summarized_sentences = 5  # Adjust the number of sentences as needed
#         summary = '.'.join(sentences[:max_summarized_sentences])

#         top_image = article.top_image  # Get the top image URL

#         analysis = TextBlob(article.text)
#         polarity = analysis.sentiment.polarity  # Get the polarity value

#         if summary == "":
#             flash('Please enter a valid URL.')
#             return redirect(url_for('index'))

#         if polarity > 0:
#             sentiment = 'happy 😊'
#         elif polarity < 0:
#             sentiment = ' sad 😟'
#         else:
#             sentiment = 'neutral 😐'

#         return render_template('index.html', title=title, authors=authors, publish_date=publish_date, summary=summary, top_image=top_image, sentiment=sentiment)

#     return render_template('index.html')

# app.secret_key = 'your_secret_key'

# if __name__ == '__main__':
#     app.run(debug=True)