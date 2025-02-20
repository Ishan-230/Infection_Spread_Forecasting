import os
import dj_database_url
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env (optional for local dev)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# SECURITY: Use environment variable for SECRET_KEY
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-xvz$8)=5k7087qxmq9sc70@=va0f(dyzbpic33-6quc*j02b')  # Replace with actual secret in production

# SECURITY: Disable debug in production
DEBUG = os.getenv('DEBUG', 'False') == 'True'

# SECURITY: Allow only Render's domain
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost').split(',')
ALLOWED_HOSTS = ["https://predictive-analytics-mbga.onrender.com/", "localhost", "127.0.0.1"]

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'app2',
    'whitenoise.runserver_nostatic',  # Whitenoise for static file handling
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # Serve static files efficiently
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'project2.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'project2.wsgi.application'


# Database Configuration (Use PostgreSQL on Render)
DATABASES = {
    'default': dj_database_url.config(default=os.getenv('DATABASE_URL', 'sqlite:///db.sqlite3'))
}


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


# Static Files (For Render Deployment)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
