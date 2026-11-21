"""
Django settings for the replay viewer.

Based on the authors' frontend_server settings (Django 2.2), updated for
current Django. Only what the replay viewer needs is kept: no database, no
admin, no S3 storage, no CORS. The authors' art assets are served straight
from the downloaded copy in external/ (they are not part of this repo).
"""
import os
import secrets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

AUTHORS_STATIC = os.environ.get(
  "GA_AUTHORS_STATIC",
  os.path.join(PROJECT_ROOT, "external", "generative_agents", "environment",
               "frontend_server", "static_dirs"))
REPLAY_DIR = os.environ.get("GA_REPLAY_DIR", os.path.join(PROJECT_ROOT, "runs", "replays"))
STORAGE_DIR = os.environ.get("GA_STORAGE", os.path.join(PROJECT_ROOT, "runs"))

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", secrets.token_urlsafe(50))
DEBUG = True
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

INSTALLED_APPS = [
  "django.contrib.staticfiles",
  "translator",
]

MIDDLEWARE = [
  "django.middleware.security.SecurityMiddleware",
  "django.middleware.common.CommonMiddleware",
  "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "frontend_server.urls"

TEMPLATES = [
  {
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [os.path.join(BASE_DIR, "templates")],
    "APP_DIRS": True,
    "OPTIONS": {
      "context_processors": [
        "django.template.context_processors.debug",
        "django.template.context_processors.request",
      ],
    },
  },
]

WSGI_APPLICATION = "frontend_server.wsgi.application"
DATABASES = {}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [AUTHORS_STATIC]
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
