from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-secret-key")

DEBUG = os.getenv("DEBUG", "True").lower() == "true"

ALLOWED_HOSTS = [x.strip() for x in os.getenv("ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",") if x.strip()]
CSRF_TRUSTED_ORIGINS = [x.strip() for x in os.getenv("CSRF_TRUSTED_ORIGINS", "").split(",") if x.strip()]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "backend.apps.BackendConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "frontend" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.template.context_processors.static",
                "django.template.context_processors.media",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL:
    import dj_database_url

    DATABASES = {
        "default": dj_database_url.parse(DATABASE_URL, conn_max_age=600)
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "frontend" / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage",
    },
}

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "data" / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# RAG Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# "torch" = sentence-transformers (local/dev/tests, ~450MB RSS).
# "onnx"  = build-exported ONNX model via onnxruntime (Render free, 512MB limit).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "torch")
EMBEDDING_ONNX_PATH = os.getenv("EMBEDDING_ONNX_PATH", "")
GEMINI_MODEL = "gemini-3.5-flash-lite"
TOP_K_RETRIEVAL = 5
VECTOR_STORE_PATH = BASE_DIR / "data" / "vector_store"
MOCK_TEST_BATCH_SIZE = 3
MOCK_TEST_QUESTIONS_PER_LLM_CALL = 5
PRACTICE_BATCH_SIZE = 3
PRACTICE_QUESTIONS_PER_LLM_CALL = 5
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_RETRY_BASE_DELAY = float(os.getenv("GEMINI_RETRY_BASE_DELAY", "1.0"))

# Document ingestion safety limits (validated at upload / process time).
# Overridable via environment variables; NOT user-editable settings.
#
# MAX_UPLOAD_FILE_SIZE_MB: bounds parse/clean/embed memory and wall time for a
# single upload. 25 MB of extracted text is far above any legitimate study
# document while keeping worst-case process RSS predictable on the 512MB
# Render free instance.
MAX_UPLOAD_FILE_SIZE_MB = int(os.getenv("MAX_UPLOAD_FILE_SIZE_MB", "25"))
# MAX_DOCUMENTS: bounds total stored files + total FAISS vectors across the
# instance. Counting is based on actual Document records only (failed uploads
# never consume a slot; nothing is auto-deleted at the limit - new uploads are
# simply rejected).
MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "20"))
# MAX_CHUNKS_PER_DOCUMENT: bounds the O(n^2) near-duplicate pass and the embed
# step so one pathological document cannot blow the gunicorn request budget
# (--timeout 180) or the vector store. 5000 confirmed by
# tests/benchmark_ingest.py: dedup(n=5000)=33s, embed ~80 chunks/s -> ~63s,
# chunker 0.2s, bulk insert 0.2s => est. 111s of the 180s budget.
MAX_CHUNKS_PER_DOCUMENT = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "5000"))