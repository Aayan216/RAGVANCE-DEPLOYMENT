"""Server-side Supabase Storage helper (REST, service-role key).

Every function is a no-op unless BOTH SUPABASE_URL and
SUPABASE_SERVICE_ROLE_KEY are present in the environment. Credentials are
read from the environment only, are never logged, and this module is never
imported by frontend code.

Object keys mirror Django's FileField relative names (e.g. "documents/x.pdf").
The documents bucket is private; access uses the service-role key with the
Authorization header on every request.
"""

import os
import tempfile
import urllib.parse

import requests

DEFAULT_DOCUMENTS_BUCKET = "ragvance-document"
DEFAULT_TIMEOUT = 20

_config_cache = None
_buckets_ready = set()


def _config():
    """Return (base_url, service_role_key) or None when not configured."""
    global _config_cache
    if _config_cache is None:
        url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        _config_cache = (url, key) if (url and key) else False
    return _config_cache or None


def reset_config_cache():
    """Testing hook: re-read environment after changing SUPABASE_* vars."""
    global _config_cache
    _config_cache = None
    _buckets_ready.clear()


def is_enabled():
    return _config() is not None


def _bucket():
    return (
        os.getenv("SUPABASE_STORAGE_BUCKET", "").strip()
        or DEFAULT_DOCUMENTS_BUCKET
    )


def _headers(content_type=None):
    _, key = _config()
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _quote_key(key):
    return "/".join(
        urllib.parse.quote(segment, safe="")
        for segment in key.split("/")
        if segment
    )


def ensure_bucket(bucket=None):
    """Create the private bucket if it does not exist yet (idempotent)."""
    cfg = _config()
    if not cfg:
        return False
    bucket = bucket or _bucket()
    if bucket in _buckets_ready:
        return True
    url, _ = cfg
    try:
        response = requests.post(
            f"{url}/storage/v1/bucket",
            headers=_headers("application/json"),
            json={"id": bucket, "name": bucket, "public": False},
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code in (200, 201, 409) or "exist" in (
            response.text or ""
        ).lower():
            _buckets_ready.add(bucket)
            return True
        print(f"[WARN] Supabase bucket check returned {response.status_code}")
        return False
    except Exception as exc:
        print(f"[WARN] Supabase bucket check failed: {type(exc).__name__}")
        return False


def upload_file(key, data, content_type="application/octet-stream", bucket=None):
    """Upload bytes to a private object (upsert). Raises on failure."""
    cfg = _config()
    if not cfg:
        raise RuntimeError("Supabase Storage is not configured")
    bucket = bucket or _bucket()
    ensure_bucket(bucket)
    url, _ = cfg
    response = requests.post(
        f"{url}/storage/v1/object/{bucket}/{_quote_key(key)}",
        headers={**_headers(content_type), "x-upsert": "true"},
        data=data,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return True


def download_file(key, bucket=None):
    """Download object bytes. Raises on failure."""
    cfg = _config()
    if not cfg:
        raise RuntimeError("Supabase Storage is not configured")
    bucket = bucket or _bucket()
    url, _ = cfg
    response = requests.get(
        f"{url}/storage/v1/object/{bucket}/{_quote_key(key)}",
        headers=_headers(),
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.content


def delete_file(key, bucket=None):
    """Delete an object. Returns False when it is already gone."""
    cfg = _config()
    if not cfg:
        raise RuntimeError("Supabase Storage is not configured")
    bucket = bucket or _bucket()
    url, _ = cfg
    response = requests.delete(
        f"{url}/storage/v1/object/{bucket}/{_quote_key(key)}",
        headers=_headers(),
        timeout=DEFAULT_TIMEOUT,
    )
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def materialize_file(relative_name, local_path):
    """Return (path_to_process, temp_path_or_None).

    Prefers the existing local file (development behavior is unchanged). If
    the local file is missing and Supabase Storage is configured, the object
    is downloaded to a temporary file; the caller must delete temp_path.
    """
    if os.path.exists(local_path):
        return local_path, None
    if not is_enabled():
        return local_path, None
    data = download_file(relative_name)
    suffix = os.path.splitext(relative_name)[1]
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return temp_path, temp_path
