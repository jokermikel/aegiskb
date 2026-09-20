import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException, status
from .config import get_settings

def hash_password(password: str) -> str:
    """PBKDF2-HMAC for this self-contained demo; use Argon2id via enterprise IdP in production."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600_000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, salt, expected = password_hash.split("$", 2)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600_000).hex()
        return hmac.compare_digest(actual, expected)
    except ValueError:
        return False


def create_access_token(user_id: int) -> str:
    payload = {"sub": str(user_id), "exp": datetime.now(timezone.utc) + timedelta(hours=8)}
    return jwt.encode(payload, get_settings().jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> int:
    try:
        return int(jwt.decode(token, get_settings().jwt_secret, algorithms=["HS256"])["sub"])
    except (jwt.PyJWTError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效或过期的访问令牌") from exc
