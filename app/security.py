import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException, status

from .config import get_settings


def hash_password(password: str) -> str:
    """用 PBKDF2-HMAC 生成演示账户的不可逆密码哈希。"""
    salt = secrets.token_hex(16)  # 每个用户独立随机盐，防止彩虹表攻击。
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600_000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password: str,password_hash: str) -> bool:
    """验证输入密码；compare_digest 防止普通字符串比较的时序泄漏。"""
    try:
        algorithm, salt, expected = password_hash.split("$", 2)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600_000).hex()
        return hmac.compare_digest(actual, expected)
    except ValueError:
        return False


def create_access_token(user_id: int) -> str:
    """生成 8 小时有效期的 JWT；sub 是用户主键，不把权限直接信任在 token 中。"""
    payload = {"sub": str(user_id), "exp": datetime.now(timezone.utc) + timedelta(hours=8)}
    return jwt.encode(payload, get_settings().jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> int:
    """验签、验过期时间并取回用户 ID；失败统一返回 401。"""
    try:
        return int(jwt.decode(token, get_settings().jwt_secret, algorithms=["HS256"])["sub"])
    except (jwt.PyJWTError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或过期的访问令牌",
        ) from exc