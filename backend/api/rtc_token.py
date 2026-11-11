"""
火山引擎 RTC Token 生成器

基于火山引擎官方 Token 算法实现（从官方 TypeScript 源码移植）。
Token = "001" + appId + base64(message + signature)
message = nonce(u32) + issuedAt(u32) + expireAt(u32) + roomId + userId + privileges
signature = HMAC-SHA256(appKey, message)

参考: https://www.volcengine.com/docs/6348/70121
"""

import hmac
import hashlib
import struct
import time
import random
import base64
from io import BytesIO


# 权限常量（与官方一致）
PRIV_PUBLISH_STREAM = 0        # 发布流（会自动包含audio+video+data）
PRIV_PUBLISH_AUDIO_STREAM = 1  # 发布音频流
PRIV_PUBLISH_VIDEO_STREAM = 2  # 发布视频流
PRIV_PUBLISH_DATA_STREAM = 3   # 发布数据流
PRIV_SUBSCRIBE_STREAM = 4      # 订阅流

VERSION = "001"
APP_ID_LENGTH = 24


class _ByteBuf:
    """字节缓冲区（小端序）"""

    def __init__(self):
        self._buf = BytesIO()

    def put_uint16(self, value: int):
        self._buf.write(struct.pack('<H', value))  # 小端

    def put_uint32(self, value: int):
        self._buf.write(struct.pack('<I', value))  # 小端

    def put_bytes(self, data: bytes):
        self.put_uint16(len(data))
        self._buf.write(data)

    def put_string(self, s: str):
        self.put_bytes(s.encode('utf-8'))

    def put_tree_map_uint32(self, m: dict):
        """写入 privilege map: count(u16) + [key(u16) + value(u32)]..."""
        self.put_uint16(len(m))
        for key in sorted(m.keys()):
            self.put_uint16(key)
            self.put_uint32(m[key])

    def pack(self) -> bytes:
        return self._buf.getvalue()


class AccessToken:
    """火山引擎 RTC Token 生成器"""

    def __init__(self, app_id: str, app_key: str, room_id: str, user_id: str):
        self.app_id = app_id
        self.app_key = app_key
        self.room_id = room_id
        self.user_id = user_id
        self.privileges: dict[int, int] = {}
        self.issued_at = int(time.time())
        self.nonce = random.randint(0, 0xFFFFFFFF)
        self.expire_at = 0

    def add_privilege(self, privilege: int, expire_timestamp: int = 0):
        """添加权限。expire_timestamp 为 Unix 时间戳，0 表示不限。"""
        self.privileges[privilege] = expire_timestamp

        # PrivPublishStream 自动包含 audio + video + data
        if privilege == PRIV_PUBLISH_STREAM:
            self.privileges[PRIV_PUBLISH_VIDEO_STREAM] = expire_timestamp
            self.privileges[PRIV_PUBLISH_AUDIO_STREAM] = expire_timestamp
            self.privileges[PRIV_PUBLISH_DATA_STREAM] = expire_timestamp

    def expire_time(self, expire_timestamp: int):
        """设置 Token 过期时间（Unix 时间戳）"""
        self.expire_at = expire_timestamp

    def _pack_message(self) -> bytes:
        """构建 message 字节流"""
        buf = _ByteBuf()
        buf.put_uint32(self.nonce)
        buf.put_uint32(self.issued_at)
        buf.put_uint32(self.expire_at)
        buf.put_string(self.room_id)
        buf.put_string(self.user_id)
        buf.put_tree_map_uint32(self.privileges)
        return buf.pack()

    def serialize(self) -> str:
        """生成最终的 Token 字符串

        官方C++格式: content = Pack(message) + Pack(signature)
        Pack(string) = u16(length) + data
        所以 content = u16(msg_len) + msg_data + u16(sig_len) + sig_data
        """
        message = self._pack_message()

        # signature = HMAC-SHA256(appKey, message)
        signature = hmac.new(
            self.app_key.encode('utf-8'),
            message,
            hashlib.sha256,
        ).digest()

        # content = Pack(message) + Pack(signature)
        # Pack(string) = u16(length) + data
        msg_packed = struct.pack('<H', len(message)) + message
        sig_packed = struct.pack('<H', len(signature)) + signature
        content = msg_packed + sig_packed

        content_b64 = base64.b64encode(content).decode('utf-8')

        # Token = VERSION + appId + base64(content)
        return VERSION + self.app_id + content_b64


def generate_rtc_token(
    app_id: str,
    app_key: str,
    room_id: str,
    user_id: str,
    expire_seconds: int = 86400,
    role: int = 1,
) -> str:
    """
    生成 RTC 鉴权 Token

    Args:
        app_id: RTC 应用 ID
        app_key: RTC 应用 Key
        room_id: 房间 ID
        user_id: 用户 ID
        expire_seconds: Token 有效期（秒），默认 24 小时
        role: 1=主播(可推拉流), 0=观众(仅拉流)

    Returns:
        Token 字符串
    """
    token = AccessToken(app_id, app_key, room_id, user_id)

    # 设置过期时间
    expire_ts = int(time.time()) + expire_seconds
    token.expire_time(expire_ts)

    # 添加权限
    token.add_privilege(PRIV_PUBLISH_STREAM, expire_ts)
    token.add_privilege(PRIV_SUBSCRIBE_STREAM, expire_ts)

    return token.serialize()
