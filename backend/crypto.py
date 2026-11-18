"""Template encryption at rest (Fernet = AES-128-CBC + HMAC-SHA256). The server must decrypt to match, so this protects the
database contents (backups, leaked connection strings, Atlas staff), not the running server.
Without TEMPLATE_ENCRYPTION_KEY templates are stored unencrypted (prefix p1:) and /api/health reports it."""
import base64

import numpy as np


class CryptoError(Exception):
    pass


class TemplateCipher:
    def __init__(self, key=""):
        self._fernet = None
        if key:
            try:
                from cryptography.fernet import Fernet
                self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
            except Exception as exc:  # noqa: BLE001
                raise CryptoError("TEMPLATE_ENCRYPTION_KEY is invalid (generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\")") from exc

    @property
    def enabled(self):
        return self._fernet is not None

    def encrypt(self, vec):
        raw = np.asarray(vec, dtype="<f4").tobytes()
        if self._fernet:
            return "f1:" + self._fernet.encrypt(raw).decode("ascii")
        return "p1:" + base64.b64encode(raw).decode("ascii")

    def decrypt(self, token, dim):
        try:
            kind, _, body = token.partition(":")
            if kind == "f1":
                if not self._fernet:
                    raise CryptoError("templates are encrypted but TEMPLATE_ENCRYPTION_KEY is not set")
                raw = self._fernet.decrypt(body.encode("ascii"))
            elif kind == "p1":
                raw = base64.b64decode(body)
            else:
                raise CryptoError("unknown template encoding")
        except CryptoError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CryptoError("template could not be decrypted (wrong key?)") from exc
        arr = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        if arr.shape != (dim,):
            raise CryptoError("template has an unexpected size")
        return arr
