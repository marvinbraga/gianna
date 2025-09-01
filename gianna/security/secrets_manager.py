"""
Secure secrets management for Gianna production.
Handles encryption, storage, and retrieval of sensitive configuration data.
"""

import base64
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

import yaml
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecretsManager:
    """Secure secrets management with encryption and validation."""

    def __init__(
        self,
        secrets_file: Optional[str] = None,
        master_key: Optional[str] = None,
        encryption_enabled: bool = True,
    ):
        """Initialize secrets manager."""
        self.secrets_file = secrets_file or "secrets/secrets.yaml"
        self.encryption_enabled = encryption_enabled
        self._secrets_cache: Dict[str, Any] = {}
        self._cipher_suite = None

        if encryption_enabled:
            self._setup_encryption(master_key)

        self._load_secrets()

    def _setup_encryption(self, master_key: Optional[str] = None):
        """Set up encryption for secrets."""
        try:
            if master_key:
                # Use provided master key
                key_bytes = master_key.encode()
            else:
                # Generate key from environment or create new
                master_key_env = os.getenv("GIANNA_MASTER_KEY")
                if master_key_env:
                    key_bytes = base64.b64decode(master_key_env)
                else:
                    # Create new key and warn
                    key_bytes = Fernet.generate_key()
                    logger.warning(
                        "No master key found, generated new key. Set GIANNA_MASTER_KEY environment variable."
                    )
                    logger.info(
                        f"Generated master key: {base64.b64encode(key_bytes).decode()}"
                    )

            # Derive key using PBKDF2
            salt = os.getenv("GIANNA_SALT", "gianna_default_salt").encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(key_bytes))
            self._cipher_suite = Fernet(key)

        except Exception as e:
            logger.error(f"Failed to setup encryption: {e}")
            self.encryption_enabled = False

    def _load_secrets(self):
        """Load secrets from file."""
        try:
            if os.path.exists(self.secrets_file):
                with open(self.secrets_file, "r") as f:
                    if self.secrets_file.endswith(".json"):
                        self._secrets_cache = json.load(f)
                    else:
                        self._secrets_cache = yaml.safe_load(f) or {}

                logger.info(f"Loaded secrets from {self.secrets_file}")
            else:
                logger.warning(f"Secrets file not found: {self.secrets_file}")
                self._secrets_cache = {}

        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            self._secrets_cache = {}

    def _save_secrets(self):
        """Save secrets to file."""
        try:
            os.makedirs(os.path.dirname(self.secrets_file), exist_ok=True)

            with open(self.secrets_file, "w") as f:
                if self.secrets_file.endswith(".json"):
                    json.dump(self._secrets_cache, f, indent=2)
                else:
                    yaml.dump(self._secrets_cache, f, default_flow_style=False)

            # Set restrictive permissions
            os.chmod(self.secrets_file, 0o600)

        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise

    def encrypt_value(self, value: str) -> str:
        """Encrypt a secret value."""
        if not self.encryption_enabled or not self._cipher_suite:
            return value

        try:
            encrypted = self._cipher_suite.encrypt(value.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            return value

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a secret value."""
        if not self.encryption_enabled or not self._cipher_suite:
            return encrypted_value

        try:
            encrypted_bytes = base64.b64decode(encrypted_value)
            decrypted = self._cipher_suite.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            return encrypted_value

    def set_secret(self, key: str, value: str, encrypt: bool = True):
        """Set a secret value."""
        try:
            keys = key.split(".")
            current = self._secrets_cache

            # Navigate to the correct nested location
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Encrypt if requested
            if encrypt and self.encryption_enabled:
                value = self.encrypt_value(value)
                key_with_prefix = f"encrypted_{keys[-1]}"
            else:
                key_with_prefix = keys[-1]

            current[key_with_prefix] = value
            self._save_secrets()

            logger.info(f"Secret set for key: {key}")

        except Exception as e:
            logger.error(f"Failed to set secret for key {key}: {e}")
            raise

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value."""
        try:
            keys = key.split(".")
            current = self._secrets_cache

            # Navigate through the nested structure
            for k in keys[:-1]:
                if k not in current:
                    return default
                current = current[k]

            final_key = keys[-1]

            # Try encrypted version first
            encrypted_key = f"encrypted_{final_key}"
            if encrypted_key in current:
                encrypted_value = current[encrypted_key]
                return self.decrypt_value(encrypted_value)

            # Try plain version
            if final_key in current:
                return current[final_key]

            # Try from environment variables
            env_key = key.upper().replace(".", "_")
            env_value = os.getenv(env_key)
            if env_value:
                return env_value

            return default

        except Exception as e:
            logger.error(f"Failed to get secret for key {key}: {e}")
            return default

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        return self.get_secret(f"api_keys.{provider}")

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "host": self.get_secret("database.host", "localhost"),
            "port": int(self.get_secret("database.port", "5432")),
            "name": self.get_secret("database.name", "gianna_production"),
            "username": self.get_secret("database.username", "gianna_user"),
            "password": self.get_secret("database.password"),
            "ssl_mode": self.get_secret("database.ssl_mode", "require"),
        }

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            "host": self.get_secret("redis.host", "localhost"),
            "port": int(self.get_secret("redis.port", "6379")),
            "password": self.get_secret("redis.password"),
            "ssl": self.get_secret("redis.ssl", "false").lower() == "true",
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return {
            "sentry_dsn": self.get_secret("monitoring.sentry.dsn"),
            "slack_webhook": self.get_secret("monitoring.slack.webhook_url"),
            "prometheus_auth": {
                "username": self.get_secret("monitoring.prometheus.username"),
                "password": self.get_secret("monitoring.prometheus.password"),
            },
            "grafana_admin_password": self.get_secret(
                "monitoring.grafana.admin_password"
            ),
        }

    def get_ssl_config(self) -> Dict[str, Any]:
        """Get SSL configuration."""
        return {
            "certificate_path": self.get_secret("ssl.certificate_path"),
            "private_key_path": self.get_secret("ssl.private_key_path"),
            "ca_bundle_path": self.get_secret("ssl.ca_bundle_path"),
        }

    def get_auth_config(self) -> Dict[str, Any]:
        """Get authentication configuration."""
        return {
            "jwt_secret": self.get_secret("auth.jwt_secret"),
            "session_secret": self.get_secret("auth.session_secret"),
        }

    def get_backup_config(self) -> Dict[str, Any]:
        """Get backup configuration."""
        return {
            "aws": {
                "access_key_id": self.get_secret("backup.aws.access_key_id"),
                "secret_access_key": self.get_secret("backup.aws.secret_access_key"),
                "bucket": self.get_secret("backup.aws.bucket"),
                "region": self.get_secret("backup.aws.region", "us-east-1"),
            },
            "gcp": {
                "service_account_key": self.get_secret(
                    "backup.gcp.service_account_key"
                ),
                "bucket": self.get_secret("backup.gcp.bucket"),
            },
        }

    def list_secrets(self, prefix: str = "") -> List[str]:
        """List available secret keys with optional prefix filter."""
        keys = []

        def extract_keys(obj: Dict[str, Any], current_prefix: str = ""):
            for key, value in obj.items():
                full_key = f"{current_prefix}.{key}" if current_prefix else key

                if isinstance(value, dict):
                    extract_keys(value, full_key)
                else:
                    # Remove 'encrypted_' prefix if present
                    display_key = full_key.replace(".encrypted_", ".")
                    if not prefix or display_key.startswith(prefix):
                        keys.append(display_key)

        extract_keys(self._secrets_cache)
        return sorted(keys)

    def validate_secrets(self) -> Dict[str, bool]:
        """Validate that required secrets are present."""
        required_secrets = [
            "api_keys.openai",
            "database.password",
            "auth.jwt_secret",
        ]

        validation_results = {}
        for secret_key in required_secrets:
            value = self.get_secret(secret_key)
            validation_results[secret_key] = value is not None and len(value) > 0

        return validation_results

    def rotate_encryption_key(self, new_master_key: str):
        """Rotate the encryption key and re-encrypt all secrets."""
        if not self.encryption_enabled:
            raise ValueError("Encryption is not enabled")

        logger.info("Starting encryption key rotation...")

        # Decrypt all current secrets
        decrypted_secrets = {}

        def decrypt_recursive(obj: Dict[str, Any], path: str = ""):
            result = {}
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    result[key] = decrypt_recursive(value, current_path)
                elif key.startswith("encrypted_"):
                    original_key = key[10:]  # Remove 'encrypted_' prefix
                    decrypted_value = self.decrypt_value(value)
                    result[original_key] = decrypted_value
                else:
                    result[key] = value
            return result

        decrypted_secrets = decrypt_recursive(self._secrets_cache)

        # Setup new encryption
        old_cipher = self._cipher_suite
        self._setup_encryption(new_master_key)

        if not self._cipher_suite:
            # Restore old cipher if new setup failed
            self._cipher_suite = old_cipher
            raise RuntimeError("Failed to setup new encryption key")

        # Re-encrypt all secrets
        self._secrets_cache = {}

        def encrypt_and_store(obj: Dict[str, Any], path: str = ""):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    encrypt_and_store(value, current_path)
                else:
                    self.set_secret(current_path, str(value), encrypt=True)

        encrypt_and_store(decrypted_secrets)

        logger.info("Encryption key rotation completed")

    def export_secrets(self, include_encrypted: bool = False) -> Dict[str, Any]:
        """Export secrets (for backup purposes)."""
        if include_encrypted:
            return self._secrets_cache.copy()
        else:
            # Return decrypted version
            decrypted = {}

            def decrypt_recursive(obj: Dict[str, Any]):
                result = {}
                for key, value in obj.items():
                    if isinstance(value, dict):
                        result[key] = decrypt_recursive(value)
                    elif key.startswith("encrypted_"):
                        original_key = key[10:]
                        result[original_key] = self.decrypt_value(value)
                    else:
                        result[key] = value
                return result

            return decrypt_recursive(self._secrets_cache)

    def import_secrets(self, secrets_data: Dict[str, Any], encrypt: bool = True):
        """Import secrets from external source."""
        logger.info("Importing secrets...")

        def import_recursive(obj: Dict[str, Any], path: str = ""):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    import_recursive(value, current_path)
                else:
                    self.set_secret(current_path, str(value), encrypt=encrypt)

        import_recursive(secrets_data)
        logger.info("Secrets import completed")


# Global secrets manager instance
_global_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance."""
    global _global_secrets_manager
    if _global_secrets_manager is None:
        _global_secrets_manager = SecretsManager()
    return _global_secrets_manager


def reset_secrets_manager():
    """Reset global secrets manager (mainly for testing)."""
    global _global_secrets_manager
    _global_secrets_manager = None
