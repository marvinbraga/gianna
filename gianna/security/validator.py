"""
Input validation and sanitization for Gianna production security.
Provides comprehensive validation for all user inputs and API requests.
"""

import base64
import hashlib
import html
import json
import logging
import re
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str, field: str = None, code: str = None):
        super().__init__(message)
        self.field = field
        self.code = code


class ValidationType(Enum):
    """Types of validation checks."""

    STRING = "string"
    EMAIL = "email"
    URL = "url"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    BASE64 = "base64"
    UUID = "uuid"
    FILENAME = "filename"
    PATH = "path"
    IP_ADDRESS = "ip_address"
    AUDIO_FORMAT = "audio_format"
    LLM_PROVIDER = "llm_provider"


@dataclass
class ValidationRule:
    """Validation rule configuration."""

    field_type: ValidationType
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    custom_validator: Optional[callable] = None


class InputValidator:
    """Comprehensive input validator with security focus."""

    def __init__(self):
        """Initialize validator with security patterns."""
        self.patterns = {
            "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
            "url": re.compile(r"^https?://[^\s/$.?#].[^\s]*$"),
            "uuid": re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                re.IGNORECASE,
            ),
            "filename": re.compile(r"^[a-zA-Z0-9._-]+\.[a-zA-Z0-9]+$"),
            "safe_path": re.compile(r"^[a-zA-Z0-9/._-]+$"),
            "ip_address": re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"),
            "alphanumeric": re.compile(r"^[a-zA-Z0-9]+$"),
            "safe_string": re.compile(r"^[a-zA-Z0-9\s._-]+$"),
        }

        # Security patterns to detect malicious input
        self.security_patterns = {
            "sql_injection": [
                re.compile(
                    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                    re.IGNORECASE,
                ),
                re.compile(r"(\-\-|\#|\/\*|\*\/)", re.IGNORECASE),
                re.compile(r"(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+", re.IGNORECASE),
            ],
            "xss": [
                re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
                re.compile(r"javascript:", re.IGNORECASE),
                re.compile(r"on\w+\s*=", re.IGNORECASE),
                re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL),
            ],
            "path_traversal": [
                re.compile(r"\.\./"),
                re.compile(r"\.\.\\"),
                re.compile(r"/etc/passwd"),
                re.compile(r"%2e%2e"),
            ],
            "command_injection": [
                re.compile(r"[;&|`$]"),
                re.compile(r"\b(rm|cat|ls|ps|kill|sudo|su)\b"),
                re.compile(r"(\$\(|\`)", re.IGNORECASE),
            ],
        }

        # Audio format validation
        self.audio_formats = ["mp3", "wav", "ogg", "flac", "m4a", "aac"]
        self.llm_providers = [
            "openai",
            "anthropic",
            "google",
            "groq",
            "nvidia",
            "ollama",
            "xai",
            "cohere",
        ]

    def validate_field(
        self, value: Any, rule: ValidationRule, field_name: str = None
    ) -> Any:
        """Validate a single field against its rule."""
        # Check if field is required
        if rule.required and (value is None or value == ""):
            raise ValidationError(
                f"Field '{field_name or 'unknown'}' is required", field_name, "required"
            )

        # If not required and empty, return None
        if not rule.required and (value is None or value == ""):
            return None

        # Type-specific validation
        validated_value = self._validate_by_type(value, rule.field_type, field_name)

        # Length validation for strings
        if rule.field_type == ValidationType.STRING and isinstance(
            validated_value, str
        ):
            if rule.min_length and len(validated_value) < rule.min_length:
                raise ValidationError(
                    f"Field '{field_name}' must be at least {rule.min_length} characters",
                    field_name,
                    "min_length",
                )
            if rule.max_length and len(validated_value) > rule.max_length:
                raise ValidationError(
                    f"Field '{field_name}' must not exceed {rule.max_length} characters",
                    field_name,
                    "max_length",
                )

        # Value range validation for numbers
        if rule.field_type in [ValidationType.INTEGER, ValidationType.FLOAT]:
            if rule.min_value is not None and validated_value < rule.min_value:
                raise ValidationError(
                    f"Field '{field_name}' must be at least {rule.min_value}",
                    field_name,
                    "min_value",
                )
            if rule.max_value is not None and validated_value > rule.max_value:
                raise ValidationError(
                    f"Field '{field_name}' must not exceed {rule.max_value}",
                    field_name,
                    "max_value",
                )

        # Pattern validation
        if rule.pattern and isinstance(validated_value, str):
            if not re.match(rule.pattern, validated_value):
                raise ValidationError(
                    f"Field '{field_name}' does not match required pattern",
                    field_name,
                    "pattern",
                )

        # Allowed values validation
        if rule.allowed_values and validated_value not in rule.allowed_values:
            raise ValidationError(
                f"Field '{field_name}' must be one of: {', '.join(map(str, rule.allowed_values))}",
                field_name,
                "allowed_values",
            )

        # Custom validator
        if rule.custom_validator:
            try:
                custom_result = rule.custom_validator(validated_value)
                if custom_result is False:
                    raise ValidationError(
                        f"Field '{field_name}' failed custom validation",
                        field_name,
                        "custom",
                    )
                elif custom_result is not True and custom_result is not None:
                    validated_value = custom_result
            except Exception as e:
                raise ValidationError(
                    f"Custom validation failed for '{field_name}': {str(e)}",
                    field_name,
                    "custom",
                )

        return validated_value

    def _validate_by_type(
        self, value: Any, field_type: ValidationType, field_name: str = None
    ) -> Any:
        """Validate value by its expected type."""
        if field_type == ValidationType.STRING:
            if not isinstance(value, str):
                if isinstance(value, (int, float, bool)):
                    value = str(value)
                else:
                    raise ValidationError(
                        f"Field '{field_name}' must be a string", field_name, "type"
                    )

            # Security check for strings
            self._check_security_threats(value, field_name)
            return value

        elif field_type == ValidationType.EMAIL:
            if not isinstance(value, str):
                raise ValidationError(
                    f"Field '{field_name}' must be a string", field_name, "type"
                )

            if not self.patterns["email"].match(value):
                raise ValidationError(
                    f"Field '{field_name}' must be a valid email address",
                    field_name,
                    "format",
                )
            return value.lower()

        elif field_type == ValidationType.URL:
            if not isinstance(value, str):
                raise ValidationError(
                    f"Field '{field_name}' must be a string", field_name, "type"
                )

            if not self.patterns["url"].match(value):
                raise ValidationError(
                    f"Field '{field_name}' must be a valid URL", field_name, "format"
                )
            return value

        elif field_type == ValidationType.INTEGER:
            try:
                return int(value)
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Field '{field_name}' must be an integer", field_name, "type"
                )

        elif field_type == ValidationType.FLOAT:
            try:
                return float(value)
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Field '{field_name}' must be a number", field_name, "type"
                )

        elif field_type == ValidationType.BOOLEAN:
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                if value.lower() in ["true", "1", "yes", "on"]:
                    return True
                elif value.lower() in ["false", "0", "no", "off"]:
                    return False
            raise ValidationError(
                f"Field '{field_name}' must be a boolean", field_name, "type"
            )

        elif field_type == ValidationType.JSON:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    raise ValidationError(
                        f"Field '{field_name}' must be valid JSON", field_name, "format"
                    )
            elif isinstance(value, (dict, list)):
                return value
            else:
                raise ValidationError(
                    f"Field '{field_name}' must be JSON data", field_name, "type"
                )

        elif field_type == ValidationType.BASE64:
            if not isinstance(value, str):
                raise ValidationError(
                    f"Field '{field_name}' must be a string", field_name, "type"
                )

            try:
                base64.b64decode(value, validate=True)
                return value
            except Exception:
                raise ValidationError(
                    f"Field '{field_name}' must be valid base64", field_name, "format"
                )

        elif field_type == ValidationType.UUID:
            if not isinstance(value, str):
                raise ValidationError(
                    f"Field '{field_name}' must be a string", field_name, "type"
                )

            if not self.patterns["uuid"].match(value):
                raise ValidationError(
                    f"Field '{field_name}' must be a valid UUID", field_name, "format"
                )
            return value.lower()

        elif field_type == ValidationType.FILENAME:
            if not isinstance(value, str):
                raise ValidationError(
                    f"Field '{field_name}' must be a string", field_name, "type"
                )

            if not self.patterns["filename"].match(value):
                raise ValidationError(
                    f"Field '{field_name}' must be a valid filename",
                    field_name,
                    "format",
                )

            # Additional security checks for filenames
            if ".." in value or "/" in value or "\\" in value:
                raise ValidationError(
                    f"Field '{field_name}' contains invalid characters",
                    field_name,
                    "security",
                )

            return value

        elif field_type == ValidationType.PATH:
            if not isinstance(value, str):
                raise ValidationError(
                    f"Field '{field_name}' must be a string", field_name, "type"
                )

            # Security check for path traversal
            if ".." in value:
                raise ValidationError(
                    f"Field '{field_name}' contains path traversal attempt",
                    field_name,
                    "security",
                )

            if not self.patterns["safe_path"].match(value):
                raise ValidationError(
                    f"Field '{field_name}' contains invalid path characters",
                    field_name,
                    "format",
                )

            return value

        elif field_type == ValidationType.IP_ADDRESS:
            if not isinstance(value, str):
                raise ValidationError(
                    f"Field '{field_name}' must be a string", field_name, "type"
                )

            if not self.patterns["ip_address"].match(value):
                raise ValidationError(
                    f"Field '{field_name}' must be a valid IP address",
                    field_name,
                    "format",
                )

            # Additional validation for IP ranges
            parts = value.split(".")
            for part in parts:
                if int(part) > 255:
                    raise ValidationError(
                        f"Field '{field_name}' must be a valid IP address",
                        field_name,
                        "format",
                    )

            return value

        elif field_type == ValidationType.AUDIO_FORMAT:
            if not isinstance(value, str):
                raise ValidationError(
                    f"Field '{field_name}' must be a string", field_name, "type"
                )

            if value.lower() not in self.audio_formats:
                raise ValidationError(
                    f"Field '{field_name}' must be a valid audio format: {', '.join(self.audio_formats)}",
                    field_name,
                    "format",
                )
            return value.lower()

        elif field_type == ValidationType.LLM_PROVIDER:
            if not isinstance(value, str):
                raise ValidationError(
                    f"Field '{field_name}' must be a string", field_name, "type"
                )

            if value.lower() not in self.llm_providers:
                raise ValidationError(
                    f"Field '{field_name}' must be a valid LLM provider: {', '.join(self.llm_providers)}",
                    field_name,
                    "format",
                )
            return value.lower()

        else:
            raise ValidationError(
                f"Unknown validation type for field '{field_name}'",
                field_name,
                "unknown_type",
            )

    def _check_security_threats(self, value: str, field_name: str = None):
        """Check for common security threats in string values."""
        for threat_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                if pattern.search(value):
                    logger.warning(
                        f"Security threat detected in field '{field_name}': {threat_type}"
                    )
                    raise ValidationError(
                        f"Field '{field_name}' contains potentially malicious content",
                        field_name,
                        f"security_{threat_type}",
                    )

    def validate_dict(
        self, data: Dict[str, Any], rules: Dict[str, ValidationRule]
    ) -> Dict[str, Any]:
        """Validate a dictionary of data against rules."""
        if not isinstance(data, dict):
            raise ValidationError("Input must be a dictionary", code="type")

        validated_data = {}

        # Check for required fields
        for field_name, rule in rules.items():
            if rule.required and field_name not in data:
                raise ValidationError(
                    f"Required field '{field_name}' is missing", field_name, "missing"
                )

        # Validate each field
        for field_name, value in data.items():
            if field_name in rules:
                try:
                    validated_data[field_name] = self.validate_field(
                        value, rules[field_name], field_name
                    )
                except ValidationError:
                    raise  # Re-raise validation errors
                except Exception as e:
                    raise ValidationError(
                        f"Validation failed for field '{field_name}': {str(e)}",
                        field_name,
                        "error",
                    )
            else:
                # Field not in rules - decide whether to include or reject
                logger.warning(f"Unknown field '{field_name}' in input data")
                # For security, we'll include but log unknown fields
                validated_data[field_name] = value

        return validated_data

    def sanitize_html(self, text: str) -> str:
        """Sanitize HTML content to prevent XSS."""
        if not isinstance(text, str):
            return text

        # Escape HTML entities
        sanitized = html.escape(text, quote=True)

        # Remove any remaining script tags or javascript
        sanitized = re.sub(
            r"<script[^>]*>.*?</script>", "", sanitized, flags=re.IGNORECASE | re.DOTALL
        )
        sanitized = re.sub(r"javascript:", "", sanitized, flags=re.IGNORECASE)

        return sanitized

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        if not isinstance(filename, str):
            return filename

        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "", filename)
        sanitized = re.sub(
            r"\.\.+", ".", sanitized
        )  # Replace multiple dots with single
        sanitized = sanitized.strip(". ")  # Remove leading/trailing dots and spaces

        # Ensure it's not empty after sanitization
        if not sanitized:
            sanitized = "unnamed_file"

        return sanitized

    def validate_audio_file(
        self, filename: str, content_type: str = None, max_size: int = None
    ) -> Dict[str, Any]:
        """Validate audio file upload."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "filename": filename,
            "format": None,
        }

        # Validate filename
        try:
            sanitized_filename = self.sanitize_filename(filename)
            if sanitized_filename != filename:
                result["warnings"].append("Filename was sanitized")
                result["filename"] = sanitized_filename

            # Extract format from filename
            parts = sanitized_filename.split(".")
            if len(parts) < 2:
                result["valid"] = False
                result["errors"].append("Filename must have an extension")
            else:
                file_format = parts[-1].lower()
                if file_format not in self.audio_formats:
                    result["valid"] = False
                    result["errors"].append(f"Invalid audio format: {file_format}")
                else:
                    result["format"] = file_format

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Filename validation failed: {str(e)}")

        # Validate content type if provided
        if content_type:
            valid_content_types = [
                "audio/mpeg",
                "audio/wav",
                "audio/ogg",
                "audio/flac",
                "audio/m4a",
                "audio/aac",
                "audio/mp4",
            ]
            if content_type not in valid_content_types:
                result["warnings"].append(f"Unexpected content type: {content_type}")

        # Validate file size if provided
        if max_size and hasattr(max_size, "__len__"):
            # This would be file content
            if len(max_size) > 50 * 1024 * 1024:  # 50MB limit
                result["valid"] = False
                result["errors"].append("File size exceeds 50MB limit")

        return result

    def validate_llm_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM API request."""
        rules = {
            "provider": ValidationRule(ValidationType.LLM_PROVIDER, required=True),
            "model": ValidationRule(
                ValidationType.STRING, required=True, max_length=100
            ),
            "message": ValidationRule(
                ValidationType.STRING, required=True, max_length=50000
            ),
            "temperature": ValidationRule(
                ValidationType.FLOAT, min_value=0.0, max_value=2.0
            ),
            "max_tokens": ValidationRule(
                ValidationType.INTEGER, min_value=1, max_value=8000
            ),
            "stream": ValidationRule(ValidationType.BOOLEAN),
        }

        return self.validate_dict(data, rules)

    def validate_user_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate general user input with security checks."""
        validated = {}

        for key, value in data.items():
            if isinstance(value, str):
                # Security checks
                self._check_security_threats(value, key)

                # Sanitize based on key name
                if "html" in key.lower() or "content" in key.lower():
                    validated[key] = self.sanitize_html(value)
                elif "filename" in key.lower() or "file" in key.lower():
                    validated[key] = self.sanitize_filename(value)
                else:
                    validated[key] = value
            else:
                validated[key] = value

        return validated

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation configuration summary."""
        return {
            "supported_types": [t.value for t in ValidationType],
            "audio_formats": self.audio_formats,
            "llm_providers": self.llm_providers,
            "security_checks": list(self.security_patterns.keys()),
        }


# Utility functions for common validation patterns
def validate_api_key(api_key: str) -> bool:
    """Validate API key format."""
    if not isinstance(api_key, str):
        return False

    # Basic format validation (adjust based on specific providers)
    if len(api_key) < 20:
        return False

    # Check for common API key patterns
    patterns = [
        r"^sk-[a-zA-Z0-9]{40,}$",  # OpenAI style
        r"^[a-zA-Z0-9]{32,}$",  # Generic style
        r"^[A-Z0-9-]{20,}$",  # Another common style
    ]

    return any(re.match(pattern, api_key) for pattern in patterns)


def create_common_rules() -> Dict[str, ValidationRule]:
    """Create common validation rules."""
    return {
        "id": ValidationRule(ValidationType.UUID, required=True),
        "name": ValidationRule(
            ValidationType.STRING, required=True, min_length=1, max_length=255
        ),
        "email": ValidationRule(ValidationType.EMAIL, required=True),
        "password": ValidationRule(
            ValidationType.STRING, required=True, min_length=8, max_length=255
        ),
        "url": ValidationRule(ValidationType.URL),
        "phone": ValidationRule(ValidationType.STRING, pattern=r"^\+?[1-9]\d{1,14}$"),
        "age": ValidationRule(ValidationType.INTEGER, min_value=0, max_value=150),
        "score": ValidationRule(ValidationType.FLOAT, min_value=0.0, max_value=1.0),
    }


# Global validator instance
_global_validator: Optional[InputValidator] = None


def get_input_validator() -> InputValidator:
    """Get global input validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = InputValidator()
    return _global_validator


def reset_input_validator():
    """Reset global input validator (mainly for testing)."""
    global _global_validator
    _global_validator = None
