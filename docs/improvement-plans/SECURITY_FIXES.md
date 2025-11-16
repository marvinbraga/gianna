# Plano de Correções de Segurança

**Status:** Pendente
**Estimativa:** 2-3 dias
**Impacto:** CRÍTICO para produção
**Prioridade:** ALTA

---

## Resumo Executivo

Este documento detalha as correções de segurança identificadas durante a análise do código. Embora o projeto já possua boas práticas de segurança (validação de entrada, secrets manager, rate limiting), foram identificados pontos que necessitam melhorias.

### Severidade dos Problemas

| Item | Severidade | Impacto | Esforço |
|------|-----------|---------|---------|
| 1. Hardcoded test API keys | 🔴 CRÍTICO | Alto | Baixo |
| 2. Default salt em criptografia | 🟡 MÉDIO | Médio | Baixo |
| 3. Cache de embeddings sem criptografia | 🟡 MÉDIO | Médio | Médio |
| 4. Deserialização JSON sem validação | 🟢 BAIXO | Baixo | Baixo |

---

## 1. Hardcoded Test API Keys

**Arquivo:** `tests/conftest.py:45-54`
**Severidade:** 🔴 CRÍTICO
**Status:** JÁ DETALHADO EM PRIORITY_1_CRITICAL.md

### Referência

Ver seção completa em:
- **Documento:** `PRIORITY_1_CRITICAL.md`
- **Seção:** "4. Corrigir Hardcoded Test API Keys"

### Resumo da Solução

```python
# Usar environment variables com fallback seguro
@pytest.fixture
def mock_api_keys(monkeypatch):
    keys = {
        "OPENAI_API_KEY": os.getenv("TEST_OPENAI_API_KEY", "sk-test-fake-key-123"),
        # ... outras keys
    }
    for key, value in keys.items():
        monkeypatch.setenv(key, value)
    return keys
```

---

## 2. Default Salt para Criptografia

**Arquivo:** `gianna/security/secrets_manager.py:63-64`
**Severidade:** 🟡 MÉDIO
**Tipo:** Weak Cryptography

### Problema Identificado

```python
# gianna/security/secrets_manager.py
salt = os.getenv("GIANNA_SALT", "gianna_default_salt").encode()
# ^ Usa salt padrão previsível se não configurado
```

### Risco de Segurança

1. **Salt Previsível:** O salt padrão "gianna_default_salt" é conhecido
2. **Rainbow Tables:** Atacante pode pré-computar hashes
3. **Múltiplas Instalações:** Todas as instalações sem configuração usam mesmo salt
4. **Compliance:** Não atende requisitos de criptografia forte (FIPS, PCI-DSS)

### Impacto

- **Confidencialidade:** Secrets podem ser decifrados com salt conhecido
- **Integridade:** Violação de best practices de criptografia
- **Compliance:** Falha em auditorias de segurança

### Solução Proposta

#### Opção 1: Gerar Salt Aleatório na Primeira Execução (RECOMENDADO)

```python
# gianna/security/secrets_manager.py

import os
import secrets
from pathlib import Path
from loguru import logger

class SecretsManager:
    def __init__(self, secrets_file: Path = None):
        self.secrets_file = secrets_file or Path.home() / ".gianna" / "secrets.enc"
        self.salt_file = self.secrets_file.parent / ".salt"

        # Garantir que diretório existe
        self.secrets_file.parent.mkdir(parents=True, exist_ok=True)

        # Obter ou gerar salt
        self.salt = self._get_or_generate_salt()

        # Derivar chave
        self.key = self._derive_key(self._get_master_password(), self.salt)
        self.cipher = Fernet(self.key)

    def _get_or_generate_salt(self) -> bytes:
        """
        Get salt from file or generate new one.

        Order of precedence:
        1. Environment variable GIANNA_SALT (base64 encoded)
        2. Salt file (~/.gianna/.salt)
        3. Generate new random salt

        Returns:
            bytes: 16-byte salt for key derivation
        """
        # 1. Check environment variable
        env_salt = os.getenv("GIANNA_SALT")
        if env_salt:
            try:
                salt = base64.b64decode(env_salt)
                if len(salt) == 16:
                    logger.info("Using salt from environment variable")
                    return salt
                else:
                    logger.warning(
                        f"GIANNA_SALT has invalid length {len(salt)}, expected 16 bytes"
                    )
            except Exception as e:
                logger.warning(f"Failed to decode GIANNA_SALT: {e}")

        # 2. Check salt file
        if self.salt_file.exists():
            try:
                with open(self.salt_file, 'rb') as f:
                    salt = f.read()

                if len(salt) == 16:
                    logger.info(f"Loaded salt from {self.salt_file}")
                    return salt
                else:
                    logger.warning(
                        f"Salt file has invalid length {len(salt)}, regenerating"
                    )
            except Exception as e:
                logger.error(f"Failed to load salt file: {e}")

        # 3. Generate new random salt
        logger.warning(
            "No salt found, generating new random salt. "
            "This will invalidate existing encrypted secrets!"
        )

        salt = secrets.token_bytes(16)  # Cryptographically secure random

        # Save to file for persistence
        try:
            with open(self.salt_file, 'wb') as f:
                f.write(salt)

            # Set restrictive permissions
            os.chmod(self.salt_file, 0o600)

            logger.info(f"Generated and saved new salt to {self.salt_file}")
        except Exception as e:
            logger.error(f"Failed to save salt file: {e}")
            logger.warning("Salt will not persist across restarts!")

        return salt

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.

        Args:
            password: Master password
            salt: 16-byte salt

        Returns:
            bytes: Base64-encoded 32-byte key for Fernet
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,  # OWASP recommendation (2024)
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def rotate_salt(self, new_salt: bytes = None):
        """
        Rotate encryption salt.

        WARNING: This will re-encrypt all secrets with new salt.
        Make sure to backup secrets before rotation.

        Args:
            new_salt: New salt (16 bytes). If None, generates random salt.
        """
        # Load all current secrets
        current_secrets = self.get_all_secrets()

        # Generate or use provided salt
        if new_salt is None:
            new_salt = secrets.token_bytes(16)
        elif len(new_salt) != 16:
            raise ValueError("Salt must be 16 bytes")

        # Save old salt for rollback
        old_salt = self.salt
        old_salt_file_backup = self.salt_file.with_suffix('.salt.bak')

        try:
            # Backup old salt
            if self.salt_file.exists():
                import shutil
                shutil.copy2(self.salt_file, old_salt_file_backup)

            # Update salt
            self.salt = new_salt

            # Re-derive key
            self.key = self._derive_key(self._get_master_password(), self.salt)
            self.cipher = Fernet(self.key)

            # Save new salt
            with open(self.salt_file, 'wb') as f:
                f.write(new_salt)
            os.chmod(self.salt_file, 0o600)

            # Re-encrypt all secrets
            for key, value in current_secrets.items():
                self.set_secret(key, value)

            logger.info("Salt rotation completed successfully")

            # Clean up backup
            if old_salt_file_backup.exists():
                old_salt_file_backup.unlink()

        except Exception as e:
            # Rollback on error
            logger.error(f"Salt rotation failed: {e}")
            logger.info("Rolling back to old salt...")

            self.salt = old_salt
            self.key = self._derive_key(self._get_master_password(), old_salt)
            self.cipher = Fernet(self.key)

            if old_salt_file_backup.exists():
                import shutil
                shutil.copy2(old_salt_file_backup, self.salt_file)

            raise
```

#### Opção 2: Forçar Configuração Explícita

```python
# Alternativa mais rigorosa
def _get_or_generate_salt(self) -> bytes:
    env_salt = os.getenv("GIANNA_SALT")

    if not env_salt:
        raise ValueError(
            "GIANNA_SALT environment variable not set! "
            "Generate one with: python -c 'import secrets, base64; "
            "print(base64.b64encode(secrets.token_bytes(16)).decode())'"
        )

    return base64.b64decode(env_salt)
```

### Implementação

#### Passo 1: Atualizar SecretsManager

```bash
# Editar arquivo
vim gianna/security/secrets_manager.py

# Implementar _get_or_generate_salt() conforme Opção 1
```

#### Passo 2: Adicionar CLI para Gerenciamento de Salt

```python
# gianna/cli/secrets.py

import click
import secrets
import base64
from gianna.security.secrets_manager import SecretsManager

@click.group()
def secrets_cli():
    """Manage Gianna secrets."""
    pass

@secrets_cli.command()
def generate_salt():
    """Generate new random salt."""
    salt = secrets.token_bytes(16)
    encoded = base64.b64encode(salt).decode()

    click.echo("Generated new salt:")
    click.echo(f"  {encoded}")
    click.echo("\nAdd to your environment:")
    click.echo(f"  export GIANNA_SALT='{encoded}'")
    click.echo("\nOr add to .env:")
    click.echo(f"  GIANNA_SALT={encoded}")

@secrets_cli.command()
@click.option('--backup/--no-backup', default=True)
def rotate_salt(backup):
    """Rotate encryption salt."""
    manager = SecretsManager()

    if backup:
        click.echo("Backing up secrets...")
        manager.export_secrets("secrets_backup.json")

    click.confirm(
        "This will re-encrypt all secrets. Continue?",
        abort=True
    )

    manager.rotate_salt()
    click.echo("✓ Salt rotation completed")

@secrets_cli.command()
def check_salt():
    """Check current salt configuration."""
    manager = SecretsManager()

    if os.getenv("GIANNA_SALT"):
        click.echo("✓ Salt configured via environment variable")
    elif manager.salt_file.exists():
        click.echo(f"✓ Salt loaded from {manager.salt_file}")
    else:
        click.echo("⚠ Using generated salt (not persisted)")
        click.echo("  Run 'gianna secrets generate-salt' to create one")

if __name__ == '__main__':
    secrets_cli()
```

#### Passo 3: Atualizar Documentação

```markdown
# docs/security/SECRETS_MANAGEMENT.md

## Salt Configuration

The secrets manager uses a salt for key derivation. You have three options:

### Option 1: Environment Variable (Recommended for production)

```bash
# Generate salt
python -c 'import secrets, base64; print(base64.b64encode(secrets.token_bytes(16)).decode())'

# Output: abc123def456...

# Add to .env
echo "GIANNA_SALT=abc123def456..." >> .env
```

### Option 2: Salt File (Automatic)

If no environment variable is set, Gianna will:
1. Generate random salt on first run
2. Save to `~/.gianna/.salt`
3. Reuse on subsequent runs

### Option 3: CLI Tool

```bash
# Generate new salt
gianna secrets generate-salt

# Check current salt
gianna secrets check-salt

# Rotate salt (re-encrypts all secrets)
gianna secrets rotate-salt
```

## Security Best Practices

1. **Never commit salt to version control**
   ```bash
   echo ".salt" >> .gitignore
   ```

2. **Use different salts per environment**
   - Development: Auto-generated
   - Staging: Environment-specific
   - Production: Securely managed

3. **Rotate salt periodically**
   ```bash
   gianna secrets rotate-salt --backup
   ```
```

#### Passo 4: Testes

```python
# tests/unit/test_secrets_manager_security.py

import pytest
import secrets
import os
from pathlib import Path
from gianna.security.secrets_manager import SecretsManager

class TestSecretsManagerSecurity:
    """Test security aspects of secrets manager."""

    def test_generates_random_salt_if_not_configured(self, tmp_path):
        """Test that random salt is generated if not configured."""
        secrets_file = tmp_path / "secrets.enc"

        # Clear environment
        os.environ.pop("GIANNA_SALT", None)

        manager = SecretsManager(secrets_file=secrets_file)

        # Salt should be 16 bytes
        assert len(manager.salt) == 16

        # Salt should be random (very unlikely to be all zeros)
        assert manager.salt != b'\x00' * 16

    def test_uses_env_salt_if_configured(self, tmp_path, monkeypatch):
        """Test that environment salt is used if configured."""
        import base64

        secrets_file = tmp_path / "secrets.enc"
        expected_salt = secrets.token_bytes(16)
        encoded_salt = base64.b64encode(expected_salt).decode()

        monkeypatch.setenv("GIANNA_SALT", encoded_salt)

        manager = SecretsManager(secrets_file=secrets_file)

        assert manager.salt == expected_salt

    def test_persists_salt_to_file(self, tmp_path):
        """Test that salt is persisted to file."""
        secrets_file = tmp_path / "secrets.enc"
        salt_file = tmp_path / ".salt"

        manager = SecretsManager(secrets_file=secrets_file)

        # Salt file should exist
        assert salt_file.exists()

        # Salt should match
        with open(salt_file, 'rb') as f:
            saved_salt = f.read()

        assert saved_salt == manager.salt

    def test_salt_file_has_restrictive_permissions(self, tmp_path):
        """Test that salt file has secure permissions."""
        secrets_file = tmp_path / "secrets.enc"
        salt_file = tmp_path / ".salt"

        manager = SecretsManager(secrets_file=secrets_file)

        # Check permissions (0o600 = rw-------)
        import stat
        st = os.stat(salt_file)
        mode = stat.S_IMODE(st.st_mode)

        assert mode == 0o600

    def test_rotate_salt_reencrypts_secrets(self, tmp_path):
        """Test that salt rotation re-encrypts all secrets."""
        secrets_file = tmp_path / "secrets.enc"

        manager = SecretsManager(secrets_file=secrets_file)

        # Set some secrets
        manager.set_secret("key1", "value1")
        manager.set_secret("key2", "value2")

        # Rotate salt
        old_salt = manager.salt
        manager.rotate_salt()

        # Salt should change
        assert manager.salt != old_salt

        # Secrets should still be accessible
        assert manager.get_secret("key1") == "value1"
        assert manager.get_secret("key2") == "value2"

    def test_different_salts_produce_different_ciphertexts(self, tmp_path):
        """Test that same plaintext with different salts produces different ciphertexts."""
        secrets_file1 = tmp_path / "secrets1.enc"
        secrets_file2 = tmp_path / "secrets2.enc"

        manager1 = SecretsManager(secrets_file=secrets_file1)
        manager2 = SecretsManager(secrets_file=secrets_file2)

        # Set same secret with different managers (different salts)
        manager1.set_secret("key", "value")
        manager2.set_secret("key", "value")

        # Read encrypted files
        with open(secrets_file1, 'rb') as f:
            ciphertext1 = f.read()

        with open(secrets_file2, 'rb') as f:
            ciphertext2 = f.read()

        # Ciphertexts should be different
        assert ciphertext1 != ciphertext2
```

### Critérios de Aceitação

- [ ] Salt não tem valor padrão hardcoded
- [ ] Salt é gerado aleatoriamente se não configurado
- [ ] Salt é persistido em arquivo com permissões 0o600
- [ ] CLI para gerenciamento de salt implementado
- [ ] Documentação atualizada
- [ ] Testes de segurança passando
- [ ] Rotação de salt funcionando

---

## 3. Cache de Embeddings Sem Criptografia

**Arquivo:** `gianna/memory/embeddings.py:32-40`
**Severidade:** 🟡 MÉDIO
**Tipo:** Data Exposure

### Problema

```python
# Cache é armazenado em plaintext
cache_file = self.cache_dir / f"{self.model_name}_cache.json"
with open(cache_file, "r") as f:
    self._embedding_cache = json.load(f)
```

**Risco:**
- Embeddings podem conter informação sensível do contexto
- Cache em plaintext no home do usuário
- Não há criptografia at rest

### Solução

```python
# gianna/memory/embeddings.py

from gianna.security.secrets_manager import SecretsManager

class EmbeddingGenerator:
    def __init__(self, model_name: str = "default", encrypt_cache: bool = False):
        self.model_name = model_name
        self.encrypt_cache = encrypt_cache

        if encrypt_cache:
            self.secrets_manager = SecretsManager()

    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / f"{self.model_name}_cache.json"

        if not cache_file.exists():
            return

        try:
            if self.encrypt_cache:
                # Load encrypted cache
                encrypted_data = cache_file.read_bytes()
                decrypted_json = self.secrets_manager.cipher.decrypt(encrypted_data)
                self._embedding_cache = json.loads(decrypted_json)
            else:
                # Load plaintext cache
                with open(cache_file, 'r') as f:
                    self._embedding_cache = json.load(f)

        except Exception as e:
            logger.error(f"Failed to load embedding cache: {e}")
            self._embedding_cache = {}

    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        cache_file = self.cache_dir / f"{self.model_name}_cache.json"

        try:
            if self.encrypt_cache:
                # Save encrypted
                json_data = json.dumps(self._embedding_cache)
                encrypted_data = self.secrets_manager.cipher.encrypt(json_data.encode())
                cache_file.write_bytes(encrypted_data)
                os.chmod(cache_file, 0o600)
            else:
                # Save plaintext
                with open(cache_file, 'w') as f:
                    json.dump(self._embedding_cache, f)

        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
```

---

## 4. Validação em Deserialização JSON

**Severidade:** 🟢 BAIXO
**Tipo:** Input Validation

### Problema

Alguns lugares usam `json.load()` sem validação:

```python
data = json.load(f)  # Sem validação de schema
```

### Solução

```python
from pydantic import BaseModel, ValidationError

class CacheEntryModel(BaseModel):
    key: str
    value: Any
    timestamp: float
    ttl: int

def load_cache(file_path: Path) -> dict:
    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    # Validate with Pydantic
    try:
        validated = {
            k: CacheEntryModel(**v)
            for k, v in raw_data.items()
        }
        return validated
    except ValidationError as e:
        logger.error(f"Invalid cache data: {e}")
        return {}
```

---

## Checklist de Segurança Geral

### Antes de Deploy em Produção

- [ ] Todas as API keys em environment variables (não hardcoded)
- [ ] Salt de criptografia configurado e único por ambiente
- [ ] Permissões de arquivos configuradas (0o600 para secrets)
- [ ] Rate limiting configurado
- [ ] Input validation em todos os endpoints
- [ ] Logging estruturado (sem leak de secrets)
- [ ] Secrets não commitados em git
- [ ] Dependencies atualizadas (sem vulnerabilidades conhecidas)
- [ ] HTTPS configurado (se aplicável)
- [ ] Backup de secrets configurado

### Audit Regular

```bash
# Check for hardcoded secrets
git secrets --scan

# Check dependencies
pip-audit

# Check for security issues
bandit -r gianna/

# Check for leaked secrets in git history
gitleaks detect
```

---

## Estimativa de Esforço

| Task | Esforço | Prioridade |
|------|---------|-----------|
| 1. Test API keys | 0.5 dia | P0 |
| 2. Salt management | 1 dia | P0 |
| 3. Encrypt cache | 0.5 dia | P1 |
| 4. JSON validation | 0.5 dia | P2 |

**Total:** 2.5 dias

---

## Métricas de Sucesso

- [ ] Zero hardcoded secrets
- [ ] Todos os secrets criptografados com salt único
- [ ] Security audit passing (bandit, pip-audit)
- [ ] Documentação de segurança completa
- [ ] Testes de segurança passando
