# Guia de Deployment - Gianna

Este guia contém instruções detalhadas para fazer deploy do Gianna em diferentes ambientes de produção.

## Índice

1. [Preparação para Deploy](#preparação-para-deploy)
2. [Deploy Local/Desenvolvimento](#deploy-localdesenvolvimento)
3. [Deploy em Servidor](#deploy-em-servidor)
4. [Deploy com Docker](#deploy-com-docker)
5. [Deploy em Cloud](#deploy-em-cloud)
6. [Monitoramento e Manutenção](#monitoramento-e-manutenção)

## Preparação para Deploy

### Checklist Pré-Deploy

- [ ] Todas as chaves de API configuradas
- [ ] Testes passando (`invoke ci`)
- [ ] Configurações de produção validadas
- [ ] Backup da base de dados atual
- [ ] Monitoring configurado
- [ ] Logs configurados adequadamente

### Configurações de Produção

Crie arquivo `.env.production`:

```env
# Ambiente
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# APIs (use chaves de produção)
OPENAI_API_KEY=sua_chave_producao
GOOGLE_API_KEY=sua_chave_producao
ELEVEN_LABS_API_KEY=sua_chave_producao

# Configurações otimizadas
LLM_DEFAULT_MODEL=gpt4
TTS_DEFAULT_TYPE=google
CACHE_ENABLED=true
PERFORMANCE_MONITORING=true

# Segurança
API_RATE_LIMIT=100
MAX_CONCURRENT_SESSIONS=50
SESSION_TIMEOUT=3600

# Base de dados
DATABASE_URL=sqlite:///data/gianna_production.db
BACKUP_ENABLED=true
BACKUP_INTERVAL=3600

# Audio
AUDIO_QUALITY=high
AUDIO_CACHE_SIZE=1000
MAX_AUDIO_DURATION=300
```

## Deploy Local/Desenvolvimento

### Setup Básico

```bash
# Clone e configuração
git clone <repository-url>
cd gianna

# Ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate     # Windows

# Dependências
pip install poetry
poetry install

# Configuração
cp .env.example .env.development
# Edite .env.development com suas configurações

# Inicialização
python main.py
```

### Executando com Hot Reload

```bash
# Para desenvolvimento com recarga automática
pip install watchdog
python -m watchdog.auto_restart --patterns="*.py" --directory=gianna python main.py
```

## Deploy em Servidor

### Servidor Ubuntu/Debian

```bash
# 1. Preparar servidor
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y portaudio19-dev ffmpeg git

# 2. Criar usuário para aplicação
sudo useradd -m -s /bin/bash gianna
sudo su - gianna

# 3. Configurar aplicação
git clone <repository-url>
cd gianna
python3 -m venv venv
source venv/bin/activate
pip install poetry
poetry install --no-dev

# 4. Configurar ambiente
cp .env.example .env.production
# Editar .env.production

# 5. Criar diretórios necessários
mkdir -p data logs backups
chmod 755 data logs backups
```

### Systemd Service

```ini
# /etc/systemd/system/gianna.service
[Unit]
Description=Gianna Voice Assistant
After=network.target

[Service]
Type=simple
User=gianna
WorkingDirectory=/home/gianna/gianna
Environment=PATH=/home/gianna/gianna/venv/bin
ExecStart=/home/gianna/gianna/venv/bin/python main.py
Restart=always
RestartSec=10

# Logs
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=gianna

# Limites de recursos
MemoryMax=2G
CPUQuota=80%

[Install]
WantedBy=multi-user.target
```

```bash
# Ativar serviço
sudo systemctl daemon-reload
sudo systemctl enable gianna
sudo systemctl start gianna
sudo systemctl status gianna
```

### Nginx Proxy (Opcional)

```nginx
# /etc/nginx/sites-available/gianna
server {
    listen 80;
    server_name sua-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (se necessário)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # Logs
    access_log /var/log/nginx/gianna-access.log;
    error_log /var/log/nginx/gianna-error.log;
}
```

## Deploy com Docker

### Dockerfile de Produção

```dockerfile
FROM python:3.11-slim as base

# Metadados
LABEL maintainer="seu-email@exemplo.com"
LABEL version="1.0"
LABEL description="Gianna Voice Assistant"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VENV_IN_PROJECT=1

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Usuário não-root
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Dependências Python
COPY --chown=app:app pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

# Código da aplicação
COPY --chown=app:app . .

# Estrutura de diretórios
RUN mkdir -p data logs backups

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from gianna.optimization.monitoring import SystemMonitor; print(SystemMonitor().check_health())"

# Porta
EXPOSE 8000

# Comando padrão
CMD ["python", "main.py"]
```

### Docker Compose Completo

```yaml
# docker-compose.yml
version: '3.8'

services:
  gianna:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: gianna-app
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=sqlite:///data/gianna.db
    volumes:
      - ./data:/home/app/data
      - ./logs:/home/app/logs
      - ./backups:/home/app/backups
    ports:
      - "8000:8000"
    restart: unless-stopped
    depends_on:
      - redis
    networks:
      - gianna-network

  redis:
    image: redis:7-alpine
    container_name: gianna-redis
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - gianna-network
    command: redis-server --appendonly yes

  nginx:
    image: nginx:alpine
    container_name: gianna-nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "80:80"
      - "443:443"
    restart: unless-stopped
    depends_on:
      - gianna
    networks:
      - gianna-network

volumes:
  redis_data:

networks:
  gianna-network:
    driver: bridge
```

### Deploy com Docker

```bash
# Build e deploy
docker-compose up --build -d

# Verificar status
docker-compose ps
docker-compose logs -f gianna

# Backup
docker-compose exec gianna python scripts/backup.py

# Atualizações
git pull
docker-compose build --no-cache
docker-compose up -d
```

## Deploy em Cloud

### AWS EC2

```bash
# 1. Criar instância EC2
# - Ubuntu 22.04 LTS
# - t3.medium (2 vCPU, 4 GB RAM)
# - 20 GB SSD
# - Security Group: 22 (SSH), 80 (HTTP), 443 (HTTPS)

# 2. Conectar e configurar
ssh -i sua-chave.pem ubuntu@ip-da-instancia

# 3. Instalar Docker
sudo apt update
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker ubuntu
sudo systemctl enable docker

# 4. Deploy da aplicação
git clone <repository-url>
cd gianna
cp .env.example .env.production
# Configurar .env.production

# 5. Executar
docker-compose up -d
```

### Google Cloud Platform

```bash
# 1. Configurar GCP CLI
gcloud auth login
gcloud config set project seu-projeto-id

# 2. Criar VM
gcloud compute instances create gianna-vm \
    --zone=us-central1-a \
    --machine-type=e2-medium \
    --boot-disk-size=20GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --tags=http-server,https-server

# 3. Configurar firewall
gcloud compute firewall-rules create allow-gianna \
    --allow tcp:8000 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow Gianna app"

# 4. Conectar e configurar
gcloud compute ssh gianna-vm --zone=us-central1-a

# Seguir passos de configuração Ubuntu/Debian
```

### Azure Container Instances

```bash
# 1. Login no Azure
az login

# 2. Criar resource group
az group create --name gianna-rg --location eastus

# 3. Build e push da imagem
az acr create --resource-group gianna-rg --name giannaregistry --sku Basic
az acr login --name giannaregistry

docker build -t giannaregistry.azurecr.io/gianna:latest .
docker push giannaregistry.azurecr.io/gianna:latest

# 4. Deploy container
az container create \
    --resource-group gianna-rg \
    --name gianna-container \
    --image giannaregistry.azurecr.io/gianna:latest \
    --cpu 2 \
    --memory 4 \
    --ports 8000 \
    --dns-name-label gianna-app \
    --environment-variables \
        ENVIRONMENT=production \
        LOG_LEVEL=INFO
```

### Kubernetes

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gianna-deployment
  labels:
    app: gianna
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gianna
  template:
    metadata:
      labels:
        app: gianna
    spec:
      containers:
      - name: gianna
        image: gianna:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: gianna-service
spec:
  selector:
    app: gianna
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

```bash
# Deploy no Kubernetes
kubectl apply -f k8s-deployment.yml
kubectl get pods -l app=gianna
kubectl get services
```

## Monitoramento e Manutenção

### Logs Centralizados

```python
# scripts/log_aggregator.py
import logging
from logging.handlers import RotatingFileHandler
import json

def setup_production_logging():
    """Configurar logging para produção."""

    # Formatter JSON
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                'timestamp': self.formatTime(record, self.datefmt),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            if record.exc_info:
                log_obj['exception'] = self.formatException(record.exc_info)
            return json.dumps(log_obj)

    # Configurar handlers
    file_handler = RotatingFileHandler(
        'logs/gianna.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())

    # Logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
```

### Monitoramento de Saúde

```python
# scripts/health_monitor.py
import requests
import smtplib
from email.message import EmailMessage
import time

class HealthMonitor:
    def __init__(self, config):
        self.config = config
        self.last_alert_time = 0

    def check_health(self):
        """Verificar saúde da aplicação."""
        try:
            response = requests.get(
                f"{self.config['app_url']}/health",
                timeout=30
            )
            return response.status_code == 200
        except:
            return False

    def send_alert(self, message):
        """Enviar alerta por email."""
        current_time = time.time()

        # Rate limiting - máximo 1 alerta por 5 minutos
        if current_time - self.last_alert_time < 300:
            return

        msg = EmailMessage()
        msg['Subject'] = 'Gianna Health Alert'
        msg['From'] = self.config['smtp_from']
        msg['To'] = self.config['alert_email']
        msg.set_content(message)

        with smtplib.SMTP(self.config['smtp_server'], 587) as server:
            server.starttls()
            server.login(self.config['smtp_user'], self.config['smtp_password'])
            server.send_message(msg)

        self.last_alert_time = current_time

    def monitor_loop(self):
        """Loop principal de monitoramento."""
        while True:
            if not self.check_health():
                self.send_alert("Gianna application is not responding!")

            time.sleep(60)  # Verificar a cada minuto

if __name__ == "__main__":
    config = {
        'app_url': 'http://localhost:8000',
        'smtp_server': 'smtp.gmail.com',
        'smtp_user': 'seu-email@gmail.com',
        'smtp_password': 'sua-senha-app',
        'smtp_from': 'monitoramento@empresa.com',
        'alert_email': 'admin@empresa.com'
    }

    monitor = HealthMonitor(config)
    monitor.monitor_loop()
```

### Backup Automatizado

```python
# scripts/backup.py
import sqlite3
import shutil
import datetime
import os
import boto3  # Para AWS S3
from pathlib import Path

class BackupManager:
    def __init__(self, config):
        self.config = config

    def backup_database(self):
        """Backup da base de dados SQLite."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"gianna_backup_{timestamp}.db"
        backup_path = Path("backups") / backup_name

        # Backup local
        shutil.copy2(self.config['database_path'], backup_path)

        # Upload para S3 (opcional)
        if self.config.get('s3_enabled'):
            self.upload_to_s3(backup_path, backup_name)

        # Limpar backups antigos (manter últimos 7 dias)
        self.cleanup_old_backups()

        return backup_path

    def upload_to_s3(self, local_path, s3_key):
        """Upload de backup para AWS S3."""
        s3 = boto3.client('s3')
        bucket = self.config['s3_bucket']

        s3.upload_file(str(local_path), bucket, f"gianna-backups/{s3_key}")

    def cleanup_old_backups(self):
        """Remover backups antigos."""
        backup_dir = Path("backups")
        cutoff = datetime.datetime.now() - datetime.timedelta(days=7)

        for backup_file in backup_dir.glob("gianna_backup_*.db"):
            if backup_file.stat().st_mtime < cutoff.timestamp():
                backup_file.unlink()

# Agendar backup diário
if __name__ == "__main__":
    config = {
        'database_path': 'data/gianna_state.db',
        's3_enabled': True,
        's3_bucket': 'meu-bucket-backup'
    }

    backup_manager = BackupManager(config)
    backup_path = backup_manager.backup_database()
    print(f"Backup criado: {backup_path}")
```

### Script de Deploy Automatizado

```bash
#!/bin/bash
# scripts/deploy.sh

set -e  # Exit on error

echo "🚀 Iniciando deploy do Gianna..."

# Configurações
APP_DIR="/home/gianna/gianna"
BACKUP_DIR="/home/gianna/backups"
SERVICE_NAME="gianna"

# Função de rollback
rollback() {
    echo "❌ Deploy falhou! Realizando rollback..."
    cd $APP_DIR
    git checkout HEAD~1
    sudo systemctl restart $SERVICE_NAME
    echo "✅ Rollback concluído"
    exit 1
}

# Trap para rollback em caso de erro
trap rollback ERR

echo "📋 Verificando pré-requisitos..."
# Verificar se serviço está rodando
if ! systemctl is-active --quiet $SERVICE_NAME; then
    echo "❌ Serviço não está ativo"
    exit 1
fi

echo "💾 Criando backup..."
cd $APP_DIR
python scripts/backup.py

echo "📥 Atualizando código..."
git fetch origin
git checkout main
git pull origin main

echo "📦 Atualizando dependências..."
source venv/bin/activate
poetry install --no-dev

echo "✅ Executando testes..."
python -m pytest tests/ --timeout=30

echo "🔄 Reiniciando serviço..."
sudo systemctl restart $SERVICE_NAME

echo "⏳ Aguardando inicialização..."
sleep 10

echo "🔍 Verificando saúde..."
if curl -f http://localhost:8000/health; then
    echo "✅ Deploy concluído com sucesso!"
else
    echo "❌ Verificação de saúde falhou"
    rollback
fi

echo "📊 Status final:"
sudo systemctl status $SERVICE_NAME --no-pager
```

### Cron Jobs para Manutenção

```bash
# Adicionar ao crontab: crontab -e

# Backup diário às 2:00
0 2 * * * cd /home/gianna/gianna && python scripts/backup.py

# Limpeza de logs às 3:00
0 3 * * * find /home/gianna/gianna/logs -name "*.log.*" -mtime +7 -delete

# Verificação de saúde a cada 5 minutos
*/5 * * * * curl -f http://localhost:8000/health > /dev/null 2>&1 || echo "Health check failed" | mail -s "Gianna Down" admin@empresa.com

# Relatório semanal aos domingos às 9:00
0 9 * * 0 cd /home/gianna/gianna && python scripts/weekly_report.py | mail -s "Gianna Weekly Report" admin@empresa.com
```

## Troubleshooting

### Problemas Comuns

1. **Serviço não inicia**
   ```bash
   sudo systemctl status gianna
   sudo journalctl -u gianna -f
   ```

2. **Alto uso de memória**
   ```bash
   # Verificar processo
   htop

   # Reiniciar com limite
   sudo systemctl edit gianna
   # Adicionar: MemoryMax=1G
   ```

3. **API keys inválidas**
   ```bash
   # Verificar configuração
   python -c "from gianna.assistants.models.factory_method import get_chain_instance; print('OK')"
   ```

4. **Problemas de áudio**
   ```bash
   # Verificar dispositivos
   python -c "import pyaudio; p = pyaudio.PyAudio(); print([p.get_device_info_by_index(i) for i in range(p.get_device_count())])"
   ```

### Logs Úteis

```bash
# Logs da aplicação
tail -f logs/gianna.log

# Logs do sistema
sudo journalctl -u gianna -f

# Logs do Docker
docker-compose logs -f gianna

# Verificar performance
htop
iostat -x 1
```

Para mais informações sobre troubleshooting, consulte [FAQ](../faq.md).
