# Environment Variables Configuration
# Copy this file to .env and update with your values

# Application Settings
APP_ENV=development
DEBUG_MODE=false
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here-change-in-production

# Database Configuration
DATABASE_URL=sqlite:///scheduling_system.db
DATABASE_POOL_SIZE=10
DATABASE_BACKUP_ENABLED=true
DATABASE_BACKUP_INTERVAL_HOURS=24

# PostgreSQL Configuration (if using PostgreSQL instead of SQLite)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=scheduling_db
POSTGRES_USER=scheduler_user
POSTGRES_PASSWORD=changeme

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60

# Authentication
JWT_SECRET_KEY=your-jwt-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
API_KEY=your-api-key-here

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=noreply@sportsdancescheduling.com
EMAIL_ADMIN=admin@sportsdancescheduling.com

# Cloud Storage (Optional)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
S3_BUCKET_NAME=sports-dance-scheduling

# Monitoring and Analytics
ENABLE_MONITORING=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_PASSWORD=admin

# Wandb Configuration
WANDB_API_KEY=
WANDB_PROJECT=sports-dance-scheduling
WANDB_ENTITY=
WANDB_MODE=online

# TensorBoard
TENSORBOARD_ENABLED=false
TENSORBOARD_LOG_DIR=runs

# Model Configuration
MODEL_PATH=models/production_model.pt
MODEL_CACHE_DIR=.cache/models
ONNX_EXPORT_ENABLED=false
USE_GPU=auto
CUDA_VISIBLE_DEVICES=0

# Performance Settings
MAX_WORKERS=10
CACHE_SIZE_MB=1024
ENABLE_PROFILING=false
MEMORY_OPTIMIZATION=true

# Scheduling Constraints
MAX_CONSECUTIVE_CLASSES=3
MIN_BREAK_MINUTES=15
MAX_DAILY_HOURS=8
MAX_WEEKLY_HOURS=40
MAX_CLASS_SIZE=30
MIN_CLASS_SIZE=5

# Feature Flags
FEATURE_ADVANCED_OPTIMIZATION=true
FEATURE_REAL_TIME_UPDATES=false
FEATURE_AUTO_SCHEDULING=true
FEATURE_CONFLICT_RESOLUTION=true
FEATURE_WORKLOAD_BALANCING=true

# External Services
CALENDAR_SYNC_ENABLED=false
GOOGLE_CALENDAR_API_KEY=
OUTLOOK_INTEGRATION_ENABLED=false
SLACK_WEBHOOK_URL=
DISCORD_WEBHOOK_URL=

# Development Settings
HOT_RELOAD=false
MOCK_DATA=false
TEST_MODE=false
VERBOSE_LOGGING=false

# Docker Settings
DOCKER_REGISTRY=docker.io
DOCKER_USERNAME=
DOCKER_PASSWORD=

# Kubernetes Settings
K8S_NAMESPACE=default
K8S_CONTEXT=production

# Sentry Error Tracking
SENTRY_DSN=
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=1.0