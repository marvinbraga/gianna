#!/bin/bash
# Backup and disaster recovery script for Gianna

set -e
set -o pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${BACKUP_DIR:-/app/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="gianna_backup_${TIMESTAMP}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

# AWS S3 Configuration (optional)
AWS_BUCKET="${AWS_BACKUP_BUCKET:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Create backup directory
create_backup_dir() {
    local backup_path="$BACKUP_DIR/$BACKUP_NAME"
    mkdir -p "$backup_path"
    echo "$backup_path"
}

# Backup databases
backup_databases() {
    local backup_path="$1"
    local db_dir="$backup_path/databases"
    mkdir -p "$db_dir"

    log_info "Backing up databases..."

    # SQLite databases
    local db_files=("gianna_state.db" "gianna_optimization.db")
    for db_file in "${db_files[@]}"; do
        if [[ -f "$PROJECT_ROOT/$db_file" ]]; then
            log_info "Backing up $db_file..."
            sqlite3 "$PROJECT_ROOT/$db_file" ".backup '$db_dir/$db_file'"

            # Create schema dump for reference
            sqlite3 "$PROJECT_ROOT/$db_file" ".schema" > "$db_dir/${db_file}.schema"

            # Get database stats
            sqlite3 "$PROJECT_ROOT/$db_file" "SELECT 'Tables: ' || COUNT(*) FROM sqlite_master WHERE type='table';" > "$db_dir/${db_file}.stats"
            sqlite3 "$PROJECT_ROOT/$db_file" "PRAGMA page_count;" >> "$db_dir/${db_file}.stats"
            sqlite3 "$PROJECT_ROOT/$db_file" "PRAGMA page_size;" >> "$db_dir/${db_file}.stats"
        else
            log_warn "$db_file not found, skipping..."
        fi
    done

    log_success "Database backup completed"
}

# Backup configuration files
backup_config() {
    local backup_path="$1"
    local config_dir="$backup_path/config"

    log_info "Backing up configuration files..."

    # Copy configuration directories
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        cp -r "$PROJECT_ROOT/config" "$config_dir"
    fi

    # Copy environment files
    for env_file in .env .env.production .env.staging; do
        if [[ -f "$PROJECT_ROOT/$env_file" ]]; then
            cp "$PROJECT_ROOT/$env_file" "$config_dir/"
        fi
    done

    # Copy Docker configurations
    if [[ -d "$PROJECT_ROOT/docker" ]]; then
        cp -r "$PROJECT_ROOT/docker" "$backup_path/"
    fi

    # Copy nginx configurations
    if [[ -d "$PROJECT_ROOT/nginx" ]]; then
        cp -r "$PROJECT_ROOT/nginx" "$backup_path/"
    fi

    # Copy monitoring configurations
    if [[ -d "$PROJECT_ROOT/monitoring" ]]; then
        cp -r "$PROJECT_ROOT/monitoring" "$backup_path/"
    fi

    log_success "Configuration backup completed"
}

# Backup application data
backup_application_data() {
    local backup_path="$1"
    local data_dir="$backup_path/data"
    mkdir -p "$data_dir"

    log_info "Backing up application data..."

    # Application data directory
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        cp -r "$PROJECT_ROOT/data"/* "$data_dir/" 2>/dev/null || true
    fi

    # Cache directory (selective backup)
    if [[ -d "$PROJECT_ROOT/cache" ]]; then
        mkdir -p "$backup_path/cache"
        # Only backup important cache files, skip temporary ones
        find "$PROJECT_ROOT/cache" -name "*.meta" -exec cp {} "$backup_path/cache/" \;
    fi

    # Logs (last 7 days)
    if [[ -d "$PROJECT_ROOT/logs" ]]; then
        mkdir -p "$backup_path/logs"
        find "$PROJECT_ROOT/logs" -name "*.log" -mtime -7 -exec cp {} "$backup_path/logs/" \;
    fi

    # Audio resources
    if [[ -d "$PROJECT_ROOT/resources" ]]; then
        cp -r "$PROJECT_ROOT/resources" "$backup_path/"
    fi

    log_success "Application data backup completed"
}

# Backup secrets (encrypted)
backup_secrets() {
    local backup_path="$1"
    local secrets_dir="$backup_path/secrets"
    mkdir -p "$secrets_dir"

    log_info "Backing up secrets..."

    # Copy encrypted secrets
    if [[ -f "$PROJECT_ROOT/secrets/secrets.yaml" ]]; then
        cp "$PROJECT_ROOT/secrets/secrets.yaml" "$secrets_dir/"
    fi

    # Create secrets inventory (without actual values)
    if [[ -f "$PROJECT_ROOT/secrets/secrets.yaml" ]]; then
        python3 << EOF
import yaml
import json

try:
    with open('$PROJECT_ROOT/secrets/secrets.yaml', 'r') as f:
        secrets = yaml.safe_load(f) or {}

    def get_keys(obj, prefix=''):
        keys = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    keys.extend(get_keys(value, current_key))
                else:
                    keys.append(current_key)
        return keys

    secret_keys = get_keys(secrets)
    inventory = {
        'total_secrets': len(secret_keys),
        'secret_keys': secret_keys,
        'backup_time': '$TIMESTAMP'
    }

    with open('$secrets_dir/inventory.json', 'w') as f:
        json.dump(inventory, f, indent=2)

except Exception as e:
    print(f"Error creating secrets inventory: {e}")
EOF
    fi

    log_success "Secrets backup completed"
}

# Create backup metadata
create_backup_metadata() {
    local backup_path="$1"

    log_info "Creating backup metadata..."

    cat > "$backup_path/backup_info.json" << EOF
{
    "backup_name": "$BACKUP_NAME",
    "timestamp": "$TIMESTAMP",
    "date": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "version": "$(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(cd "$PROJECT_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
    "git_commit": "$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "backup_size": "$(du -sh "$backup_path" | cut -f1)",
    "hostname": "$(hostname)",
    "user": "$(whoami)",
    "environment": "${ENVIRONMENT:-production}"
}
EOF

    # Create manifest of all files
    find "$backup_path" -type f -exec ls -la {} \; > "$backup_path/manifest.txt"

    # Create checksums
    find "$backup_path" -type f -exec sha256sum {} \; > "$backup_path/checksums.sha256"

    log_success "Backup metadata created"
}

# Compress backup
compress_backup() {
    local backup_path="$1"
    local compressed_path="${backup_path}.tar.gz"

    log_info "Compressing backup..."

    cd "$BACKUP_DIR"
    tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"

    # Remove uncompressed directory
    rm -rf "$BACKUP_NAME"

    log_success "Backup compressed: ${compressed_path}"
    echo "$compressed_path"
}

# Upload to cloud storage
upload_to_cloud() {
    local backup_file="$1"

    if [[ -n "$AWS_BUCKET" ]]; then
        log_info "Uploading backup to AWS S3..."

        if command -v aws >/dev/null 2>&1; then
            aws s3 cp "$backup_file" "s3://$AWS_BUCKET/gianna-backups/" \
                --region "$AWS_REGION" \
                --storage-class STANDARD_IA

            log_success "Backup uploaded to S3"
        else
            log_warn "AWS CLI not found, skipping S3 upload"
        fi
    fi
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"

    log_info "Verifying backup integrity..."

    # Test archive integrity
    if tar -tzf "$backup_file" >/dev/null 2>&1; then
        log_success "Backup archive integrity verified"
    else
        log_error "Backup archive is corrupted!"
        return 1
    fi

    # Extract and verify checksums
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"

    tar -xzf "$backup_file"
    local extracted_dir=$(ls -1)

    if [[ -f "$extracted_dir/checksums.sha256" ]]; then
        cd "$extracted_dir"
        if sha256sum -c checksums.sha256 --quiet; then
            log_success "All file checksums verified"
        else
            log_error "Checksum verification failed!"
            cd /
            rm -rf "$temp_dir"
            return 1
        fi
    fi

    cd /
    rm -rf "$temp_dir"

    log_success "Backup verification completed"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups..."

    # Remove local backups older than retention period
    find "$BACKUP_DIR" -name "gianna_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete

    # Cleanup S3 backups if configured
    if [[ -n "$AWS_BUCKET" ]] && command -v aws >/dev/null 2>&1; then
        local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%Y-%m-%d)

        aws s3 ls "s3://$AWS_BUCKET/gianna-backups/" \
            | awk '{print $1" "$2" "$4}' \
            | while read -r date time filename; do
                if [[ "$date" < "$cutoff_date" ]]; then
                    aws s3 rm "s3://$AWS_BUCKET/gianna-backups/$filename"
                    log_info "Removed old S3 backup: $filename"
                fi
            done
    fi

    log_success "Cleanup completed"
}

# Restore from backup
restore_backup() {
    local backup_file="$1"

    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi

    log_warn "Starting restore process..."
    log_warn "This will overwrite existing data!"

    # Create restore directory
    local restore_dir=$(mktemp -d)
    cd "$restore_dir"

    # Extract backup
    log_info "Extracting backup..."
    tar -xzf "$backup_file"
    local backup_name=$(ls -1)

    # Verify backup before restore
    if [[ -f "$backup_name/checksums.sha256" ]]; then
        cd "$backup_name"
        if ! sha256sum -c checksums.sha256 --quiet; then
            log_error "Backup verification failed! Restore aborted."
            cd /
            rm -rf "$restore_dir"
            return 1
        fi
        cd "$restore_dir"
    fi

    # Stop services
    log_info "Stopping services..."
    cd "$PROJECT_ROOT"
    docker-compose -f docker/docker-compose.prod.yml down || true

    # Restore databases
    if [[ -d "$restore_dir/$backup_name/databases" ]]; then
        log_info "Restoring databases..."
        for db_file in "$restore_dir/$backup_name/databases"/*.db; do
            if [[ -f "$db_file" ]]; then
                local db_name=$(basename "$db_file")
                cp "$db_file" "$PROJECT_ROOT/$db_name"
                log_info "Restored $db_name"
            fi
        done
    fi

    # Restore configuration
    if [[ -d "$restore_dir/$backup_name/config" ]]; then
        log_info "Restoring configuration..."
        cp -r "$restore_dir/$backup_name/config"/* "$PROJECT_ROOT/"
    fi

    # Restore data
    if [[ -d "$restore_dir/$backup_name/data" ]]; then
        log_info "Restoring application data..."
        mkdir -p "$PROJECT_ROOT/data"
        cp -r "$restore_dir/$backup_name/data"/* "$PROJECT_ROOT/data/"
    fi

    # Restore secrets
    if [[ -d "$restore_dir/$backup_name/secrets" ]]; then
        log_info "Restoring secrets..."
        mkdir -p "$PROJECT_ROOT/secrets"
        cp -r "$restore_dir/$backup_name/secrets"/* "$PROJECT_ROOT/secrets/"
    fi

    # Cleanup restore directory
    cd /
    rm -rf "$restore_dir"

    # Restart services
    log_info "Starting services..."
    cd "$PROJECT_ROOT"
    docker-compose -f docker/docker-compose.prod.yml up -d

    log_success "Restore completed successfully!"
}

# Test restore process
test_restore() {
    local backup_file="$1"

    log_info "Testing restore process (dry run)..."

    # Create test directory
    local test_dir=$(mktemp -d)
    cd "$test_dir"

    # Extract and verify
    if tar -xzf "$backup_file" >/dev/null 2>&1; then
        local backup_name=$(ls -1)

        # Check essential components
        local components=("databases" "config" "backup_info.json" "checksums.sha256")
        local missing=()

        for component in "${components[@]}"; do
            if [[ ! -e "$backup_name/$component" ]]; then
                missing+=("$component")
            fi
        done

        if [[ ${#missing[@]} -eq 0 ]]; then
            log_success "Restore test passed - all components present"
        else
            log_warn "Restore test warnings - missing components: ${missing[*]}"
        fi
    else
        log_error "Restore test failed - cannot extract backup"
        cd /
        rm -rf "$test_dir"
        return 1
    fi

    cd /
    rm -rf "$test_dir"

    log_success "Restore test completed"
}

# List available backups
list_backups() {
    log_info "Available local backups:"

    if [[ -d "$BACKUP_DIR" ]]; then
        ls -la "$BACKUP_DIR"/gianna_backup_*.tar.gz 2>/dev/null | while read -r line; do
            echo "  $line"
        done
    else
        log_warn "Backup directory not found: $BACKUP_DIR"
    fi

    # List S3 backups if configured
    if [[ -n "$AWS_BUCKET" ]] && command -v aws >/dev/null 2>&1; then
        log_info "Available S3 backups:"
        aws s3 ls "s3://$AWS_BUCKET/gianna-backups/" | while read -r line; do
            echo "  $line"
        done
    fi
}

# Show help
show_help() {
    cat << EOF
Gianna Backup and Recovery Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    backup              Create full backup (default)
    restore FILE        Restore from backup file
    test FILE           Test restore process (dry run)
    list                List available backups
    cleanup             Remove old backups
    verify FILE         Verify backup integrity

Options:
    --retention DAYS    Backup retention period (default: 30)
    --upload            Upload to cloud storage
    --no-compress       Skip compression
    --help             Show this help

Environment Variables:
    BACKUP_DIR          Backup directory (default: /app/backups)
    AWS_BACKUP_BUCKET   S3 bucket for cloud backup
    AWS_REGION          AWS region (default: us-east-1)
    RETENTION_DAYS      Backup retention days (default: 30)

Examples:
    $0                          # Create backup
    $0 backup --upload          # Create and upload backup
    $0 restore backup.tar.gz    # Restore from backup
    $0 test backup.tar.gz       # Test restore process
    $0 cleanup                  # Remove old backups

EOF
}

# Main backup function
create_backup() {
    local upload_cloud=false
    local skip_compress=false

    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --upload)
                upload_cloud=true
                shift
                ;;
            --no-compress)
                skip_compress=true
                shift
                ;;
            --retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                return 1
                ;;
        esac
    done

    log_info "Starting Gianna backup process..."

    # Create backup directory
    local backup_path=$(create_backup_dir)

    # Perform backup components
    backup_databases "$backup_path"
    backup_config "$backup_path"
    backup_application_data "$backup_path"
    backup_secrets "$backup_path"
    create_backup_metadata "$backup_path"

    # Compress backup
    local final_backup
    if [[ "$skip_compress" == "false" ]]; then
        final_backup=$(compress_backup "$backup_path")
    else
        final_backup="$backup_path"
    fi

    # Verify backup
    if [[ "$skip_compress" == "false" ]]; then
        verify_backup "$final_backup"
    fi

    # Upload to cloud if requested
    if [[ "$upload_cloud" == "true" ]]; then
        upload_to_cloud "$final_backup"
    fi

    # Cleanup old backups
    cleanup_old_backups

    log_success "Backup completed successfully!"
    log_info "Backup location: $final_backup"
    log_info "Backup size: $(du -sh "$final_backup" | cut -f1)"
}

# Main script logic
main() {
    cd "$PROJECT_ROOT"

    # Parse command
    case "${1:-backup}" in
        backup)
            shift
            create_backup "$@"
            ;;
        restore)
            if [[ -z "$2" ]]; then
                log_error "Restore requires backup file path"
                show_help
                exit 1
            fi
            restore_backup "$2"
            ;;
        test)
            if [[ -z "$2" ]]; then
                log_error "Test requires backup file path"
                show_help
                exit 1
            fi
            test_restore "$2"
            ;;
        list)
            list_backups
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        verify)
            if [[ -z "$2" ]]; then
                log_error "Verify requires backup file path"
                show_help
                exit 1
            fi
            verify_backup "$2"
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
