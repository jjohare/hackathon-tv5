#!/bin/bash
set -euo pipefail

# Milvus Backup and Restore Script
# Uses Milvus Backup tool for production-grade backups

NAMESPACE="milvus-prod"
BACKUP_BUCKET="gs://gmc-o-milvus-backups"  # Change to your bucket
BACKUP_NAME="${BACKUP_NAME:-backup-$(date +%Y%m%d-%H%M%S)}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Backup Milvus data
backup() {
    log_info "Starting Milvus backup: ${BACKUP_NAME}"

    # Deploy Milvus Backup pod
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: milvus-backup
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  containers:
  - name: backup
    image: milvusdb/milvus-backup:v0.4.0
    command:
    - /bin/sh
    - -c
    - |
      milvus-backup create \
        --milvus-address=milvus-proxy:19530 \
        --backup-name=${BACKUP_NAME} \
        --minio-address=minio:9000 \
        --minio-bucket=milvus-bucket
    env:
    - name: MINIO_ACCESS_KEY
      value: minioadmin
    - name: MINIO_SECRET_KEY
      value: minioadmin123
EOF

    # Wait for backup to complete
    kubectl wait --for=condition=Ready pod/milvus-backup -n ${NAMESPACE} --timeout=5m
    kubectl logs -f milvus-backup -n ${NAMESPACE}

    # Export to GCS
    log_info "Exporting backup to ${BACKUP_BUCKET}..."
    kubectl run -n ${NAMESPACE} backup-export --image=gcr.io/google.com/cloudsdktool/cloud-sdk:alpine --rm -it --restart=Never -- \
        sh -c "gsutil -m rsync -r gs://milvus-bucket/backups/${BACKUP_NAME} ${BACKUP_BUCKET}/${BACKUP_NAME}"

    kubectl delete pod milvus-backup -n ${NAMESPACE}
    log_info "Backup complete: ${BACKUP_NAME}"
}

# Restore from backup
restore() {
    if [ $# -eq 0 ]; then
        log_error "Usage: $0 restore <backup-name>"
        exit 1
    fi

    RESTORE_NAME=$1
    log_warn "Restoring from backup: ${RESTORE_NAME}"
    log_warn "This will replace current data. Press Ctrl+C to cancel..."
    sleep 10

    # Import from GCS
    log_info "Importing backup from ${BACKUP_BUCKET}..."
    kubectl run -n ${NAMESPACE} backup-import --image=gcr.io/google.com/cloudsdktool/cloud-sdk:alpine --rm -it --restart=Never -- \
        sh -c "gsutil -m rsync -r ${BACKUP_BUCKET}/${RESTORE_NAME} gs://milvus-bucket/backups/${RESTORE_NAME}"

    # Deploy restore pod
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: milvus-restore
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  containers:
  - name: restore
    image: milvusdb/milvus-backup:v0.4.0
    command:
    - /bin/sh
    - -c
    - |
      milvus-backup restore \
        --milvus-address=milvus-proxy:19530 \
        --backup-name=${RESTORE_NAME} \
        --minio-address=minio:9000 \
        --minio-bucket=milvus-bucket
    env:
    - name: MINIO_ACCESS_KEY
      value: minioadmin
    - name: MINIO_SECRET_KEY
      value: minioadmin123
EOF

    kubectl wait --for=condition=Ready pod/milvus-restore -n ${NAMESPACE} --timeout=5m
    kubectl logs -f milvus-restore -n ${NAMESPACE}
    kubectl delete pod milvus-restore -n ${NAMESPACE}

    log_info "Restore complete"
}

# List backups
list_backups() {
    log_info "Available backups in ${BACKUP_BUCKET}:"
    gsutil ls "${BACKUP_BUCKET}/" || log_error "Failed to list backups"
}

# Main
case "${1:-}" in
    backup)
        backup
        ;;
    restore)
        restore "${2:-}"
        ;;
    list)
        list_backups
        ;;
    *)
        echo "Usage: $0 {backup|restore <name>|list}"
        exit 1
        ;;
esac
