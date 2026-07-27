#!/bin/sh
# backup.sh — Periodic SQLite backup sidecar
#
# Runs inside the backup container. Copies the SQLite DB file
# to a timestamped backup every $BACKUP_INTERVAL_HOURS hours.
# Old backups older than $BACKUP_RETAIN_DAYS days are pruned.

set -e

DATA_DIR="${DATA_DIR:-/data}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
INTERVAL="${BACKUP_INTERVAL:-6}"
RETAIN_DAYS="${BACKUP_RETAIN_DAYS:-30}"

echo "Backup sidecar starting"
echo "  Source:  $DATA_DIR"
echo "  Dest:    $BACKUP_DIR"
echo "  Every:   ${INTERVAL}h"
echo "  Retain:  ${RETAIN_DAYS} days"

mkdir -p "$BACKUP_DIR"

do_backup() {
    TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

    for db_file in "$DATA_DIR"/*.db; do
        [ -f "$db_file" ] || continue
        BASENAME=$(basename "$db_file" .db)
        DEST="$BACKUP_DIR/${BASENAME}_${TIMESTAMP}.db"

        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Backing up $db_file -> $DEST"

        # Use SQLite .backup for a consistent snapshot (no locking issues)
        # Falls back to plain cp if sqlite3 CLI isn't available
        if command -v sqlite3 >/dev/null 2>&1; then
            sqlite3 "$db_file" ".backup '$DEST'"
        else
            cp "$db_file" "$DEST"
        fi

        # Compress the backup
        if command -v gzip >/dev/null 2>&1; then
            gzip "$DEST"
            DEST="${DEST}.gz"
        fi

        SIZE=$(du -h "$DEST" 2>/dev/null | cut -f1)
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Backup complete: $DEST ($SIZE)"
    done
}

prune_old() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pruning backups older than ${RETAIN_DAYS} days"
    find "$BACKUP_DIR" -name "*.db" -o -name "*.db.gz" | while read -r f; do
        if [ -f "$f" ]; then
            AGE=$(( ($(date +%s) - $(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null)) / 86400 ))
            if [ "$AGE" -gt "$RETAIN_DAYS" ]; then
                echo "  Removing: $f (${AGE} days old)"
                rm -f "$f"
            fi
        fi
    done
}

# Initial backup on start
do_backup

# Loop
while true; do
    sleep_seconds=$((INTERVAL * 3600))
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sleeping ${sleep_seconds}s until next backup"
    sleep "$sleep_seconds"
    do_backup
    prune_old
done
