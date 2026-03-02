# Use Bash to run this script.
#!/usr/bin/env bash
# Exit on error (-e), unset variable usage (-u), and pipeline failures (pipefail).
set -eu pipefail

# Compose service name defined in docker-compose.yml.
SERVICE_NAME="rtdetr-container"
# Fixed Docker container name used by this service.
CONTAINER_NAME="rtdetr-v2-trt"

# Remove any existing container with the same name to avoid name-conflict errors.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
# Build or rebuild the image for the selected compose service.
docker compose build "$SERVICE_NAME"
# Start the service in detached mode, force recreation, and remove orphan containers.
docker compose up -d --force-recreate --remove-orphans "$SERVICE_NAME"