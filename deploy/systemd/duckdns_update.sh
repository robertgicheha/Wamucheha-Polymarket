#!/usr/bin/env bash
#
# Point the DuckDNS record at this host.
#
# DuckDNS is free but does not track your IP for you. If the VPS gets a new
# address the domain silently stops resolving and the dashboard disappears —
# so this runs on a timer.
#
# Requires DUCKDNS_TOKEN in /etc/duckdns.env (mode 600). DuckDNS tokens are
# per-account, found at https://www.duckdns.org/ after logging in.
#
#   https://www.duckdns.org/update?domains=<domain>&token=<token>&ip=
#
# An empty ip= tells DuckDNS to use the caller's source address, which is
# exactly what we want and avoids needing to know the public IP.
set -euo pipefail

ENV_FILE=/etc/duckdns.env
DOMAIN_FILE=/etc/duckdns.domain
UPDATE_URL="https://www.duckdns.org/update"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

if [[ ! -r "$ENV_FILE" ]]; then
    log "ERROR: $ENV_FILE is missing or unreadable; cannot update DuckDNS"
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ -z "${DUCKDNS_TOKEN:-}" ]]; then
    log "ERROR: DUCKDNS_TOKEN is not set in $ENV_FILE"
    exit 1
fi
if [[ ! -r "$DOMAIN_FILE" ]]; then
    log "ERROR: $DOMAIN_FILE is missing; cannot update DuckDNS"
    exit 1
fi
# shellcheck disable=SC1090
source "$DOMAIN_FILE"

if [[ -z "${DUCKDNS_DOMAIN:-}" ]]; then
    log "ERROR: DUCKDNS_DOMAIN is not set in $DOMAIN_FILE"
    exit 1
fi

# DuckDNS answers with OK or a diagnostic token on a single line.
response=$(curl -fsS --max-time 20 \
    "${UPDATE_URL}?domains=${DUCKDNS_DOMAIN}&token=${DUCKDNS_TOKEN}&ip=&verbose=true" 2>&1) || {
    log "ERROR: DuckDNS update request failed"
    exit 1
}

if [[ "$response" == OK* ]]; then
    # DuckDNS echoes the IP it just recorded on the second line of verbose
    # output; logging it makes a misconfigured record obvious at a glance.
    recorded=$(printf '%s\n' "$response" | sed -n '2p')
    log "DuckDNS ${DUCKDNS_DOMAIN}.duckdns.org -> ${recorded:-ok}"
    exit 0
fi

log "ERROR: DuckDNS rejected the update: ${response}"
exit 1
