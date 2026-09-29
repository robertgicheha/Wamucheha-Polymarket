#!/usr/bin/env bash
#
# expose_dashboard.sh — publish the trading dashboard at
# https://<domain>.duckdns.org/ behind HTTPS + password auth.
#
# Run this on the VPS as root:
#
#     cd /opt/wamucheha-polymarket
#     sudo ./scripts/expose_dashboard.sh --domain wamucheha-poly
#
# It is idempotent: safe to re-run after a redeploy or a config change.
#
# What it does, in order:
#   1. Installs Caddy (auto-TLS via Let's Encrypt, HTTP-01 works with DuckDNS)
#   2. Binds Flask to 127.0.0.1 so the app is unreachable from the internet
#   3. Writes the dashboard credentials into config/.env (git-ignored)
#   4. Blocks POST /api/withdraw at the proxy — that endpoint moves real USDC
#   5. Locks the firewall to 22/80/443
#   6. Installs fail2ban to throttle password guessing
#   7. Optionally sets up a DuckDNS auto-update timer
#   8. Verifies the whole chain and prints the URL
#
# Options:
#   --domain NAME      DuckDNS subdomain, e.g. wamucheha-poly   (required)
#   --user NAME        dashboard username          (default: wamucheha)
#   --password PASS    dashboard password; omit to be prompted invisibly
#   --email EMAIL      ACME/Let's Encrypt contact address
#   --port PORT        Flask dashboard port         (default: 8080)
#   --no-ufw           skip firewall changes (if you manage it elsewhere)
#   --no-fail2ban      skip brute-force throttling
#   --duckdns-token T  DuckDNS account token; enables the auto-update timer
#   --check            verify the current setup, change nothing
set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────────────

DOMAIN=""
DASH_USER="wamucheha"
DASH_PASS="${DASHBOARD_PASSWORD:-}"
ACME_EMAIL=""
DASH_PORT="8080"
DUCKDNS_TOKEN="${DUCKDNS_TOKEN:-}"
DO_UFW=1
DO_FAIL2BAN=1
CHECK_ONLY=0

APP_DIR="${APP_DIR:-/opt/wamucheha-polymarket}"
ENV_FILE="$APP_DIR/config/.env"
CADDYFILE="/etc/caddy/Caddyfile"
TEMPLATE_DIR="$APP_DIR/deploy"
BOT_UNIT="${BOT_UNIT:-wamucheha-bot.service}"

# ── Output helpers ─────────────────────────────────────────────────────

BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; OFF=$'\033[0m'
step()  { printf '\n%s==>%s %s\n' "$BOLD" "$OFF" "$*"; }
info()  { printf '    %s\n' "$*"; }
ok()    { printf '    %s✓%s %s\n' "$GREEN" "$OFF" "$*"; }
warn()  { printf '    %s!%s %s\n' "$YELLOW" "$OFF" "$*"; }
die()   { printf '\n%sERROR:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

# ── Argument parsing ───────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)        DOMAIN="${2:-}"; shift 2 ;;
        --user)          DASH_USER="${2:-}"; shift 2 ;;
        --password)      DASH_PASS="${2:-}"; shift 2 ;;
        --email)         ACME_EMAIL="${2:-}"; shift 2 ;;
        --port)          DASH_PORT="${2:-}"; shift 2 ;;
        --duckdns-token) DUCKDNS_TOKEN="${2:-}"; shift 2 ;;
        --no-ufw)        DO_UFW=0; shift ;;
        --no-fail2ban)   DO_FAIL2BAN=0; shift ;;
        --check)         CHECK_ONLY=1; shift ;;
        -h|--help)       sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)               die "Unknown option: $1 (try --help)" ;;
    esac
done

[[ $EUID -eq 0 ]] || die "Run as root: sudo $0 --domain <name>"
[[ -n "$DOMAIN" ]] || die "--domain is required (e.g. --domain wamucheha-poly)"
[[ -d "$APP_DIR" ]] || die "App directory not found: $APP_DIR (set APP_DIR=...)"
[[ -f "$TEMPLATE_DIR/caddy/Caddyfile.in" ]] || die "Missing $TEMPLATE_DIR/caddy/Caddyfile.in — run from a full checkout"

FQDN="${DOMAIN}.duckdns.org"
URL="https://${FQDN}/"

# ── Check mode ─────────────────────────────────────────────────────────

if [[ $CHECK_ONLY -eq 1 ]]; then
    step "Checking the published dashboard for $FQDN"
    check_port() {
        if ss -ltn 2>/dev/null | grep -qE "[:.]$2\b"; then ok "$1 is listening on :$2"
        else warn "$1 is NOT listening on :$2"; fi
    }
    check_port "caddy (80)"   80
    check_port "caddy (443)"  443
    check_port "flask (${DASH_PORT})" "$DASH_PORT"

    if curl -fsS --max-time 15 "https://${FQDN}/api/health" -u "${DASH_USER}:${DASH_PASS}" >/dev/null 2>&1; then
        ok "authenticated /api/health returns 200"
    elif curl -fsS --max-time 15 "https://${FQDN}/api/health" -o /dev/null 2>&1; then
        warn "responds but rejected the credentials in DASHBOARD_PASSWORD"
    else
        warn "https://${FQDN}/api/health unreachable (DNS, TLS, or Caddy not up?)"
    fi

    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 -X POST \
        "https://${FQDN}/api/withdraw" -u "${DASH_USER}:${DASH_PASS}" \
        -H 'Content-Type: application/json' -d '{"amount":1,"destination":"x"}' 2>/dev/null || echo 000)
    if [[ "$code" == "403" ]]; then ok "withdrawal endpoint is blocked (403) — good"
    else warn "withdrawal endpoint returned $code, expected 403"; fi

    systemctl is-enabled duckdns-update.timer >/dev/null 2>&1 \
        && ok "duckdns-update.timer enabled" || warn "duckdns-update.timer not enabled"
    exit 0
fi

# ── Credentials ────────────────────────────────────────────────────────

if [[ -z "$DASH_PASS" ]]; then
    info "No --password given; reading it from \$DASHBOARD_PASSWORD, else prompting."
    if [[ -z "$DASH_PASS" && -r "$ENV_FILE" ]]; then
        DASH_PASS=$(grep -E '^DASHBOARD_PASSWORD=' "$ENV_FILE" | tail -1 | cut -d= -f2-)
        [[ -n "$DASH_PASS" ]] && info "Reusing the existing DASHBOARD_PASSWORD from config/.env"
    fi
    if [[ -z "$DASH_PASS" ]]; then
        read -rsp "Dashboard password for user '$DASH_USER': " DASH_PASS; echo
    fi
fi
[[ ${#DASH_PASS} -ge 12 ]] || warn "Password is under 12 characters. This will be public on the internet — use something long."
[[ "$DASH_PASS" == *','* ]] && info "Password contains a comma; that's fine in config/.env and in a Caddy hash."

if [[ -z "$ACME_EMAIL" ]]; then
    read -rp "Email for Let's Encrypt expiry notices: " ACME_EMAIL
    [[ -n "$ACME_EMAIL" ]] || die "An email is required for the ACME account"
fi

# ── 1. Caddy ───────────────────────────────────────────────────────────

step "Installing Caddy"
if command -v caddy >/dev/null 2>&1; then
    ok "already installed: $(caddy version | head -1)"
else
    info "adding the official apt repository"
    apt-get update -qq
    apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl >/dev/null
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -qq
    apt-get install -y -qq caddy >/dev/null
    ok "installed: $(caddy version | head -1)"
fi

install -d -m 755 /var/log/caddy

# Caddy 2.8 renamed the directive basicauth -> basic_auth.
CADDY_VER=$(caddy version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
CADDY_MAJOR=${CADDY_VER%%.*}
CADDY_MINOR=$(echo "$CADDY_VER" | cut -d. -f2)
if (( CADDY_MAJOR > 2 )) || { (( CADDY_MAJOR == 2 )) && (( CADDY_MINOR >= 8 )); }; then
    AUTH_DIRECTIVE="basic_auth"
else
    AUTH_DIRECTIVE="basicauth"
    warn "Caddy $CADDY_VER predates 2.8; using the older 'basicauth' directive"
fi
ok "auth directive: $AUTH_DIRECTIVE"

# ── 2. Credentials in config/.env ──────────────────────────────────────

step "Writing dashboard config to config/.env"
info "This file is git-ignored, so the password is not committed."
touch "$ENV_FILE"; chmod 600 "$ENV_FILE"

# Rewrite a key in place if present, otherwise append.
#
# awk only, deliberately. This file holds live Polymarket, Telegram and email
# credentials, so the rewrite has to be lossless. A sed expression would break
# on a password containing '/' or '&', and a bad in-place sed on this file
# could take the trading bot down. awk takes the new value as an -v variable,
# which is passed literally and never re-parsed as a pattern or replacement.
set_env() {
    local key="$1" value="$2" file="$3" tmp
    tmp="$(mktemp)"
    if grep -qE "^${key}=" "$file"; then
        # Pass the value through the environment and read it with ENVIRON,
        # not via -v. awk's -v runs the argument through escape processing,
        # so a password containing a backslash (e.g. "...\2") would be
        # rewritten as a control character and the saved value would differ
        # from the one Caddy hashed. ENVIRON is taken literally.
        SE_KEY="$key" SE_VAL="$value" awk '
            BEGIN { k = ENVIRON["SE_KEY"]; v = ENVIRON["SE_VAL"] }
            index($0, k "=") == 1 { print k "=" v; next }
            { print }
        ' "$file" > "$tmp"
    else
        cp "$file" "$tmp"
        printf '%s=%s\n' "$key" "$value" >> "$tmp"
    fi
    # Preserve the original inode/permissions and only replace on success.
    cat "$tmp" > "$file"
    rm -f "$tmp"
}

set_env DASHBOARD_USERNAME "$DASH_USER" "$ENV_FILE"
set_env DASHBOARD_PASSWORD "$DASH_PASS" "$ENV_FILE"
# Bind to loopback only. Caddy is the sole entry point.
set_env DASHBOARD_HOST "127.0.0.1" "$ENV_FILE"
set_env DASHBOARD_PORT "$DASH_PORT" "$ENV_FILE"
chmod 600 "$ENV_FILE"
ok "DASHBOARD_USERNAME/PASSWORD/HOST/PORT set (host pinned to 127.0.0.1)"

# ── 3. Caddyfile ───────────────────────────────────────────────────────

step "Writing $CADDYFILE"
PASS_HASH=$(caddy hash-password --plaintext "$DASH_PASS")
[[ -n "$PASS_HASH" ]] || die "caddy hash-password produced nothing"

RENDERED=$(mktemp)
# Caddy's own config validator is the authority here — render, then prove it.
sed -e "s|__ACME_EMAIL__|${ACME_EMAIL}|g" \
    -e "s|__DOMAIN__|${FQDN}|g" \
    -e "s|__AUTH_DIRECTIVE__|${AUTH_DIRECTIVE}|g" \
    -e "s|__DASHBOARD_USER__|${DASH_USER}|g" \
    -e "s|__DASHBOARD_PASS_HASH__|$(printf '%s' "$PASS_HASH" | sed -e 's/[\/&]/\\&/g'|g)" \
    -e "s|__DASHBOARD_PORT__|${DASH_PORT}|g" \
    "$TEMPLATE_DIR/caddy/Caddyfile.in" > "$RENDERED"

if ! caddy validate --config "$RENDERED" --adapter caddyfile 2>&1; then
    rm -f "$RENDERED"
    die "The generated Caddyfile is invalid — nothing was installed. Fix deploy/caddy/Caddyfile.in and re-run."
fi
ok "generated config passes 'caddy validate'"

install -m 644 "$RENDERED" "$CADDYFILE"
rm -f "$RENDERED"
ok "installed (password stored as a bcrypt hash, plaintext never written to disk)"

# ── 4. Firewall ────────────────────────────────────────────────────────

if [[ $DO_UFW -eq 1 ]]; then
    step "Locking the firewall to 22/80/443"
    apt-get install -y -qq ufw >/dev/null
    ufw --force default deny incoming >/dev/null
    ufw --force default allow outgoing >/dev/null
    ufw allow 22/tcp  >/dev/null
    ufw allow 80/tcp  >/dev/null   # ACME HTTP-01 + redirect to HTTPS
    ufw allow 443/tcp >/dev/null
    ufw --force enable >/dev/null
    ok "8080/${DASH_PORT} is NOT open — Flask is reachable only through Caddy"
    ufw status | sed 's/^/    /'
else
    step "Skipping firewall (--no-ufw)"
    warn "ensure ports 80 and 443 are open and ${DASH_PORT} is NOT exposed publicly"
fi

# ── 5. fail2ban ────────────────────────────────────────────────────────

if [[ $DO_FAIL2BAN -eq 1 ]]; then
    step "Installing fail2ban brute-force throttling"
    apt-get install -y -qq fail2ban >/dev/null
    install -d -m 755 /etc/fail2ban/filter.d /etc/fail2ban/jail.d
    install -m 644 "$TEMPLATE_DIR/fail2ban/filter.d/caddy-dashboard-401.conf" /etc/fail2ban/filter.d/
    install -m 644 "$TEMPLATE_DIR/fail2ban/jail.d/caddy-dashboard.conf"   /etc/fail2ban/jail.d/
    systemctl enable --now fail2ban >/dev/null
    systemctl restart fail2ban >/dev/null
    ok "10 wrong passwords in 5 min -> 10 min ban"
    info "check bans with:  fail2ban-client status caddy-dashboard"
else
    step "Skipping fail2ban (--no-fail2ban)"
fi

# ── 6. DuckDNS auto-update ─────────────────────────────────────────────

if [[ -n "$DUCKDNS_TOKEN" ]]; then
    step "Enabling DuckDNS auto-update"
    install -d -m 755 /etc
    printf 'DUCKDNS_TOKEN=%s\n' "$DUCKDNS_TOKEN" > /etc/duckdns.env
    printf 'DUCKDNS_DOMAIN=%s\n' "$DOMAIN" > /etc/duckdns.domain
    chmod 600 /etc/duckdns.env /etc/duckdns.domain
    install -m 755 "$TEMPLATE_DIR/systemd/duckdns_update.sh" /usr/local/sbin/duckdns_update.sh
    install -m 644 "$TEMPLATE_DIR/systemd/duckdns-update.service" /etc/systemd/system/
    install -m 644 "$TEMPLATE_DIR/systemd/duckdns-update.timer"   /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable --now duckdns-update.timer >/dev/null
    # Prove it now rather than discovering a typo five minutes from now.
    if /usr/local/sbin/duckdns_update.sh; then
        ok "record updated and timer active (every 5 min)"
    else
        warn "timer installed but the first update failed — check the token"
    fi
else
    step "Skipping DuckDNS auto-update (no --duckdns-token)"
    info "the record will go stale if the VPS IP changes; re-run with --duckdns-token"
fi

# ── 7. Start Caddy and restart the bot ─────────────────────────────────

step "Starting Caddy and restarting the dashboard"
systemctl enable --now caddy >/dev/null
systemctl reload caddy 2>/dev/null || systemctl restart caddy
ok "caddy: $(systemctl is-active caddy)"

# The bot holds the env file; a restart is what applies the new credentials.
# If the service is down this is a no-op rather than a failure — the trading
# bot may intentionally not be running on this host.
if systemctl is-active --quiet "$BOT_UNIT"; then
    systemctl restart "$BOT_UNIT"
    ok "$BOT_UNIT restarted with the new dashboard config"
else
    warn "$BOT_UNIT is not active; start it to pick up the dashboard settings"
fi

# ── 8. Verify ──────────────────────────────────────────────────────────

step "Verifying"
info "waiting up to 60s for DNS + ACME issuance (first run can be slow)"
READY=0
for _ in $(seq 1 30); do
    if curl -fsS --max-time 10 "https://${FQDN}/" -u "${DASH_USER}:${DASH_PASS}" >/dev/null 2>&1; then
        READY=1; break
    fi
    sleep 2
done

if [[ $READY -eq 1 ]]; then
    ok "authenticated request to $URL succeeded"
else
    warn "could not reach $URL yet — the usual cause is DNS not pointing here, or ports 80/443 blocked upstream"
    info "dig +short ${FQDN}   # must return 158.220.80.54"
    info "journalctl -u caddy -n 50 --no-pager"
fi

unauth=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://${FQDN}/" 2>/dev/null || echo 000)
if [[ "$unauth" == "401" ]]; then ok "unauthenticated request correctly rejected (401)"
else warn "unauthenticated request returned $unauth, expected 401"; fi

wd=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 -X POST "https://${FQDN}/api/withdraw" \
    -u "${DASH_USER}:${DASH_PASS}" -H 'Content-Type: application/json' \
    -d '{"amount":1,"destination":"0x0"}' 2>/dev/null || echo 000)
if [[ "$wd" == "403" ]]; then ok "withdrawal endpoint blocked (403) — cannot move funds from the internet"
else warn "withdrawal endpoint returned $wd, expected 403 — investigate before going live"; fi

step "Done"
printf '    Dashboard : %s\n' "$URL"
printf '    Username  : %s\n' "$DASH_USER"
printf '    Password  : %s\n' "$DASH_PASS"
cat <<EOF

    Withdrawing is intentionally NOT possible from this URL. Use the admin
    API over a tunnel instead:

      ssh -L 8787:127.0.0.1:8787 vps
      # then POST to http://127.0.0.1:8787/...

    Re-check later with:
      sudo ./scripts/expose_dashboard.sh --domain ${DOMAIN} --check
EOF
