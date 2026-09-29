#!/usr/bin/env bash
#
# Regression test for the set_env() helper in scripts/expose_dashboard.sh.
#
# That function rewrites keys in the VPS config/.env, which holds live
# Polymarket, Telegram and email credentials. A subtle escaping bug there would
# silently corrupt a real credential and take the trading bot down, so the
# behaviour is pinned here. Two bugs this has already caught:
#
#   1. A sed-based rewrite mis-parsed values containing '/' or '&'.
#   2. Passing the value via `awk -v` applied escape processing, so a password
#      containing a backslash ("...\2") was written back as a control char and
#      no longer matched the hash Caddy was given.
#
# Run:  bash tests/test_expose_dashboard_env.sh
set -u

# Keep this copy identical to the one in the script; scripts/expose_dashboard.sh
# is not importable, so the function is duplicated deliberately.
set_env() {
    local key="$1" value="$2" file="$3" tmp
    tmp="$(mktemp)"
    if grep -qE "^${key}=" "$file"; then
        SE_KEY="$key" SE_VAL="$value" awk '
            BEGIN { k = ENVIRON["SE_KEY"]; v = ENVIRON["SE_VAL"] }
            index($0, k "=") == 1 { print k "=" v; next }
            { print }
        ' "$file" > "$tmp"
    else
        cp "$file" "$tmp"
        printf '%s=%s\n' "$key" "$value" >> "$tmp"
    fi
    cat "$tmp" > "$file"
    rm -f "$tmp"
}

F=$(mktemp)
trap 'rm -f "$F"' EXIT
cat > "$F" <<'ORIGINAL'
# comment line
POLYMARKET_API_KEY=0xdeadbeef&live=true
TELEGRAM_BOT_TOKEN=123:ABC-DEF_slash/here
DASHBOARD_USERNAME=wamucheha
DASHBOARD_PASSWORD=oldpass
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
ORIGINAL

# comma, slash, ampersand, dollar, backslash, single quote
PW="Wam,uc/ch&a\$ha\\2'5"

set_env DASHBOARD_PASSWORD "$PW"      "$F"
set_env DASHBOARD_USERNAME wamucheha  "$F"
set_env DASHBOARD_HOST     127.0.0.1  "$F"
set_env DASHBOARD_PORT     8080       "$F"
set_env BRAND_NEW          "added=with=equals" "$F"

fail=0
check() {
    if [ "$2" = "ok" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s\n' "$1"; fail=1; fi
}

grep -qxF 'POLYMARKET_API_KEY=0xdeadbeef&live=true' "$F" \
    && check "live API key untouched" ok || check "live API key untouched" bad
grep -qxF 'TELEGRAM_BOT_TOKEN=123:ABC-DEF_slash/here' "$F" \
    && check "telegram token untouched" ok || check "telegram token untouched" bad
grep -qxF 'DASHBOARD_HOST=127.0.0.1' "$F" \
    && check "host rewritten to loopback" ok || check "host rewritten to loopback" bad
grep -qxF 'DASHBOARD_PORT=8080' "$F" \
    && check "port preserved" ok || check "port preserved" bad
grep -qxF 'DASHBOARD_USERNAME=wamucheha' "$F" \
    && check "username stable" ok || check "username stable" bad
[ "$(grep -c '^DASHBOARD_PASSWORD=' "$F")" -eq 1 ] \
    && check "no duplicate password key" ok || check "no duplicate password key" bad
[ "$(grep -m1 '^DASHBOARD_PASSWORD=' "$F")" = "DASHBOARD_PASSWORD=$PW" ] \
    && check "password roundtrip exact" ok || check "password roundtrip exact" bad
[ "$(grep -c '^' "$F")" -eq 8 ] \
    && check "line count correct (7 original + 1 appended)" ok || check "line count correct" bad

if [ $fail -eq 0 ]; then echo "  all set_env checks passed"; fi
exit $fail
