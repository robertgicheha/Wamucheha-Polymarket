#!/usr/bin/env bash
# End-to-end test of the proxy logic in deploy/caddy/Caddyfile.in.
#
# Renders the real template, rewrites only the deployment-specific bits
# (ACME -> local CA, public domain -> localhost, upstream port), then drives a
# live Caddy in front of a mock upstream and asserts the behaviour that
# matters: password required, correct password accepted, withdrawal refused.
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)

# Needs a real caddy binary: bcrypt hashing and the config adapter are not
# reimplementable, and a stub would not exercise the auth path.
CADDY=${CADDY_BIN:-$(command -v caddy || true)}
if [[ -z "$CADDY" ]]; then
    for candidate in /tmp/caddy "$REPO/.tools/caddy"; do
        [[ -x "$candidate" ]] && { CADDY="$candidate"; break; }
    done
fi
if [[ -z "$CADDY" ]]; then
    echo "  SKIP  caddy binary not found (set CADDY_BIN=/path/to/caddy)"
    exit 0
fi
echo "  using caddy: $CADDY"

WORK=$(mktemp -d)
DOMAIN=wamucheha-poly.duckdns.org
UPSTREAM_PORT=18080
LISTEN_PORT=18443
# Throwaway credentials. The real ones are written to the git-ignored VPS
# config/.env and hashed into /etc/caddy/Caddyfile at deploy time; they must
# never appear in a committed file.
USER_NAME=testuser
USER_PASS='correct-horse-battery-staple'

cleanup() {
    [[ -n "${CADDY_PID:-}" ]] && kill "$CADDY_PID" 2>/dev/null
    [[ -n "${UPSTREAM_PID:-}" ]] && kill "$UPSTREAM_PID" 2>/dev/null
    wait 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT

fail=0
check() {
    if [ "$2" = "ok" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s  (got: %s)\n' "$1" "${3:-}"; fail=1; fi
}

# 1. Real bcrypt hash from the real binary, so auth is genuinely exercised.
HASH=$("$CADDY" hash-password --plaintext "$USER_PASS")
[[ -n "$HASH" ]] || { echo "could not hash password"; exit 1; }

# 2. Render the actual template from the repo.
sed -e "s|__ACME_EMAIL__|test@example.com|g" \
    -e "s|__DOMAIN__|${DOMAIN}|g" \
    -e "s|__AUTH_DIRECTIVE__|basic_auth|g" \
    -e "s|__DASHBOARD_USER__|${USER_NAME}|g" \
    -e "s|__DASHBOARD_PASS_HASH__|$(printf '%s' "$HASH" | sed -e 's/[\/&]/\\&/g')|g" \
    -e "s|__DASHBOARD_PORT__|${UPSTREAM_PORT}|g" \
    "$REPO/deploy/caddy/Caddyfile.in" > "$WORK/rendered"

# 3. Swap only the environment-specific parts for a local test.
sed -e '/^[[:space:]]*email /d' \
    -e '/^[[:space:]]*acme_ca /d' \
    -e "s|^${DOMAIN} {|https://localhost:${LISTEN_PORT} {|" \
    -e "s|/var/log/caddy/wamucheha-access.log|${WORK}/access.log|" \
    "$WORK/rendered" > "$WORK/Caddyfile"

echo "  -- config under test --"
grep -E "basic_auth|wamucheha|respond @withdraw|reverse_proxy" "$WORK/Caddyfile" | sed 's/^/     /'

# 4. Mock upstream.
python3 "$REPO/tests/mock_dashboard_upstream.py" &
UPSTREAM_PID=$!
sleep 1

# 5. Caddy.
"$CADDY" run --config "$WORK/Caddyfile" --adapter caddyfile >"$WORK/caddy.log" 2>&1 &
CADDY_PID=$!
for _ in $(seq 1 25); do
    curl -sk "https://localhost:${LISTEN_PORT}/" -o /dev/null 2>/dev/null && break
    sleep 0.4
done

B="https://localhost:${LISTEN_PORT}"
echo "  -- behaviour --"

# No credentials -> refused.
code=$(curl -sk -o /dev/null -w '%{http_code}' "$B/")
[ "$code" = "401" ] && check "no credentials -> 401" ok || check "no credentials -> 401" bad "$code"

# Wrong credentials -> refused.
code=$(curl -sk -o /dev/null -w '%{http_code}' "$B/" -u "${USER_NAME}:wrongpassword")
[ "$code" = "401" ] && check "wrong password -> 401" ok || check "wrong password -> 401" bad "$code"

# Correct credentials -> reaches the app.
body=$(curl -sk "$B/" -u "${USER_NAME}:${USER_PASS}")
echo "$body" | grep -q '"path": "/"'
[ $? -eq 0 ] && check "correct password -> reaches app" ok || check "correct password -> reaches app" bad "$body"

# The Authorization header must survive the hop, otherwise the app's own
# Basic Auth layer would reject a request the proxy already approved.
echo "$body" | grep -q '"auth_header_present": true' \
    && check "auth header forwarded to app (one prompt only)" ok \
    || check "auth header forwarded to app (one prompt only)" bad "$body"

# Real client IP is passed through. Loopback may appear as 127.0.0.1 or ::1
# depending on how curl resolves localhost, so assert the property (a loopback
# address) rather than a literal.
ip=$(echo "$body" | sed -n 's/.*"x_real_ip": "\([^"]*\)".*/\1/p')
case "$ip" in
    127.0.0.1|::1) check "X-Real-IP reflects the real client" ok ;;
    *)            check "X-Real-IP reflects the real client" bad "$ip" ;;
esac

# Spoofed X-Forwarded-For must be replaced, not appended to. The security
# property is that the injected value does not survive.
spoof=$(curl -sk "$B/" -u "${USER_NAME}:${USER_PASS}" -H 'X-Forwarded-For: 6.6.6.6')
if echo "$spoof" | grep -q '6\.6\.6\.6'; then
    check "spoofed X-Forwarded-For is overwritten" bad "$(echo "$spoof" | sed -n 's/.*"x_forwarded_for": "\([^"]*\)".*/\1/p')"
else
    check "spoofed X-Forwarded-For is overwritten" ok
fi

# The money-moving endpoint, authenticated, must still be refused.
code=$(curl -sk -o "$WORK/wd" -w '%{http_code}' -X POST "$B/api/withdraw" \
    -u "${USER_NAME}:${USER_PASS}" -H 'Content-Type: application/json' \
    -d '{"amount":500,"destination":"0xattacker"}')
[ "$code" = "403" ] && check "authenticated POST /api/withdraw -> 403" ok \
    || check "authenticated POST /api/withdraw -> 403" bad "$code"
grep -q 'Withdrawal is disabled' "$WORK/wd" \
    && check "withdrawal refusal explains itself" ok \
    || check "withdrawal refusal explains itself" bad "$(cat "$WORK/wd")"

# The upstream must never have been asked to execute the transfer.
if grep -q 'WITHDRAWAL WOULD EXECUTE' "$WORK/wd"; then
    check "upstream never saw the transfer request" bad "request reached the app"
else
    check "upstream never saw the transfer request" ok
fi

# Trailing-slash variant of the path must be blocked too.
code=$(curl -sk -o /dev/null -w '%{http_code}' -X POST "$B/api/withdraw/" \
    -u "${USER_NAME}:${USER_PASS}" -H 'Content-Type: application/json' -d '{"amount":1}')
[ "$code" = "403" ] && check "POST /api/withdraw/ (trailing slash) -> 403" ok \
    || check "POST /api/withdraw/ (trailing slash) -> 403" bad "$code"

# Security headers present.
curl -skI "$B/" -u "${USER_NAME}:${USER_PASS}" > "$WORK/head"
grep -qi 'strict-transport-security' "$WORK/head" \
    && check "HSTS header present" ok || check "HSTS header present" bad
grep -qi 'x-frame-options: DENY' "$WORK/head" \
    && check "clickjacking protection present" ok || check "clickjacking protection present" bad

# A real read-only API route still works when authenticated. /api/state is an
# actual dashboard endpoint; using a route that does not exist would let the
# mock upstream's catch-all answer 200 and prove nothing about pass-through.
code=$(curl -sk -o /dev/null -w '%{http_code}' "$B/api/state" -u "${USER_NAME}:${USER_PASS}")
[ "$code" = "200" ] && check "authenticated API access works (/api/state)" ok \
    || check "authenticated API access works (/api/state)" bad "$code"

# And is refused without credentials.
code=$(curl -sk -o /dev/null -w '%{http_code}' "$B/api/state")
[ "$code" = "401" ] && check "API access without credentials -> 401" ok \
    || check "API access without credentials -> 401" bad "$code"

[ $fail -eq 0 ] && echo "  all proxy checks passed"
exit $fail
