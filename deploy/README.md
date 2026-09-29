# Publishing the dashboard

Makes the real-time dashboard reachable at `https://<subdomain>.duckdns.org/`
over HTTPS, behind a password, without exposing the Flask port to the internet.

## TL;DR

On the VPS, from a full checkout:

```bash
cd /opt/wamucheha-polymarket
sudo ./scripts/expose_dashboard.sh --domain wamucheha-poly
```

It will prompt for the password and a Let's Encrypt contact email. Add
`--duckdns-token <token>` to also create/refresh the DNS record automatically.

Then open `https://wamucheha-poly.duckdns.org/` and sign in.

## What it actually sets up

```
internet ──443──> Caddy ──127.0.0.1:8080──> Flask (waitress)
           401 + bcrypt
           403 on /api/withdraw
```

| Concern | How it is handled |
|---|---|
| TLS | Caddy obtains and renews a Let's Encrypt cert automatically (HTTP-01 works with DuckDNS) |
| Password | bcrypt hash in `/etc/caddy/Caddyfile`; plaintext only in git-ignored `config/.env` |
| Raw app port | `DASHBOARD_HOST=127.0.0.1`, and UFW does not open `8080` |
| Brute force | fail2ban bans an IP for 10 min after 10 failed passwords in 5 min |
| The withdrawal endpoint | Hard 403 at the proxy — unreachable from the internet |
| IP changes | Optional DuckDNS timer refreshes the record every 5 min |

## Why the withdrawal endpoint is blocked

`POST /api/withdraw` transfers real USDC out of the trading wallet. Sharing one
password between read-only monitoring and a money-moving endpoint means a single
leaked or guessed password is enough to drain the account. The proxy refuses
that path outright, so a compromised dashboard password costs you visibility
rather than funds.

To withdraw, reach the admin API over an SSH tunnel instead — it never leaves
the loopback interface:

```bash
ssh -L 8787:127.0.0.1:8787 vps
curl -H "X-Admin-Token: $ADMIN_API_TOKEN" http://127.0.0.1:8787/...
```

## Options

```
--domain NAME      DuckDNS subdomain, e.g. wamucheha-poly   (required)
--user NAME        dashboard username          (default: wamucheha)
--password PASS    omit to be prompted invisibly, or reuse the existing one
--email EMAIL      ACME/Let's Encrypt contact address
--port PORT        Flask dashboard port         (default: 8080)
--duckdns-token T  DuckDNS account token; enables the auto-update timer
--no-ufw           skip firewall changes if you manage them elsewhere
--no-fail2ban      skip brute-force throttling
--check            verify the current setup, change nothing
```

The script is idempotent. Re-run it after a redeploy or to rotate the password
— it rewrites the credentials, re-renders the Caddyfile, validates it with
`caddy validate`, and restarts the services.

To rotate the password without putting it in shell history:

```bash
sudo ./scripts/expose_dashboard.sh --domain wamucheha-poly
# prompted for the password; input is hidden
```

## DuckDNS setup

1. Log in at <https://www.duckdns.org/> and create the subdomain
   (e.g. `wamucheha-poly`).
2. Copy the token from the site.
3. Point the record at this VPS and run the script with `--duckdns-token`:

```bash
sudo ./scripts/expose_dashboard.sh --domain wamucheha-poly --duckdns-token <token>
```

The token is stored in `/etc/duckdns.env` (mode 600) and read by
`duckdns-update.service`, which runs every 5 minutes. It never touches the
repo or `config/.env`.

Without the token the record still works, but goes stale if the VPS address
changes and the dashboard silently disappears.

## Verifying

```bash
sudo ./scripts/expose_dashboard.sh --domain wamucheha-poly --check
```

Checks that 80/443/8080 are listening as expected, that an authenticated
request succeeds, that an unauthenticated one gets 401, and that
`/api/withdraw` is refused with 403.

Useful while debugging:

```bash
dig +short wamucheha-poly.duckdns.org   # must return the VPS public IP
journalctl -u caddy -n 50 --no-pager    # ACME issuance, TLS errors
tail -f /var/log/caddy/wamucheha-access.log
fail2ban-client status caddy-dashboard
```

## First run takes a minute

Caddy has to complete the ACME challenge before it will answer on 443, and
Let's Encrypt rate-limits by domain. If issuance fails, the usual causes are:

- `wamucheha-poly.duckdns.org` does not resolve to this VPS yet
- ports 80/443 are blocked upstream of the VPS
- too many certificates were already issued for this domain

## Hardening beyond this

The dashboard is intentionally public, so anyone can hit it. Two upgrades if
you want less exposure:

- **Restrict by IP** — add a `remote_ip` matcher in `caddy/Caddyfile.in` to
  allow only your home IP. Strongest option, but you need a static IP.
- **Tailscale instead of public DNS** — no open ports, no ACME, no password.
  Requires the Tailscale client on every device you check from.

Edit `deploy/caddy/Caddyfile.in` (not `/etc/caddy/Caddyfile`, which is
overwritten on each run) and re-run the script.
