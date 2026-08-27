#!/usr/bin/env python3
"""Promote a user to admin (there is deliberately no self-service route for it).

    make admin EMAIL=you@company.com
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.accounts import Accounts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--revoke", action="store_true")
    args = ap.parse_args()

    accounts = Accounts()
    accounts.init_schema()
    with accounts.conn() as c, c.cursor() as cur:
        cur.execute("UPDATE users SET is_admin = %s WHERE email = %s RETURNING email, is_admin",
                    (not args.revoke, args.email.strip().lower()))
        row = cur.fetchone()
        c.commit()
    if not row:
        print(f"no user with email {args.email!r} -- sign up in the app first")
        return 1
    print(f"{row['email']}: is_admin = {row['is_admin']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
