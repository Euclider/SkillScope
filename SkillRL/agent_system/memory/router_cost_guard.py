"""Optional cohort-wide, pre-request money reservations (including in-flight calls).

The accounting exchange coefficient is a frozen conservative budget conversion,
not a live currency quote. Missing provider usage retains the entire reservation.
No existing router protocol/cache identity or historical default is changed.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

from .router_cache import RouterBudgetExceeded, RouterCacheError, canonical_json


class CostGuard:
    def __init__(self, profile_path):
        self.profile_path = Path(profile_path).resolve()
        self.profile = json.loads(self.profile_path.read_text())
        expected = {'schema_version', 'cap_rmb', 'rmb_per_usd_accounting', 'input_usd_per_million', 'output_usd_per_million'}
        if set(self.profile) != expected or self.profile['schema_version'] != 'skillrl.router.cost_cap.v1':
            raise RouterCacheError('Invalid money-cap profile')
        for key in expected - {'schema_version'}:
            value = self.profile[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise RouterCacheError('Money limits and accounting rates must be positive finite numbers')
        self.path = self.profile_path.with_suffix('.sqlite3')
        if self.path.is_symlink():
            raise RouterCacheError('Symlinked money ledger')
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS profile (id INTEGER PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS charges (id INTEGER PRIMARY KEY, key TEXT UNIQUE, reserved INTEGER, charged INTEGER, settled INTEGER)')
            value = canonical_json(self.profile)
            db.execute('INSERT OR IGNORE INTO profile VALUES (1, ?)', (value,))
            if db.execute('SELECT value FROM profile WHERE id=1').fetchone() != (value,):
                raise RouterCacheError('Changed monetary budget; no implicit extension')

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @property
    def cap_nano_usd(self):
        return int(Decimal(str(self.profile['cap_rmb'])) / Decimal(str(self.profile['rmb_per_usd_accounting'])) * 10**9)

    def charge(self, prompt, completion):
        amount = (Decimal(prompt) * Decimal(str(self.profile['input_usd_per_million']))
                  + Decimal(completion) * Decimal(str(self.profile['output_usd_per_million']))) * 1000
        return int(amount.to_integral_value(rounding=ROUND_CEILING))

    def reserve(self, key, request, max_completion_tokens):
        # UTF-8 bytes upper-bound ordinary byte-BPE tokens; the explicit extra
        # 8192 covers chat/schema framing. No cache-discount assumption is used.
        upper_prompt = len(canonical_json(request).encode('utf-8')) + 8192
        amount = self.charge(upper_prompt, max_completion_tokens)
        with self.transaction() as db:
            spent = db.execute('SELECT COALESCE(SUM(charged), 0) FROM charges').fetchone()[0]
            if spent + amount > self.cap_nano_usd:
                raise RouterBudgetExceeded('Cohort monetary cap reached including next-call reservation; no request sent')
            db.execute('INSERT INTO charges (key, reserved, charged, settled) VALUES (?, ?, ?, 0)', (key, amount, amount))

    def settle(self, key, usage):
        values = [usage.get(name) for name in ('prompt_tokens', 'completion_tokens')]
        if any(type(value) is not int or value < 0 for value in values):
            return  # Timeout/unknown usage remains conservatively fully charged.
        actual = self.charge(*values)
        with self.transaction() as db:
            row = db.execute('SELECT reserved, settled FROM charges WHERE key=?', (key,)).fetchone()
            if row is None or row[1]:
                raise RouterCacheError('Missing/already-settled money reservation')
            db.execute('UPDATE charges SET charged=?, settled=1 WHERE key=?', (actual, key))
        if actual > row[0]:
            raise RouterBudgetExceeded('Provider usage exceeded the conservative reservation; halt for manual billing audit')

    def stats(self):
        with self.transaction() as db:
            count, nano, pending = db.execute('SELECT COUNT(*), COALESCE(SUM(charged), 0), COALESCE(SUM(1-settled), 0) FROM charges').fetchone()
        return {'attempts': count, 'unreconciled_calls': pending, 'usd_charged_or_reserved': nano / 1e9,
                'rmb_charged_or_reserved': nano / 1e9 * self.profile['rmb_per_usd_accounting'],
                'cap_rmb': self.profile['cap_rmb'], 'provider_invoice_verified': False}


def from_environment():
    path = os.environ.get('SKILLNET_ROUTER_COST_PROFILE')
    return CostGuard(path) if path else None
