"""Frozen continuation constants and conservative, crash-safe budget accounting."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path

ROOT = Path('/home/hhai/pretrain')
RUN = ROOT / 'continuation/20260911-v1'
PARENT = ROOT / 'checkpoints/full/final/lit_model.pth'
PARENT_TOKENS = 19_999_703_040
SEQ = 2048
BLOCK = 2049
MICRO = 6
ACCUM = 85
GLOBAL = MICRO * ACCUM
STEP_TOKENS = GLOBAL * SEQ
CAP = 2_000_000_000
PILOT_STEPS = 190
BRANCH_STEPS = 1724
OLD_WEIGHTS = {'web': .425, 'dclm': .425, 'math': .03, 'code': .12}
MIXES = {
    'A': {**{f'replay_{s}': .3 * w for s, w in OLD_WEIGHTS.items()},
          **{f'fresh_{s}': .7 * w for s, w in OLD_WEIGHTS.items()}},
    'B': {**{f'replay_{s}': .3 * w for s, w in OLD_WEIGHTS.items()},
          'fresh_dclm': .35, 'fresh_web': .20, 'fresh_narrative': .05,
          'fresh_math': .05, 'fresh_code': .05},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def owned(path: Path, root: Path = RUN) -> Path:
    resolved = path.resolve()
    if resolved == root.resolve() or root.resolve() not in resolved.parents:
        raise ValueError(f'output is outside the dedicated run: {path}')
    return resolved


def quotas(weights: dict[str, float], n: int) -> dict[str, int]:
    if n < 0 or not math.isclose(sum(weights.values()), 1, abs_tol=1e-12):
        raise ValueError('invalid mixture')
    if any(w <= 0 for w in weights.values()):
        raise ValueError('non-positive weight')
    result = {k: math.floor(w * n) for k, w in weights.items()}
    order = sorted(weights, key=lambda k: (-(weights[k] * n - result[k]), k))
    for k in order[:n - sum(result.values())]:
        result[k] += 1
    assert sum(result.values()) == n
    return result


def learning_rate(step: int) -> float:
    """Branch-local full-horizon schedule; pilot stop does not shorten cooldown."""
    if not 0 <= step < BRANCH_STEPS:
        raise ValueError('step outside branch schedule')
    hold = 172
    if step < hold:
        return 4e-5
    progress = (step - hold) / (BRANCH_STEPS - 1 - hold)
    return 4e-6 + .5 * (4e-5 - 4e-6) * (1 + math.cos(math.pi * progress))


class Budget:
    """Reserve before any backward; incomplete/repeated steps are never refunded.

    A power loss may overcount at most one full optimizer step, never undercount
    it. Committed optimizer progress is stored independently in checkpoints.
    """
    def __init__(self, path: Path, cap: int = CAP):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY, cap INTEGER)')
        self.db.execute('INSERT OR IGNORE INTO policy VALUES (1, ?)', (cap,))
        if self.db.execute('SELECT cap FROM policy WHERE id=1').fetchone()[0] != cap:
            raise ValueError('budget cap changed')
        self.db.execute('CREATE TABLE IF NOT EXISTS charges (id INTEGER PRIMARY KEY, branch TEXT, '
                        'step INTEGER, tokens INTEGER, created REAL, completed INTEGER DEFAULT 0)')
        self.db.commit()
        self.cap = cap

    @property
    def spent(self) -> int:
        return self.db.execute('SELECT COALESCE(SUM(tokens),0) FROM charges').fetchone()[0]

    def reserve(self, branch: str, step: int, tokens: int = STEP_TOKENS) -> int:
        if branch not in MIXES or step < 0 or tokens <= 0:
            raise ValueError('invalid charge')
        try:
            self.db.execute('BEGIN IMMEDIATE')
            if self.spent + tokens > self.cap:
                raise RuntimeError('strict total token budget exhausted')
            cursor = self.db.execute('INSERT INTO charges(branch,step,tokens,created) VALUES (?,?,?,?)',
                                     (branch, step, tokens, time.time()))
            self.db.commit()
            return cursor.lastrowid
        except BaseException:
            self.db.rollback()
            raise

    def complete(self, charge: int) -> None:
        with self.db:
            self.db.execute('UPDATE charges SET completed=1 WHERE id=?', (charge,))

    def close(self) -> None:
        self.db.close()
