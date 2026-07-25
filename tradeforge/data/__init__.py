"""TradeForge core data + levels library.

Data layer only — no trading/strategy logic. Deterministic, point-in-time
correct. Bars are stored tz-naive in UTC; equity sessions are evaluated in
America/New_York (DST-correct), crypto on the UTC calendar.

This package's ``__init__`` is intentionally import-light: importing
``data.levels`` or ``data.sessions`` must never pull in ingest/broker code.
"""
