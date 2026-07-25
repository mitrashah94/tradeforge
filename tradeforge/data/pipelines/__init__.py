"""TradeForge data pipelines (corporate actions, ingest helpers).

Import-light by design: importing a single pipeline module (e.g.
``data.pipelines.corporate_actions``) must not transitively import the
Alpaca ingest path or any broker SDK.
"""
