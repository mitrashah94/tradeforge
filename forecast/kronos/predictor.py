"""forecast/kronos/predictor.py — thin Kronos wrapper + the distribution stats.

Wraps ``KronosTokenizer`` / ``Kronos`` / ``KronosPredictor`` (default the
4.1M-param **Kronos-mini**, CPU/Apple-MPS-feasible for a premarket batch of ~40
symbols). ``forecast_distribution`` draws ``n_paths`` future trajectories
(``sample_count`` multi-path sampling) and reduces them to
``{exp_return, vol, downside_cvar, prob_up}``.

Importable WITHOUT torch: the heavy imports are lazy (only ``_load_model`` touches
torch/huggingface), and the path-sampling step is injectable (a ``sampler``
callable), so CI mocks the model and tests run with no weights download. The pure
:func:`forecast_stats` reduction is unit-tested directly. The look-ahead guard
(:func:`forecast.kronos.leakage.assert_post_cutoff`) fires FIRST on every call.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Optional, Sequence

import numpy as np

from forecast.kronos.leakage import KRONOS_TRAINING_CUTOFF, assert_post_cutoff

DEFAULT_MODEL_ID = "kronos-mini"
_DOWNSIDE_ALPHA = 0.05   # CVaR tail fraction

# HuggingFace repo + paired tokenizer per model id. Kronos-mini pairs with the 2k
# tokenizer; small/base pair with the base tokenizer (per the Kronos README).
_MODEL_HF: dict = {
    "kronos-mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k"),
    "kronos-small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"),
    "kronos-base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base"),
}


# --------------------------------------------------------------------------- #
# Pure reduction: terminal H-step returns -> the forecast distribution
# --------------------------------------------------------------------------- #
def forecast_stats(terminal_returns: Sequence[float], alpha: float = _DOWNSIDE_ALPHA) -> dict:
    """Reduce sampled terminal returns to ``{exp_return, vol, downside_cvar, prob_up}``.

    * ``exp_return``   — mean horizon return across paths;
    * ``vol``          — sample std of the horizon returns (path dispersion);
    * ``downside_cvar``— POSITIVE expected-shortfall magnitude: ``-mean`` of the
      worst ``alpha`` fraction of returns (0 when that tail is non-negative). A
      larger value is a worse downside (the budgeter vetoes beyond a threshold);
    * ``prob_up``      — fraction of paths with a positive horizon return.

    Deterministic and torch-free; given the same sampled paths it always returns
    the same numbers.
    """
    arr = np.asarray(list(terminal_returns), dtype="float64")
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"exp_return": float("nan"), "vol": float("nan"),
                "downside_cvar": float("nan"), "prob_up": float("nan")}
    exp_return = float(arr.mean())
    vol = float(arr.std(ddof=1)) if n > 1 else 0.0
    k = max(1, int(np.ceil(alpha * n)))
    worst = np.sort(arr)[:k]
    cvar = float(-worst.mean())
    downside_cvar = cvar if cvar > 0 else 0.0
    prob_up = float((arr > 0).mean())
    return {"exp_return": exp_return, "vol": vol,
            "downside_cvar": downside_cvar, "prob_up": prob_up}


# --------------------------------------------------------------------------- #
# The predictor
# --------------------------------------------------------------------------- #
class KronosForecaster:
    """Kronos OHLCV forecaster: multi-path sampling -> a return/vol/downside frame.

    Parameters
    ----------
    model_id
        HuggingFace model tag (default ``kronos-mini``).
    device
        ``"cpu"`` / ``"mps"`` / ``"cuda"`` (the premarket batch is CPU/MPS-feasible
        on the mini model).
    sampler
        OPTIONAL injected ``sampler(ohlc_history, horizon, n_paths, seed) ->
        np.ndarray`` of terminal H-step returns. When provided, the real model is
        NEVER loaded — this is the test/mocked path AND a deterministic-seed escape
        hatch. When ``None``, :meth:`forecast_distribution` lazily loads the torch
        model on first use.
    cutoff
        The leakage cutoff enforced on every call.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        *,
        sampler: Optional[Callable] = None,
        cutoff: date = KRONOS_TRAINING_CUTOFF,
        context: int = 128,
    ):
        self.model_id = model_id
        self.device = device
        self._sampler = sampler
        self.cutoff = cutoff
        self.context = int(context)
        self._model = None  # lazily loaded torch model (None until first real use)

    # -- model loading (lazy; torch only imported here) -------------------- #
    def _load_model(self):
        """Lazily construct the torch Kronos model. Raises if torch is absent.

        The Kronos model classes live in the upstream repo's ``model`` package
        (github.com/shiyu-coder/Kronos — not on PyPI), so the repo checkout path
        is supplied via the ``KRONOS_REPO_PATH`` env var and inserted on
        ``sys.path``. Weights resolve from HuggingFace via :data:`_MODEL_HF`.
        """
        if self._model is not None:
            return self._model
        import os
        import sys
        repo = os.environ.get("KRONOS_REPO_PATH")
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            import torch  # noqa: F401
            from model import Kronos, KronosPredictor, KronosTokenizer
        except ImportError as e:  # pragma: no cover - exercised only with torch absent
            raise ImportError(
                "Kronos requires the optional ML extras (pip3 install "
                "'tradeforge[ml]') plus the Kronos repo on KRONOS_REPO_PATH "
                "(git clone https://github.com/shiyu-coder/Kronos). Install them, "
                "or pass a `sampler=` to run without the model."
            ) from e
        hf_model, hf_tok = _MODEL_HF.get(
            self.model_id, (f"NeoQuasar/{self.model_id}", "NeoQuasar/Kronos-Tokenizer-base")
        )
        tok = KronosTokenizer.from_pretrained(hf_tok)
        net = Kronos.from_pretrained(hf_model)
        self._model = KronosPredictor(net, tok, device=self.device, max_context=512)
        return self._model

    # -- the one public call ---------------------------------------------- #
    def forecast_distribution(
        self,
        symbol: str,
        asof,
        ohlc_history,
        *,
        horizon: int = 5,
        n_paths: int = 32,
        seed: int = 0,
    ) -> dict:
        """Forecast the H-step return distribution for ``symbol`` as of ``asof``.

        Enforces the look-ahead guard FIRST (every input bar must be post-cutoff),
        then samples ``n_paths`` horizon trajectories (via the injected sampler or
        the lazily-loaded model) and reduces them with :func:`forecast_stats`.
        Returns ``{exp_return, vol, downside_cvar, prob_up}`` plus ``symbol`` /
        ``horizon`` / ``n_paths`` / ``model_id`` for the store row.
        """
        assert_post_cutoff(asof, ohlc_history, self.cutoff)
        terminal_returns = self._sample(ohlc_history, horizon, n_paths, seed)
        stats = forecast_stats(terminal_returns)
        stats.update({
            "symbol": symbol, "horizon": int(horizon),
            "n_paths": int(n_paths), "model_id": self.model_id,
        })
        return stats

    # -- sampling (injected or real model) -------------------------------- #
    def _sample(self, ohlc_history, horizon: int, n_paths: int, seed: int) -> np.ndarray:
        if self._sampler is not None:
            return np.asarray(self._sampler(ohlc_history, horizon, n_paths, seed),
                              dtype="float64")
        return self._sample_with_model(ohlc_history, horizon, n_paths, seed)

    def _sample_with_model(self, ohlc_history, horizon, n_paths, seed) -> np.ndarray:  # pragma: no cover - needs weights
        """Draw terminal H-step returns from the real model (requires torch+weights).

        ``KronosPredictor.predict``'s ``sample_count`` AVERAGES its samples
        internally (one mean path — useless for a distribution), so multi-path
        sampling is done via ``predict_batch`` with ``n_paths`` COPIES of the same
        series at ``sample_count=1``: one forward pass, ``n_paths`` independent
        sampled trajectories. Each path's terminal close becomes a cumulative
        H-step return vs the last known close. Seeded for determinism.
        """
        import pandas as pd
        import torch

        model = self._load_model()
        hist = ohlc_history.tail(self.context) if hasattr(ohlc_history, "tail") else ohlc_history
        cols = ["open", "high", "low", "close"] + (["volume"] if "volume" in hist.columns else [])
        frame = hist[cols].astype("float64").reset_index(drop=True)
        if "volume" in frame.columns:
            frame["volume"] = frame["volume"].fillna(0.0)
        # timestamps: the ts_utc column when present, else the (datetime) index.
        if "ts_utc" in hist.columns:
            x_ts = pd.Series(pd.to_datetime(hist["ts_utc"]).to_numpy())
        else:
            x_ts = pd.Series(pd.to_datetime(list(hist.index)))
        y_ts = pd.Series(
            pd.bdate_range(start=x_ts.iloc[-1] + pd.Timedelta(days=1), periods=int(horizon))
        )
        last_close = float(frame["close"].iloc[-1])

        torch.manual_seed(int(seed))
        preds = model.predict_batch(
            [frame] * int(n_paths),
            [x_ts] * int(n_paths),
            [y_ts] * int(n_paths),
            pred_len=int(horizon),
            T=1.0, top_p=0.9, sample_count=1, verbose=False,
        )
        return np.asarray(
            [float(p["close"].iloc[-1]) / last_close - 1.0 for p in preds],
            dtype="float64",
        )
