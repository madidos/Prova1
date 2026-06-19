"""
etf_tracking.py — Analisi della qualita di replica di ETF vs indice di riferimento.

Riscrittura pulita e robusta del notebook "Analisi_TER".

Cosa fa, in breve
-----------------
Per ogni ETF, confronta la sua serie storica (prezzo total-return, da Yahoo
Finance) con la serie NET-return in EUR del suo indice MSCI di riferimento e
calcola metriche di replica oneste:

  - Tracking Difference (TD)  : CAGR(ETF) - CAGR(indice) sul periodo comune.
                                Atteso ~ -TER (al netto di tasse/securities lending).
  - Gap vs TER                : TD + TER. ~0 = efficiente; molto negativo = perde
                                piu della sua commissione (drag fiscale/replica scarsa).
  - Tracking Error (TE)       : dev.std. delle differenze di rendimento MENSILI
                                (non sovrapposte) annualizzata. Piu basso = replica
                                piu fedele. E' il vero indicatore di affidabilita.

Differenze chiave rispetto all'originale (perche e piu affidabile)
------------------------------------------------------------------
  * Niente Google Drive / Colab: i dati arrivano da file o URL espliciti.
  * Tracking Error calcolato su rendimenti mensili NON sovrapposti (l'originale
    usava differenze YoY a 12 mesi mensilmente -> fortemente autocorrelate, std
    sottostimata e poco significativa).
  * Tracking Difference come differenza di CAGR sul periodo comune, confrontata
    esplicitamente con il TER (l'originale non chiudeva il cerchio TD vs TER).
  * Validazione del periodo comune minimo: niente classifiche su 3 mesi di dati.
  * Errori per singolo ETF tracciati e riportati, non silenziati.
  * Controllo coerenza valuta (indice in EUR -> warning se ETF non in EUR).

Il modulo NON dipende da Streamlit: e usabile da CLI, notebook o GUI.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_INDEX_BASE_URL = (
    "https://raw.githubusercontent.com/paolocole/"
    "Stock-Indexes-Historical-Data/main/DAILY/NET/EUR/"
)

# Universo ETF candidato (dal notebook originale): ticker, TER %, indice di riferimento.
DEFAULT_ETF_UNIVERSE = pd.DataFrame(
    [
        ("IWDA.AS", 0.20, "WORLD"),
        ("XDWD.DE", 0.12, "WORLD"),
        ("SPPW.DE", 0.12, "WORLD"),
        ("VWCE.DE", 0.19, "WORLD"),      # NB: VWCE = FTSE All-World (include EM): confronto solo indicativo
        ("IUSQ.DE", 0.20, "WORLD"),
        ("IMAE.AS", 0.12, "EUROPE"),
        ("XMEU.DE", 0.12, "EUROPE"),
        ("CEU2.PA", 0.12, "EUROPE"),
        ("CEBZ.DE", 0.12, "EUROPE"),
        ("EMIM.AS", 0.18, "EMERGING MARKETS IMI"),
        ("XMME.DE", 0.18, "EMERGING MARKETS"),
        ("IEMA.AS", 0.18, "EMERGING MARKETS"),
        ("EMMUSC.MI", 0.15, "EMERGING MARKETS"),
        ("AEEM.PA", 0.20, "EMERGING MARKETS"),
        ("AEME.AS", 0.18, "EMERGING MARKETS"),
    ],
    columns=["ticker", "ter", "index"],
)

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Caricamento serie indici
# --------------------------------------------------------------------------- #
def _to_monthly(series: pd.Series) -> pd.Series:
    """Serie di prezzo -> ultimo valore di ogni mese, indicizzata per Period mensile."""
    s = series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    return s.resample("ME").last().to_period("M")


def load_index_panel_from_mapping(
    mapping: pd.DataFrame,
    base_url: str = DEFAULT_INDEX_BASE_URL,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Costruisce il pannello degli indici dal mapping (foglio 'selezionati').

    Si attende un DataFrame con:
      - indice di riga = nome dell'indice (es. 'WORLD', 'EUROPE', ...)
      - colonne 'Path' e 'File' che compongono l'URL del CSV nel repo.

    Ritorna (pannello_mensile, errori) dove errori = {indice: messaggio}.
    """
    required = {"Path", "File"}
    if not required.issubset(set(mapping.columns)):
        raise ValueError(
            f"Il mapping deve contenere le colonne {required}. "
            f"Trovate: {list(mapping.columns)}"
        )

    panel = pd.DataFrame()
    errors: dict[str, str] = {}
    for name in mapping.index:
        try:
            path = str(mapping.loc[name, "Path"]).replace("\\", "/")
            file = str(mapping.loc[name, "File"])
            url = f"{base_url}{path}/{file}.csv".replace("\\", "/")
            raw = pd.read_csv(url, index_col=0)
            col = _to_monthly(raw.iloc[:, 0]).rename(str(name))
            panel = pd.concat([panel, col], axis=1)
        except Exception as exc:  # noqa: BLE001 - vogliamo il motivo, per indice
            errors[str(name)] = f"{type(exc).__name__}: {exc}"
    return panel.sort_index(), errors


def load_index_panel_from_csv(file_or_buffer) -> pd.DataFrame:
    """
    Carica un pannello indici gia pronto: prima colonna = data, una colonna per indice.
    """
    raw = pd.read_csv(file_or_buffer, index_col=0)
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw[~raw.index.isna()].sort_index()
    return raw.resample("ME").last().to_period("M")


# --------------------------------------------------------------------------- #
# Download ETF (Yahoo Finance)
# --------------------------------------------------------------------------- #
def download_etf(ticker: str, max_retries: int = 3) -> pd.Series:
    """
    Scarica la serie total-return mensile di un ETF (Close auto-adjusted).
    Solleva ValueError se non ci sono dati.
    """
    import time

    import yfinance as yf

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                period="max",
                auto_adjust=True,       # Close = total return (dividendi reinvestiti)
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                raise ValueError("nessun dato restituito da Yahoo Finance")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns:
                raise ValueError(f"colonna 'Close' assente (colonne: {list(df.columns)})")
            return _to_monthly(df["Close"]).rename(ticker)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    raise ValueError(f"download fallito per {ticker}: {last_err}")


def get_etf_currency(ticker: str) -> Optional[str]:
    """Valuta di quotazione dell'ETF, se disponibile (best-effort, non bloccante)."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).fast_info
        return getattr(info, "currency", None) or info.get("currency")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Metriche
# --------------------------------------------------------------------------- #
@dataclass
class TrackingResult:
    ticker: str
    index: str
    ter_pct: float
    n_months: int
    start: Optional[str]
    end: Optional[str]
    td_pct: Optional[float] = None        # Tracking Difference annualizzato (%)
    gap_vs_ter_pct: Optional[float] = None  # TD + TER (%)
    te_pct: Optional[float] = None         # Tracking Error annualizzato (%)
    currency: Optional[str] = None
    warning: str = ""
    error: str = ""

    def as_row(self) -> dict:
        return {
            "ETF": self.ticker,
            "Indice": self.index,
            "TER %": round(self.ter_pct, 3),
            "Tracking Diff % (ann.)": None if self.td_pct is None else round(self.td_pct, 3),
            "Gap vs TER %": None if self.gap_vs_ter_pct is None else round(self.gap_vs_ter_pct, 3),
            "Tracking Error % (ann.)": None if self.te_pct is None else round(self.te_pct, 3),
            "Mesi comuni": self.n_months,
            "Periodo": "" if not self.start else f"{self.start} -> {self.end}",
            "Valuta": self.currency or "",
            "Note": "; ".join(x for x in [self.warning, self.error] if x),
        }


def _cagr(levels: pd.Series) -> float:
    rets = levels.pct_change().dropna()
    n = len(rets)
    if n == 0:
        return np.nan
    growth = float((1.0 + rets).prod())
    return growth ** (MONTHS_PER_YEAR / n) - 1.0


def compute_tracking(
    etf: pd.Series,
    index: pd.Series,
    ticker: str,
    index_name: str,
    ter_pct: float,
    min_months: int = 24,
    currency: Optional[str] = None,
) -> TrackingResult:
    """Allinea ETF e indice sul periodo comune e calcola TD, gap-vs-TER e TE."""
    merged = pd.concat([etf.rename("etf"), index.rename("idx")], axis=1).dropna()
    n = len(merged)
    start = str(merged.index[0]) if n else None
    end = str(merged.index[-1]) if n else None
    res = TrackingResult(
        ticker=ticker, index=index_name, ter_pct=ter_pct,
        n_months=n, start=start, end=end, currency=currency,
    )

    if currency and currency.upper() != "EUR":
        res.warning = f"valuta {currency} != EUR (indice in EUR): possibile mismatch FX"

    if n < min_months:
        res.error = f"dati insufficienti ({n} < {min_months} mesi)"
        return res

    td = _cagr(merged["etf"]) - _cagr(merged["idx"])
    # Tracking Error su rendimenti mensili non sovrapposti, annualizzato.
    active = merged["etf"].pct_change() - merged["idx"].pct_change()
    te = active.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)

    res.td_pct = td * 100.0
    res.te_pct = te * 100.0
    res.gap_vs_ter_pct = (td * 100.0) + ter_pct  # atteso ~0
    return res


def resolve_index_column(index_name: str, panel: pd.DataFrame) -> Optional[str]:
    """Match robusto del nome indice contro le colonne del pannello."""
    if index_name in panel.columns:
        return index_name
    norm = lambda s: str(s).upper().replace("MSCI", "").strip()
    target = norm(index_name)
    for col in panel.columns:
        if norm(col) == target:
            return col
    return None


def run_analysis(
    etf_table: pd.DataFrame,
    index_panel: pd.DataFrame,
    min_months: int = 24,
    check_currency: bool = False,
    downloader=download_etf,
) -> pd.DataFrame:
    """
    Esegue l'analisi completa.

    etf_table: colonne ['ticker', 'ter', 'index'].
    index_panel: pannello mensile, una colonna per indice.
    Ritorna un DataFrame ordinato per Indice e poi per Tracking Error crescente.
    """
    rows: list[dict] = []
    etf_cache: dict[str, pd.Series] = {}

    for _, r in etf_table.iterrows():
        ticker = str(r["ticker"]).strip()
        index_name = str(r["index"]).strip()
        try:
            ter_pct = float(r["ter"])
        except (TypeError, ValueError):
            ter_pct = np.nan

        if not ticker or not index_name:
            continue

        col = resolve_index_column(index_name, index_panel)
        if col is None:
            rows.append(
                TrackingResult(
                    ticker=ticker, index=index_name, ter_pct=ter_pct,
                    n_months=0, start=None, end=None,
                    error=f"indice '{index_name}' assente nel pannello",
                ).as_row()
            )
            continue

        try:
            if ticker not in etf_cache:
                etf_cache[ticker] = downloader(ticker)
            etf_series = etf_cache[ticker]
        except Exception as exc:  # noqa: BLE001
            rows.append(
                TrackingResult(
                    ticker=ticker, index=index_name, ter_pct=ter_pct,
                    n_months=0, start=None, end=None,
                    error=str(exc),
                ).as_row()
            )
            continue

        currency = get_etf_currency(ticker) if check_currency else None
        res = compute_tracking(
            etf_series, index_panel[col], ticker, index_name,
            ter_pct, min_months=min_months, currency=currency,
        )
        rows.append(res.as_row())

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Ordina: per indice, poi tracking error crescente (NaN in fondo).
    out["_te_sort"] = out["Tracking Error % (ann.)"].fillna(np.inf)
    out = (
        out.sort_values(["Indice", "_te_sort"])
        .drop(columns="_te_sort")
        .reset_index(drop=True)
    )
    return out


def build_base100(
    tickers: list[str],
    index_col: str,
    index_panel: pd.DataFrame,
    etf_cache: dict[str, pd.Series],
) -> pd.DataFrame:
    """Serie base-100 di ETF + indice sul periodo comune, per il grafico."""
    cols = {}
    if index_col in index_panel.columns:
        cols[index_col] = index_panel[index_col]
    for t in tickers:
        if t in etf_cache:
            cols[t] = etf_cache[t]
    if not cols:
        return pd.DataFrame()
    df = pd.concat(cols, axis=1).dropna()
    if df.empty:
        return df
    return 100.0 * df / df.iloc[0]
