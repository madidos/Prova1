"""
app.py — Interfaccia web (Streamlit) per l'analisi della qualita di replica degli ETF.

Avvio:
    pip install -r requirements.txt
    streamlit run app.py

Si apre nel browser su http://localhost:8501
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import etf_tracking as et

st.set_page_config(page_title="ETF Tracking & TER", page_icon="📊", layout="wide")

st.title("📊 Analisi replica ETF vs indice (Tracking & TER)")
st.caption(
    "Per ogni ETF confronta la serie total-return (Yahoo Finance) con l'indice "
    "NET-return in EUR e calcola metriche di replica oneste."
)

with st.expander("Come leggere le metriche", expanded=False):
    st.markdown(
        """
- **Tracking Difference (TD)** — quanto l'ETF resta indietro rispetto all'indice ogni anno
  (CAGR ETF − CAGR indice). Atteso ≈ **−TER**, scalato per il rendimento dell'indice.
- **Gap vs TER** — `TD + TER`. Vicino a **0** = efficiente; molto **negativo** = perde più
  della sua commissione (drag fiscale o replica scarsa); **positivo** = il prestito titoli
  compensa la commissione.
- **Tracking Error (TE)** — oscillazione delle differenze di rendimento mensili (annualizzata).
  Più **basso** = replica più fedele. È l'indicatore di **affidabilità** su cui ordino la classifica.
        """
    )

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Impostazioni")
    min_months = st.slider("Mesi comuni minimi", 12, 120, 24, step=6,
                           help="Sotto questa soglia l'ETF viene segnalato, non classificato.")
    check_currency = st.checkbox("Controlla valuta ETF (più lento)", value=False)
    base_url = st.text_input("Base URL indici (repo)", value=et.DEFAULT_INDEX_BASE_URL)
    st.divider()
    st.caption("Dati ETF: Yahoo Finance · Indici: serie NET EUR fornite dall'utente.")


@st.cache_data(show_spinner=False, ttl=3600)
def cached_download(ticker: str) -> pd.Series:
    return et.download_etf(ticker)


# --------------------------------------------------------------------------- #
# 1) Sorgente indici
# --------------------------------------------------------------------------- #
st.subheader("1 · Serie storiche degli indici (benchmark)")

src = st.radio(
    "Sorgente",
    ["Mapping Excel (foglio 'selezionati')", "Pannello indici CSV già pronto"],
    horizontal=True,
)

if "index_panel" not in st.session_state:
    st.session_state.index_panel = None

if src.startswith("Mapping"):
    up = st.file_uploader(
        "Carica `etfs_with_msci.xlsx` (deve contenere il foglio 'selezionati' con colonne Path/File)",
        type=["xlsx", "xls"],
    )
    sheet = st.text_input("Nome foglio", value="selezionati")
    if up and st.button("Carica indici dal mapping"):
        try:
            mapping = pd.read_excel(up, sheet_name=sheet, index_col=0)
            with st.spinner("Scarico le serie degli indici dal repo…"):
                panel, errs = et.load_index_panel_from_mapping(mapping, base_url=base_url)
            st.session_state.index_panel = panel
            st.success(f"Caricati {panel.shape[1]} indici, {panel.shape[0]} mesi.")
            if errs:
                with st.expander(f"⚠️ {len(errs)} indici non caricati"):
                    st.json(errs)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Errore nel mapping: {exc}")
else:
    up = st.file_uploader(
        "Carica un CSV: prima colonna = data, una colonna per indice (livelli di prezzo)",
        type=["csv"],
    )
    if up and st.button("Carica pannello indici"):
        try:
            st.session_state.index_panel = et.load_index_panel_from_csv(up)
            p = st.session_state.index_panel
            st.success(f"Caricati {p.shape[1]} indici, {p.shape[0]} mesi.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Errore nel CSV: {exc}")

panel = st.session_state.index_panel
if panel is not None and not panel.empty:
    st.write("**Indici disponibili:**", ", ".join(map(str, panel.columns)))
    st.dataframe(panel.tail(3), width='stretch')

# --------------------------------------------------------------------------- #
# 2) Tabella ETF
# --------------------------------------------------------------------------- #
st.subheader("2 · ETF da analizzare")
st.caption("Modifica liberamente: aggiungi/rimuovi righe. 'index' deve combaciare con un indice del pannello.")

etf_table = st.data_editor(
    et.DEFAULT_ETF_UNIVERSE.copy(),
    num_rows="dynamic",
    width='stretch',
    column_config={
        "ticker": st.column_config.TextColumn("Ticker (Yahoo)", help="es. XDWD.DE, IMAE.AS"),
        "ter": st.column_config.NumberColumn("TER %", min_value=0.0, max_value=2.0, step=0.01, format="%.2f"),
        "index": st.column_config.TextColumn("Indice di riferimento"),
    },
    key="etf_editor",
)

# --------------------------------------------------------------------------- #
# 3) Esecuzione
# --------------------------------------------------------------------------- #
st.subheader("3 · Analisi")

run = st.button("▶️ Esegui analisi", type="primary",
                disabled=(panel is None or panel.empty))
if panel is None or panel.empty:
    st.info("Carica prima le serie degli indici (sezione 1).")

if run:
    etf_cache: dict[str, pd.Series] = {}

    def downloader(t: str) -> pd.Series:
        s = cached_download(t)
        etf_cache[t] = s
        return s

    prog = st.progress(0.0, text="Scarico e analizzo gli ETF…")
    rows = []
    n = max(len(etf_table), 1)
    for i, (_, r) in enumerate(etf_table.iterrows()):
        single = et.run_analysis(pd.DataFrame([r]), panel,
                                 min_months=min_months,
                                 check_currency=check_currency,
                                 downloader=downloader)
        rows.append(single)
        prog.progress((i + 1) / n, text=f"Analizzato {r['ticker']}")
    prog.empty()

    results = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    results["_te_sort"] = results["Tracking Error % (ann.)"].fillna(float("inf"))
    results = results.sort_values(["Indice", "_te_sort"]).drop(columns="_te_sort").reset_index(drop=True)

    st.session_state.results = results
    st.session_state.etf_cache = etf_cache

# --------------------------------------------------------------------------- #
# 4) Risultati
# --------------------------------------------------------------------------- #
if st.session_state.get("results") is not None:
    results = st.session_state.results
    st.subheader("4 · Risultati")

    ok = results[results["Note"].fillna("") == ""]
    bad = results[results["Note"].fillna("") != ""]

    for idx_name, group in ok.groupby("Indice"):
        st.markdown(f"#### {idx_name}")
        g = group.drop(columns=["Indice"]).reset_index(drop=True)
        styler = g.style.highlight_min(
            subset=["Tracking Error % (ann.)"], color="#1b5e20"
        ).format(precision=3)
        st.dataframe(styler, width='stretch')
        st.caption("🟩 = miglior Tracking Error del gruppo (replica più fedele).")

    if not bad.empty:
        with st.expander(f"⚠️ {len(bad)} ETF non valutati (dati insufficienti / errori)"):
            st.dataframe(bad[["ETF", "Indice", "Note"]], width='stretch')

    st.download_button(
        "💾 Scarica risultati (CSV)",
        results.to_csv(index=False).encode("utf-8"),
        file_name="analisi_tracking_etf.csv",
        mime="text/csv",
    )

    # ---- Grafici ----
    cache = st.session_state.get("etf_cache", {})
    indices_with_data = sorted(ok["Indice"].unique())
    if indices_with_data and cache:
        sel = st.selectbox("Indice da visualizzare", indices_with_data)
        col = et.resolve_index_column(sel, panel)
        tickers = ok[ok["Indice"] == sel]["ETF"].tolist()

        # ---- Bar chart differenze YoY con linea TER ----
        st.markdown("#### Differenza rendimento annuo (ETF − indice)")
        st.caption("Barre verdi = ETF sopra l'indice · rosse = sotto · linea tratteggiata = −TER atteso")
        for ticker in tickers:
            if ticker not in cache or col is None:
                continue
            ter_vals = ok[ok["ETF"] == ticker]["TER %"].values
            if len(ter_vals) == 0:
                continue
            ter_val = float(ter_vals[0])
            diff = et.compute_yoy_diff(cache[ticker], panel[col])
            if diff.empty:
                continue
            diff_df = pd.DataFrame({
                "data": diff.index.to_timestamp(),
                "diff": diff.values * 100,
            })
            bars = (
                alt.Chart(diff_df)
                .mark_bar(size=4)
                .encode(
                    x=alt.X("data:T", title=None),
                    y=alt.Y("diff:Q", title="Diff % (YoY)"),
                    color=alt.condition(
                        alt.datum.diff > 0,
                        alt.value("#2e7d32"),
                        alt.value("#c62828"),
                    ),
                    tooltip=["data:T", alt.Tooltip("diff:Q", format=".2f")],
                )
                .properties(title=f"{ticker}  vs  {sel}", height=320)
            )
            rule = (
                alt.Chart(pd.DataFrame({"y": [-ter_val]}))
                .mark_rule(color="red", strokeDash=[6, 3], size=1)
                .encode(y="y:Q")
            )
            st.altair_chart(bars + rule, use_container_width=True)

        # ---- Base 100 ----
        st.markdown("#### Andamento (base 100) ETF vs indice")
        base = et.build_base100(tickers, col, panel, cache)
        if not base.empty:
            base.index = base.index.to_timestamp()
            st.line_chart(base, width='stretch')
        else:
            st.info("Nessun periodo comune sufficiente per il grafico.")

        st.markdown("#### TER e metriche di replica (per indice selezionato)")
        metrics = (
            ok[ok["Indice"] == sel][["ETF", "TER %", "Tracking Diff % (ann.)", "Gap vs TER %"]]
            .dropna()
            .set_index("ETF")
        )
        if not metrics.empty:
            st.bar_chart(metrics)
            st.caption(
                "**TER %** = commissione annua dichiarata · "
                "**Tracking Diff %** = quanto l'ETF resta indietro rispetto all'indice (CAGR ETF − CAGR indice) · "
                "**Gap vs TER %** = TD + TER (vicino a 0 = efficiente)."
            )
