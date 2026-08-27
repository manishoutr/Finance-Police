from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from reconciler import run_reconciliation  # noqa: E402
from evaluate_reconciliation import evaluate, load_inputs, write_outputs  # noqa: E402
from classify_exceptions import classify_exceptions  # noqa: E402
from finance_assistant import ask_finance_controller  # noqa: E402
from finance_tools import get_cash_position  # noqa: E402
from input_validation import validate_source_frames  # noqa: E402
from human_review import (  # noqa: E402
    approve_match,
    reject_match,
    get_audit_log,
    get_operational_status_map,
)

OUTPUTS = BASE_DIR / "outputs"
RAW = BASE_DIR / "data" / "raw"
REQUIRED_UPLOADS = {"invoices.csv", "payments.csv", "settlements.csv"}
HELD_OUT = BASE_DIR / "evaluation" / "held_out_results.json"

st.set_page_config(
    page_title="Finance Controller | Reconciliation Operations",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#142033; --muted:#667085; --line:#e6eaf0; --panel:#ffffff; --bg:#f5f7fa; --navy:#132238; }
    .stApp { background:var(--bg); color:var(--ink); }
    .block-container { max-width:1500px; padding-top:1.2rem; padding-bottom:2rem; }
    [data-testid="stSidebar"] { background:#101b2d; }
    [data-testid="stSidebar"] * { color:#eef3f8 !important; }
    .brand { padding:8px 4px 18px; }
    .brand-title { font-size:21px; font-weight:800; letter-spacing:-.3px; }
    .brand-sub { color:#aab7c9 !important; font-size:12px; margin-top:3px; }
    .hero { background:var(--navy); color:white; border-radius:16px; padding:24px 28px; margin-bottom:18px; }
    .hero-title { font-size:31px; font-weight:800; letter-spacing:-.7px; }
    .hero-sub { color:#b9c5d5; margin-top:4px; font-size:14px; }
    .eyebrow { color:#64748b; text-transform:uppercase; letter-spacing:.08em; font-size:11px; font-weight:800; }
    .kpi { background:var(--panel); border:1px solid var(--line); border-radius:13px; padding:16px; min-height:102px; }
    .kpi-label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; font-weight:800; }
    .kpi-value { color:var(--ink); font-size:27px; font-weight:800; margin-top:7px; }
    .kpi-note { color:#8a94a6; font-size:11px; margin-top:4px; }
    .panel { background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px; }
    .status-pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:4px 9px; font-size:11px; font-weight:800; }
    .demo-note { background:#fff9e8; border:1px solid #f1dfaa; border-radius:10px; padding:10px 12px; color:#624f1f; font-size:12px; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def current_currency() -> str:
    """Return the single currency used by the currently loaded invoice source."""
    path = RAW / "invoices.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, usecols=["currency"])
            values = sorted({str(v).strip().upper() for v in df["currency"].dropna() if str(v).strip()})
            if len(values) == 1:
                return values[0]
        except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
            pass
    return "USD"


def money(value, currency: str | None = None) -> str:
    if value is None or pd.isna(value):
        value = 0
    return f"{currency or current_currency()} {float(value):,.2f}"


def cash_money(section: dict) -> str:
    by_currency = section.get("by_currency") or {}
    if len(by_currency) == 1:
        currency, amount = next(iter(by_currency.items()))
        return f"{currency} {float(amount):,.2f}"
    if not by_currency:
        return "0.00"
    return "; ".join(f"{k} {float(v):,.2f}" for k, v in sorted(by_currency.items()))


def load_current() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    recon_path = OUTPUTS / "reconciliation_results.csv"
    exc_path = OUTPUTS / "exceptions.csv"
    eval_path = OUTPUTS / "evaluation_results.json"
    recon = pd.read_csv(recon_path) if recon_path.exists() else pd.DataFrame()
    exc = pd.read_csv(exc_path) if exc_path.exists() else pd.DataFrame()
    metrics = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    return recon, exc, metrics


def load_held_out() -> dict:
    if not HELD_OUT.exists():
        return {}
    return json.loads(HELD_OUT.read_text(encoding="utf-8"))


def _ground_truth_matches_current_dataset(invoices: pd.DataFrame, payments: pd.DataFrame) -> bool:
    """Return True only when bundled ground truth belongs to the current source data.

    Uploaded operational data must never inherit benchmark metrics from a
    different dataset. This check is deliberately strict and is used only to
    decide whether evaluation is appropriate, not to change evaluation logic.
    """
    gt_path = BASE_DIR / "data" / "ground_truth" / "ground_truth.csv"
    if not gt_path.exists():
        return False
    try:
        ground_truth = pd.read_csv(gt_path)
        invoice_ids = set(invoices["invoice_id"].astype(str))
        gt_invoice_ids = set(ground_truth["invoice_id"].dropna().astype(str))
        if invoice_ids != gt_invoice_ids:
            return False

        payment_ids = set(payments["payment_id"].astype(str))
        gt_payment_ids: set[str] = set()
        for value in ground_truth["payment_ids"].dropna():
            gt_payment_ids.update(
                token.strip() for token in str(value).split(",") if token.strip()
            )
        # Ground truth may intentionally omit payment IDs for missing-payment
        # invoices, so compare only the IDs it actually claims.
        return gt_payment_ids.issubset(payment_ids)
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError, ValueError):
        return False


def run_pipeline() -> None:
    invoices = pd.read_csv(RAW / "invoices.csv")
    payments = pd.read_csv(RAW / "payments.csv")
    settlements = pd.read_csv(RAW / "settlements.csv")
    validate_source_frames({"invoices.csv": invoices, "payments.csv": payments, "settlements.csv": settlements})
    run_reconciliation(BASE_DIR)

    # Evaluate only when the bundled ground truth demonstrably belongs to the
    # current dataset. Otherwise remove stale evaluation artifacts so the UI
    # cannot present unrelated benchmark metrics as operational accuracy.
    gt_path = BASE_DIR / "data" / "ground_truth" / "ground_truth.csv"
    if _ground_truth_matches_current_dataset(invoices, payments):
        results, ground_truth = load_inputs(
            OUTPUTS / "reconciliation_results.csv",
            gt_path,
        )
        metrics = evaluate(results, ground_truth)
        write_outputs(metrics, OUTPUTS)
    else:
        for stale in (OUTPUTS / "evaluation_results.json", OUTPUTS / "evaluation_report.csv"):
            if stale.exists():
                stale.unlink()

    exceptions = classify_exceptions(
        pd.read_csv(OUTPUTS / "reconciliation_results.csv"),
        payments=payments,
    )
    exceptions.to_csv(OUTPUTS / "exceptions.csv", index=False)


def save_uploaded(files: list) -> None:
    payloads: dict[str, bytes] = {}
    frames: dict[str, pd.DataFrame] = {}
    seen = set()
    for uploaded in files:
        name = Path(uploaded.name).name
        if name in REQUIRED_UPLOADS:
            data = uploaded.getvalue()
            payloads[name] = data
            frames[name] = pd.read_csv(io.BytesIO(data))
            seen.add(name)
    missing = REQUIRED_UPLOADS - seen
    if missing:
        raise ValueError(f"Please upload all three files: {', '.join(sorted(missing))}")
    validate_source_frames(frames)
    RAW.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        (RAW / name).write_bytes(data)


def operational_status_map() -> dict[tuple[str, str], str]:
    return get_operational_status_map(BASE_DIR)


def with_customer_name(df: pd.DataFrame) -> pd.DataFrame:
    """Add customer names from the authoritative invoice source for display only."""
    if df.empty:
        return df.copy()
    out = df.copy()
    if "customer_name" in out.columns:
        return out
    invoice_path = RAW / "invoices.csv"
    if invoice_path.exists() and "invoice_id" in out.columns:
        invoices = pd.read_csv(invoice_path, usecols=["invoice_id", "customer_name"])
        invoices["invoice_id"] = invoices["invoice_id"].astype(str)
        out["invoice_id"] = out["invoice_id"].astype(str)
        out = out.merge(invoices.drop_duplicates("invoice_id"), on="invoice_id", how="left")
    else:
        out["customer_name"] = ""
    return out


def with_operational_status(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    mapping = operational_status_map()
    out["operational_status"] = out.apply(
        lambda r: mapping.get(
            (str(r.get("invoice_id")), "" if pd.isna(r.get("payment_id")) else str(r.get("payment_id"))),
            str(r.get("status", "UNMATCHED")),
        ), axis=1,
    )
    return out


def render_kpis(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, note) in zip(cols, items):
        with col:
            note_html = f'<div class="kpi-note">{note}</div>' if note else ""
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{note_html}</div>',
                unsafe_allow_html=True,
            )


def exception_view(recon: pd.DataFrame, exceptions: pd.DataFrame) -> pd.DataFrame:
    if exceptions.empty:
        return exceptions.copy()
    out = exceptions.copy()
    mapping = operational_status_map()
    out["status"] = out.apply(
        lambda r: mapping.get(
            (str(r.get("invoice_id")), "" if pd.isna(r.get("payment_id")) else str(r.get("payment_id"))),
            "UNMATCHED",
        ), axis=1,
    )
    if not recon.empty:
        lookup = recon[["invoice_id", "payment_id", "status"]].copy()
        lookup["_i"] = lookup["invoice_id"].astype(str)
        lookup["_p"] = lookup["payment_id"].fillna("").astype(str)
        out["_i"] = out["invoice_id"].astype(str)
        out["_p"] = out["payment_id"].fillna("").astype(str)
        out = out.merge(lookup[["_i", "_p", "status"]].rename(columns={"status": "original_status"}), on=["_i", "_p"], how="left")
        out.drop(columns=["_i", "_p"], inplace=True)
    else:
        out["original_status"] = "UNMATCHED"
    return out


# Header
st.markdown(
    '<div class="hero"><div class="hero-title">Finance Controller</div><div class="hero-sub">Reconciliation operations • exceptions • cash visibility • human controls</div></div>',
    unsafe_allow_html=True,
)

recon, exceptions, metrics = load_current()
held_out = load_held_out()

NAV = ["OVERVIEW", "RECONCILIATION", "EXCEPTIONS", "CASH POSITION", "ASK CONTROLLER", "AUDIT TRAIL", "BENCHMARK"]
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-title">Finance Controller</div><div class="brand-sub">Control center</div></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", NAV, label_visibility="collapsed")
    st.divider()
    st.caption("Source of truth: deterministic backend")
    st.caption("AI is read-only; human decisions are audited")
    with st.expander("Data refresh", expanded=False):
        uploads = st.file_uploader("Upload source CSVs", type="csv", accept_multiple_files=True, key="nav_uploads")
        if st.button("Run reconciliation", type="primary", use_container_width=True, key="nav_run"):
            if not uploads:
                st.error("Upload invoices.csv, payments.csv and settlements.csv first.")
            else:
                try:
                    save_uploaded(uploads)
                    with st.spinner("Refreshing controls..."):
                        run_pipeline()
                    st.success("Controls refreshed.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

# ---------------- Overview ----------------
if page == "OVERVIEW":
    st.markdown('<div class="eyebrow">Executive control view</div>', unsafe_allow_html=True)
    st.title("Overview")
    if not metrics:
        st.info("No evaluation output is available yet. Run reconciliation from the Data refresh panel.")
    render_kpis([
        ("Records processed", f"{metrics.get('records_processed', 0):,}", None),
        ("Automatic match rate", f"{metrics.get('automatic_match_rate', 0):.1%}", None),
        ("Precision", f"{metrics.get('precision', 0):.1%}", None),
        ("Recall", f"{metrics.get('recall', 0):.1%}", None),
        ("F1", f"{metrics.get('f1_score', 0):.1%}", None),
        ("Matched", f"{metrics.get('automatic_matches', 0):,}", None),
        ("Review", f"{metrics.get('review_records', 0):,}", "Human attention"),
        ("Unmatched", f"{metrics.get('unmatched_records', 0):,}", "Invoice-level"),
        ("Exception value", money(metrics.get('total_value_exceptions', 0)), "Current dataset"),
    ])
    st.markdown("### Control signals")
    left, right = st.columns(2)
    if not recon.empty:
        status_counts = recon["status"].value_counts().reindex(["MATCHED", "REVIEW", "UNMATCHED"], fill_value=0).reset_index()
        status_counts.columns = ["Status", "Count"]
        with left:
            st.plotly_chart(px.bar(status_counts, x="Status", y="Count", text="Count", template="plotly_white", title="Reconciliation status"), use_container_width=True)
    if not exceptions.empty:
        ex_counts = exceptions["exception_type"].value_counts().reset_index()
        ex_counts.columns = ["Exception type", "Count"]
        with right:
            st.plotly_chart(px.bar(ex_counts, x="Count", y="Exception type", orientation="h", template="plotly_white", title="Exception categories"), use_container_width=True)
        value_df = exceptions.copy()
        value_df["exception_value"] = pd.to_numeric(value_df.get("invoice_amount"), errors="coerce").fillna(0).abs()
        value_df = value_df.groupby("exception_type", as_index=False)["exception_value"].sum().sort_values("exception_value", ascending=True)
        st.plotly_chart(px.bar(value_df, x="exception_value", y="exception_type", orientation="h", text="exception_value", template="plotly_white", title="Exception monetary value"), use_container_width=True)

# ---------------- Reconciliation ----------------
elif page == "RECONCILIATION":
    st.markdown('<div class="eyebrow">Transaction matching</div>', unsafe_allow_html=True)
    st.title("Reconciliation")
    if recon.empty:
        st.warning("No reconciliation results available.")
    else:
        data = with_customer_name(with_operational_status(recon))
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            statuses = st.multiselect("Status", ["MATCHED", "REVIEW", "UNMATCHED"], default=["MATCHED", "REVIEW", "UNMATCHED"])
        with c2:
            conf_range = st.slider("Confidence", 0, 100, (0, 100))
        with c3:
            query = st.text_input("Search", placeholder="Invoice, payment, customer, reference or reason")
        view = data[data["status"].isin(statuses)].copy()
        view["confidence_score"] = pd.to_numeric(view["confidence_score"], errors="coerce")
        view = view[view["confidence_score"].fillna(0).between(conf_range[0], conf_range[1])]
        if query:
            q = query.casefold()
            mask = view.astype(str).apply(lambda col: col.str.casefold().str.contains(q, regex=False)).any(axis=1)
            view = view[mask]
        display = view[["customer_name", "invoice_id", "payment_id", "invoice_amount", "payment_amount", "confidence_score", "status", "operational_status", "reason"]].rename(columns={"customer_name": "Customer", "confidence_score": "Confidence", "status": "Original", "operational_status": "Operational", "invoice_id": "Invoice", "payment_id": "Payment", "invoice_amount": "Invoice amount", "payment_amount": "Payment amount", "reason": "Reason"})
        st.dataframe(display, use_container_width=True, hide_index=True, height=430)
        st.caption(f"Showing {len(view):,} of {len(data):,} records")

        st.markdown("### Record details")
        if not view.empty:
            labels = [f"{r.invoice_id} • {r.payment_id if pd.notna(r.payment_id) else 'No payment'}" for r in view.itertuples()]
            selected = st.selectbox("Select a record", labels)
            row = view.iloc[labels.index(selected)]
            with st.expander("Matching signals", expanded=True):
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Invoice amount", money(row.get("invoice_amount")))
                d2.metric("Payment amount", money(row.get("payment_amount")))
                d3.metric("Amount difference", money(row.get("amount_difference")))
                d4.metric("Confidence", f"{float(row.get('confidence_score', 0)):.1f}")
                s1, s2, s3 = st.columns(3)
                s1.write(f"**Customer similarity:** {row.get('customer_similarity', 'N/A')}")
                s2.write(f"**Date difference:** {row.get('date_difference', 'N/A')}")
                s3.write(f"**Reference match:** {row.get('reference_match', 'N/A')}")
                st.write(f"**Customer:** {row.get('customer_name', 'N/A')}  ")
                st.write(f"**Reason:** {row.get('reason', 'N/A')}")
            st.markdown("#### Match evidence")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Customer similarity", f"{float(row.get('customer_similarity', 0) or 0):.1f}%")
            e2.metric("Date difference", f"{float(row.get('date_difference', 0) or 0):.0f} days")
            e3.metric("Reference", "MATCH" if bool(row.get("reference_match", False)) else "MISSING")
            e4.metric("Amount variance", money(row.get("amount_difference")))
            operational = str(row.get("operational_status", row.get("status", "UNMATCHED")))
            if operational in {"APPROVED", "REJECTED"}:
                st.success(f"Human decision recorded: **{operational}**") if operational == "APPROVED" else st.error(f"Human decision recorded: **{operational}**")
                audit = get_audit_log(BASE_DIR)
                if not audit.empty:
                    inv_key = str(row["invoice_id"])
                    pay_key = "" if pd.isna(row.get("payment_id")) else str(row.get("payment_id"))
                    matches = audit[(audit["invoice_id"].astype(str) == inv_key) & (audit["payment_id"].fillna("").astype(str) == pay_key)]
                    if not matches.empty:
                        latest = matches.iloc[-1]
                        st.caption(f"Decision: {latest.get('decision')} • {latest.get('timestamp')} • Audit ID {latest.get('audit_id')}")
            if str(row.get("status")) == "REVIEW":
                st.markdown("### Human decision")
                st.info("REVIEW cases require an explicit human decision. The original reconciliation result will not be changed.")
                reason = st.text_area("Decision reason", key=f"decision_reason_{row['invoice_id']}_{row.get('payment_id')}", placeholder="Explain what you verified before deciding.")
                a, b = st.columns(2)
                payment_id = None if pd.isna(row.get("payment_id")) else str(row.get("payment_id"))
                with a:
                    if st.button("APPROVE MATCH", type="primary", use_container_width=True):
                        try:
                            record = approve_match(str(row["invoice_id"]), payment_id, reason, BASE_DIR)
                            st.success(f"Approved. Audit ID: {record['audit_id']}")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                with b:
                    if st.button("REJECT MATCH", use_container_width=True):
                        try:
                            record = reject_match(str(row["invoice_id"]), payment_id, reason, BASE_DIR)
                            st.success(f"Rejected. Audit ID: {record['audit_id']}")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

# ---------------- Exceptions ----------------
elif page == "EXCEPTIONS":
    st.markdown('<div class="eyebrow">Exception management</div>', unsafe_allow_html=True)
    st.title("Exceptions")
    ex = exception_view(recon, exceptions)
    if ex.empty:
        st.success("No exceptions found.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            types = st.multiselect("Exception type", sorted(ex["exception_type"].dropna().unique()), default=sorted(ex["exception_type"].dropna().unique()))
        with c2:
            severities = st.multiselect("Severity", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
        with c3:
            statuses = st.multiselect("Status", sorted(ex["status"].dropna().unique()), default=sorted(ex["status"].dropna().unique()))
        ex["monetary_impact"] = pd.to_numeric(ex.get("invoice_amount"), errors="coerce").fillna(pd.to_numeric(ex.get("amount_difference"), errors="coerce")).fillna(0).abs()
        ex["priority"] = ex.apply(lambda r: "CRITICAL" if float(r.get("monetary_impact", 0) or 0) >= 100000 else str(r.get("severity", "LOW")), axis=1)
        ex["priority_rank"] = ex["priority"].map({"CRITICAL":0, "HIGH":1, "MEDIUM":2, "LOW":3}).fillna(9)
        ex = with_customer_name(ex)
        view = ex[ex["exception_type"].isin(types) & ex["severity"].isin(severities) & ex["status"].isin(statuses)].sort_values(["priority_rank", "monetary_impact"], ascending=[True, False])
        cols = ["customer_name", "invoice_id", "payment_id", "exception_type", "priority", "status", "monetary_impact", "confidence_score", "reason", "recommended_action"]
        table = view[cols].rename(columns={"customer_name":"Customer", "invoice_id":"Invoice", "payment_id":"Payment", "exception_type":"Type", "priority":"Priority", "status":"Status", "monetary_impact":"Impact", "confidence_score":"Confidence", "reason":"Reason", "recommended_action":"Recommended action"})
        st.dataframe(table, use_container_width=True, hide_index=True, height=520)
        st.caption(f"{len(view):,} exceptions after filters • prioritized by financial impact")
        st.markdown("### Review an exception")
        if not view.empty:
            labels = [f"{r.invoice_id} • {r.payment_id if pd.notna(r.payment_id) else 'No payment'} • {r.exception_type}" for r in view.itertuples()]
            selected = st.selectbox("Select exception", labels, key="exception_selector")
            row = view.iloc[labels.index(selected)]
            st.markdown(f"**{row.get('customer_name', 'Unknown customer')}** • {row.get('exception_type')} • {row.get('priority')} priority")
            st.write(row.get("reason", "No reason available."))
            payment_id = None if pd.isna(row.get("payment_id")) else str(row.get("payment_id"))
            operational = operational_status_map().get((str(row.get("invoice_id")), "" if payment_id is None else payment_id), str(row.get("status", "REVIEW")))
            if operational == "REVIEW":
                decision_reason = st.text_area("Human decision reason", key=f"exception_reason_{row['invoice_id']}_{payment_id}", placeholder="Explain what you verified before deciding.")
                a, b = st.columns(2)
                with a:
                    if st.button("APPROVE MATCH", type="primary", use_container_width=True, key="exception_approve"):
                        try:
                            record = approve_match(str(row["invoice_id"]), payment_id, decision_reason, BASE_DIR)
                            st.success(f"Approved. Audit ID: {record['audit_id']}")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                with b:
                    if st.button("REJECT MATCH", use_container_width=True, key="exception_reject"):
                        try:
                            record = reject_match(str(row["invoice_id"]), payment_id, decision_reason, BASE_DIR)
                            st.success(f"Rejected. Audit ID: {record['audit_id']}")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
            else:
                st.info(f"This exception already has a human decision: **{operational}**.")

# ---------------- Cash ----------------
elif page == "CASH POSITION":
    st.markdown('<div class="eyebrow">Synthetic cash visibility</div>', unsafe_allow_html=True)
    st.title("Cash Position")
    st.markdown('<div class="demo-note"><b>DEMO / SYNTHETIC MODEL:</b> These figures are calculated only from the project dataset. They are not a bank balance, accounting ledger, or production cash forecast.</div>', unsafe_allow_html=True)
    try:
        cash = get_cash_position(BASE_DIR)
        render_kpis([
            ("CONFIRMED", cash_money(cash["confirmed_cash"]), "Incoming payments"),
            ("PENDING", cash_money(cash["pending_review_payments"]), "Awaiting human decision"),
            ("UNRESOLVED", cash_money(cash["unmatched_incoming_payments"]), "Unmatched incoming"),
            ("EXPECTED", cash_money(cash["expected_incoming_cash"]), "Current invoice dataset"),
        ])
        chart = pd.DataFrame({"Category":["Confirmed","Pending","Unresolved","Expected"], "Amount":[cash["confirmed_cash"].get("amount") or 0, cash["pending_review_payments"].get("amount") or 0, cash["unmatched_incoming_payments"].get("amount") or 0, cash["expected_incoming_cash"].get("amount") or 0]})
        st.plotly_chart(px.bar(chart, x="Category", y="Amount", text="Amount", template="plotly_white", title="Synthetic cash / receivables view"), use_container_width=True)
        st.markdown("### Largest pending receivables")
        pending = cash.get("largest_pending_receivables", [])
        if pending:
            st.dataframe(pd.DataFrame(pending), use_container_width=True, hide_index=True)
        else:
            st.success("No pending REVIEW payments.")
    except Exception as exc:
        st.error(f"Could not calculate cash position: {exc}")

# ---------------- Ask Controller ----------------
elif page == "ASK CONTROLLER":
    st.markdown('<div class="eyebrow">Read-only AI finance assistant</div>', unsafe_allow_html=True)
    st.title("Ask Controller")
    st.caption("The deterministic backend remains the source of truth. The assistant retrieves facts through read-only finance tools and cannot approve or modify records.")
    mode = os.getenv("FINANCE_ASSISTANT_MODE", "demo").casefold()
    if mode == "demo":
        st.info("Learning mode: no OpenAI API calls are made.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not configured.")
    examples = [
        "What is our current reconciliation status?",
        "How much money is unresolved?",
        "Why was INV-2024-00047 not reconciled?",
        "Show me the five largest exceptions.",
        "What is our current cash position?",
    ]
    ex_cols = st.columns(len(examples))
    chosen = None
    for i, (col, example) in enumerate(zip(ex_cols, examples)):
        with col:
            if st.button(example, key=f"ask_example_{i}", use_container_width=True):
                chosen = example
    if "finance_messages" not in st.session_state:
        st.session_state.finance_messages = []
    if "finance_previous_response_id" not in st.session_state:
        st.session_state.finance_previous_response_id = None
    if st.button("Clear conversation"):
        st.session_state.finance_messages = []
        st.session_state.finance_previous_response_id = None
        st.rerun()
    for message in st.session_state.finance_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = chosen or st.chat_input("Ask about reconciliation, invoices, payments, exceptions or cash...")
    if prompt:
        st.session_state.finance_messages.append({"role":"user", "content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Checking finance records..."):
                try:
                    answer, response_id = ask_finance_controller(prompt, previous_response_id=st.session_state.finance_previous_response_id, base_dir=BASE_DIR)
                    st.markdown(answer)
                    st.session_state.finance_messages.append({"role":"assistant", "content":answer})
                    st.session_state.finance_previous_response_id = response_id
                except Exception as exc:
                    st.error(f"Finance Controller unavailable: {exc}")

# ---------------- Audit ----------------
elif page == "AUDIT TRAIL":
    st.markdown('<div class="eyebrow">Human control evidence</div>', unsafe_allow_html=True)
    st.title("Audit Trail")
    st.caption("Append-only human decisions. Original reconciliation results remain preserved.")
    audit = get_audit_log(BASE_DIR)
    if audit.empty:
        st.info("No human decisions recorded yet.")
    else:
        view = audit[["timestamp", "invoice_id", "payment_id", "decision", "previous_status", "new_status", "reason"]].copy()
        view.columns = ["Timestamp", "Invoice", "Payment", "Decision", "Previous status", "New status", "Reason"]
        st.dataframe(view.sort_values("Timestamp", ascending=False), use_container_width=True, hide_index=True, height=600)

# ---------------- Benchmark ----------------
elif page == "BENCHMARK":
    st.markdown('<div class="eyebrow">Independent evaluation</div>', unsafe_allow_html=True)
    st.title("FINAL HELD-OUT TEST")
    st.caption("Fresh synthetic dataset evaluated with the existing reconciliation engine. No dashboard or AI feature is used to improve these metrics.")
    if not held_out:
        st.warning("Held-out benchmark results are not available. Run scripts/run_held_out_benchmark.py first.")
    else:
        render_kpis([
            ("Records", f"{held_out.get('records_processed', 0):,}", "Held-out invoices"),
            ("Match rate", f"{held_out.get('automatic_match_rate', 0):.1%}", None),
            ("Precision", f"{held_out.get('precision', 0):.1%}", None),
            ("Recall", f"{held_out.get('recall', 0):.1%}", None),
            ("F1", f"{held_out.get('f1_score', 0):.1%}", None),
        ])
        render_kpis([
            ("False positives", f"{held_out.get('false_positives', 0):,}", None),
            ("False negatives", f"{held_out.get('false_negatives', 0):,}", None),
            ("Exception rate", f"{held_out.get('exception_rate', 0):.1%}", None),
            ("Exception value", money(held_out.get('total_value_exceptions', 0)), None),
            ("Processing time", f"{held_out.get('processing_time_seconds', 0):.2f}s", None),
        ])
        st.markdown("### Benchmark metadata")
        st.dataframe(pd.DataFrame([{
            "Benchmark": held_out.get("benchmark", "FINAL HELD-OUT TEST"),
            "Seed": held_out.get("seed"),
            "Invoices": held_out.get("invoice_records"),
            "Payments": held_out.get("payment_records"),
            "Settlements": held_out.get("settlement_records"),
        }]), use_container_width=True, hide_index=True)

st.caption("Finance Controller • deterministic financial controls • read-only AI • human decisions audited")
