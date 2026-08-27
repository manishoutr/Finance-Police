"""OpenAI-powered Finance Controller assistant.

The LLM is an explanation/orchestration layer only. All financial facts come
from read-only deterministic tools over the existing Finance Controller data.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from finance_tools import TOOL_FUNCTIONS

DEFAULT_MODEL = "gpt-5.6-luna"

SYSTEM_INSTRUCTIONS = """
You are the AI Finance Controller assistant for a synthetic finance operations system.

SOURCE OF TRUTH:
- The deterministic reconciliation outputs, evaluation outputs, exception outputs,
  and raw financial CSVs are the only source of financial facts.
- You have read-only backend tools that retrieve those facts.

MANDATORY BEHAVIOR:
1. For any question involving financial facts, IDs, amounts, counts, statuses,
   exceptions, reconciliation, payments, invoices, settlements, or cash, use the
   relevant backend tool(s) before answering.
2. Never invent or estimate a financial number, invoice, payment, exception,
   confidence score, or status.
3. Never independently decide that an invoice/payment is matched, unmatched,
   approved, rejected, or reconciled. Report the stored deterministic status only.
4. Never modify, approve, reject, delete, or create financial records.
5. Clearly separate FACTS from RECOMMENDATION when a recommendation is useful.
6. Mention relevant invoice/payment IDs when discussing individual records.
7. If the available data is insufficient, explicitly say that the data is
   insufficient instead of guessing.
8. Keep answers concise and useful to a finance operator.
9. When a tool returns no record, say that no matching record was found.
10. For cash position, use the methodology and labels returned by get_cash_position.
    Clearly label values as CONFIRMED, PENDING, UNRESOLVED, or EXPECTED. Do not
    describe this synthetic model as a bank balance, forecast, or production accounting result.
11. Never use ground_truth.csv as an operational source of truth. It is a benchmark
    artifact, not a finance record.
12. You are strictly READ-ONLY. You may only retrieve, search, summarize, and explain
    information through the supplied finance tools. You have no write, approval, rejection,
    deletion, creation, or modification capability.
13. Never claim to have completed a prohibited action. If asked to perform one, clearly
    refuse and state that human approval/rejection is a separate explicit UI action.
14. Backend financial data is authoritative. Use retrieved tool data for financial answers;
    do not rely on memory or user-supplied numbers when authoritative data is available.
15. Perform arithmetic through backend tools where possible and distinguish FACTS from
    RECOMMENDATIONS.

OUTPUT STYLE:
- Lead with the answer.
- Use short bullets for supporting facts.
- For individual cases, include IDs, amount, status, confidence, and the relevant
  matching signals when available.
- Recommendations must be labeled as recommendations and must not be phrased as
  completed actions.
""".strip()


# Deterministic preflight guard for prohibited write/approval requests.
# This runs before DEMO_MODE or OpenAI so the model can never be the gatekeeper
# for actions that the Finance Controller is not permitted to perform.
PROHIBITED_REQUESTS = (
    # Approval/rejection intent, including ordinary paraphrases such as
    # "please approve this invoice" and "go ahead and approve it".
    (r"\b(?:please\s+|go\s+ahead\s+and\s+|can\s+you\s+|could\s+you\s+|i\s+(?:want|need)\s+you\s+to\s+)?(?:approve|authorize|accept|reject|deny)\b", "approve or authorize reconciliation matches"),
    (r"\b(?:mark|treat|consider|set)\b.*\b(?:matched|reconciled|approved|rejected)\b", "change reconciliation results"),
    (r"\b(?:delete|remove|erase|drop)\b.*\b(?:unmatched|payment|invoice|transaction|record)\b", "delete financial records"),
    (r"\b(?:create|add|generate)\b.*\binvoice\b", "create invoices"),
    (r"\b(?:edit|change|update|modify|set|overwrite|alter)\b.*\binvoice\b", "edit invoice records or amounts"),
    (r"\b(?:edit|change|update|modify|set|overwrite|alter)\b.*\bpayment\b.*\b(?:amount|value)\b", "edit payment amounts"),
    (r"\b(?:change|edit|modify|update|overwrite|alter)\b.*\b(?:confidence|score)\b", "change confidence scores"),
    (r"\b(?:modify|change|edit|update|overwrite|alter)\b.*\bground[ _-]?truth\b", "modify ground truth"),
    (r"\b(?:modify|change|edit|update|overwrite|alter)\b.*\b(?:benchmark|evaluation|precision|recall|f1)\b", "modify evaluation or benchmark results"),
)

PROHIBITED_ACTION_MESSAGE = (
    "I’m read-only for financial data. I can retrieve and explain existing finance records, "
    "but I cannot {action}. Human approval or rejection must be performed explicitly in the "
    "Human Review UI, and financial records, reconciliation results, ground truth, and benchmark "
    "outputs cannot be modified by the AI."
)


def _blocked_financial_action(user_message: str) -> str | None:
    """Return a safe refusal when a user asks the AI to perform a prohibited action."""
    import re as _re
    text = str(user_message or "").casefold().strip()
    for pattern, action in PROHIBITED_REQUESTS:
        if _re.search(pattern, text):
            return PROHIBITED_ACTION_MESSAGE.format(action=action)
    return None


def _tool_schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TOOLS = [
    _tool_schema(
        "get_reconciliation_summary",
        "Get the stored reconciliation benchmark summary. Use for reconciliation status, match rate, precision, recall, F1, and exception value.",
        {},
        [],
    ),
    _tool_schema(
        "get_exceptions",
        "Get stored exceptions, optionally filtered by deterministic exception type and severity.",
        {
            "exception_type": {"type": ["string", "null"], "description": "Exception type filter, such as AMOUNT_MISMATCH, or null."},
            "severity": {"type": ["string", "null"], "description": "Severity filter: HIGH, MEDIUM, LOW, or null."},
        },
        ["exception_type", "severity"],
    ),
    _tool_schema(
        "get_invoice_details",
        "Get an invoice, its stored reconciliation result, matched payment, settlement, signals, and related exceptions.",
        {"invoice_id": {"type": "string", "description": "Invoice identifier."}},
        ["invoice_id"],
    ),
    _tool_schema(
        "get_payment_details",
        "Get a payment, associated invoice(s), reconciliation records, settlements, and exceptions.",
        {"payment_id": {"type": "string", "description": "Payment identifier."}},
        ["payment_id"],
    ),
    _tool_schema(
        "get_top_exceptions",
        "Get the highest-value unresolved exceptions, ranked by invoice value exposure.",
        {"limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Number of exceptions to return."}},
        ["limit"],
    ),
    _tool_schema(
        "get_exception_summary",
        "Get exception counts and monetary exposure grouped by exception type.",
        {},
        [],
    ),
    _tool_schema(
        "search_transactions",
        "Search raw invoices, raw payments, and reconciliation records by customer, invoice ID, payment ID, reference, or other transaction text.",
        {
            "query": {"type": "string", "description": "Search text such as a customer name, invoice ID, payment ID, or reference."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Maximum number of combined results."},
        },
        ["query", "limit"],
    ),
    _tool_schema(
        "get_cash_position",
        "Calculate the synthetic cash position from actual invoices, payments, reconciliation results, and human-review audit data. Return CONFIRMED, PENDING, UNRESOLVED, and EXPECTED values plus largest pending receivables. This is not a bank balance or forecast.",
        {},
        [],
    ),
]


def _dispatch(name: str, arguments: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    if name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown finance tool: {name}"}
    try:
        return TOOL_FUNCTIONS[name](base_dir=base_dir, **arguments)
    except Exception as exc:  # Tool failures are returned to the model as data.
        return {"error": f"Finance data tool failed: {type(exc).__name__}: {exc}"}


def _function_calls(response: Any) -> list[Any]:
    return [item for item in response.output if getattr(item, "type", None) == "function_call"]



def _money(value: Any, currency: str | None = None) -> str:
    """Format a monetary value without assuming INR.

    Finance data carries its own currency. When a caller has no currency
    available, USD is used because the bundled benchmark is denominated in USD.
    This is presentation only and never converts amounts.
    """
    try:
        amount = float(value)
    except (TypeError, ValueError):
        amount = 0.0
    code = (str(currency).strip().upper() if currency else "USD")
    return f"{code} {amount:,.2f}"


def _cash_money(section: dict[str, Any]) -> str:
    by_currency = section.get("by_currency") or {}
    if len(by_currency) == 1:
        currency, amount = next(iter(by_currency.items()))
        return f"{currency} {float(amount):,.2f}"
    if not by_currency:
        return "0.00"
    return "; ".join(f"{k} {float(v):,.2f}" for k, v in sorted(by_currency.items()))


def _demo_answer(user_message: str, *, base_dir: Path) -> str:
    """Deterministic, zero-cost learning mode using the same finance tools.

    This mode intentionally does not pretend to be an LLM. It maps common
    finance-operator questions to read-only backend tools and formats their
    real results into concise answers.
    """
    q = user_message.casefold().strip()

    def tool(name: str, **kwargs: Any) -> dict[str, Any]:
        return _dispatch(name, kwargs, base_dir)

    if any(term in q for term in ("reconciliation status", "match rate", "matched", "unresolved invoices", "unresolved invoice", "review cases", "how many invoices are unresolved")):
        data = tool("get_reconciliation_summary")
        if data.get("error"):
            return f"I couldn't retrieve the reconciliation summary: {data['error']}"
        if "how many invoices are unresolved" in q or "unresolved invoice" in q:
            unresolved = int(data.get("review_count", 0)) + int(data.get("unmatched_count", 0))
            return (
                f"**{unresolved} invoices** currently require attention: "
                f"{data.get('review_count', 0)} REVIEW and {data.get('unmatched_count', 0)} UNMATCHED."
            )
        return (
            "### Reconciliation status\n"
            f"- Records processed: **{data.get('total_records', 0)}**\n"
            f"- Automatically matched: **{data.get('matched_count', 0)}**\n"
            f"- Review: **{data.get('review_count', 0)}**\n"
            f"- Unmatched: **{data.get('unmatched_count', 0)}**\n"
            f"- Match rate: **{float(data.get('match_rate', 0)):.1%}**\n"
            f"- Precision: **{float(data.get('precision', 0)):.1%}**\n"
            f"- Recall: **{float(data.get('recall', 0)):.1%}**\n"
            f"- F1: **{float(data.get('f1_score', 0)):.1%}**\n"
            f"- Total exception value: **{_money(data.get('total_exception_value', 0))}**"
        )

    if "confirmed cash" in q or "confirmed incoming" in q:
        data = tool("get_cash_position")
        if data.get("error"):
            return f"I couldn't retrieve the cash position: {data['error']}"
        section = data.get("confirmed_cash", {})
        return f"**CONFIRMED** incoming cash is **{_cash_money(section)}** across {section.get('payment_count', 0)} confirmed incoming payments. This is based on the synthetic dataset and is not a bank balance."

    if "unresolved" in q and ("money" in q or "receivable" in q or "receivables" in q):
        data = tool("get_cash_position")
        if data.get("error"):
            return f"I couldn't retrieve the cash position: {data['error']}"
        receivables = data.get("unresolved_receivables", {})
        incoming = data.get("unmatched_incoming_payments", {})
        return (
            f"**UNRESOLVED** receivables: **{_cash_money(receivables)}**. "
            f"Separately, **UNRESOLVED** unmatched incoming payments total **{_cash_money(incoming)}** across {incoming.get('payment_count', 0)} payments. "
            "These figures come from the synthetic dataset."
        )

    if ("cash" in q and "exception" not in q) or "pending receivables" in q or "confirmed cash" in q or "unresolved" in q and "money" in q:
        data = tool("get_cash_position")
        if data.get("error"):
            return f"I couldn't retrieve the cash position: {data['error']}"
        if "largest pending" in q or "pending receivables" in q:
            rows = data.get("largest_pending_receivables", [])
            if not rows:
                return "No pending REVIEW receivables are currently available."
            lines = ["### Largest pending receivables"]
            for i, row in enumerate(rows[:5], 1):
                lines.append(f"{i}. **{row.get('invoice_id')}** / **{row.get('payment_id')}**: {row.get('currency', 'UNKNOWN')} {float(row.get('amount', 0)):,.2f} payment against {row.get('currency', 'UNKNOWN')} {float(row.get('invoice_amount', 0)):,.2f} invoice")
            return "\n".join(lines)
        confirmed = data.get("confirmed_cash", {})
        pending = data.get("pending_review_payments", {})
        unresolved = data.get("unmatched_incoming_payments", {})
        expected = data.get("expected_incoming_cash", {})
        return (
            "### Synthetic cash position\n"
            f"- **CONFIRMED** incoming payments: **{_cash_money(confirmed)}** ({confirmed.get('payment_count', 0)} payments)\n"
            f"- **PENDING** review payments: **{_cash_money(pending)}** ({pending.get('payment_count', 0)} payments)\n"
            f"- **UNRESOLVED** unmatched incoming payments: **{_cash_money(unresolved)}** ({unresolved.get('payment_count', 0)} payments)\n"
            f"- **CONFIRMED** receivables: **{_cash_money(data.get('confirmed_receivables', {}))}**\n"
            f"- **UNRESOLVED** receivables: **{_cash_money(data.get('unresolved_receivables', {}))}**\n"
            f"- **EXPECTED** incoming cash: **{_cash_money(expected)}**\n\n"
            "This is calculated from the synthetic dataset only. It is not a bank balance or production cash forecast."
        )

    if "five largest" in q or "largest exceptions" in q or "top exceptions" in q:
        data = tool("get_top_exceptions", limit=5)
        rows = data.get("exceptions", [])
        if not rows:
            return "No unresolved exceptions were found in the available data."
        lines = ["### Five largest exceptions"]
        for i, row in enumerate(rows[:5], 1):
            lines.append(
                f"{i}. **{row.get('exception_id', 'N/A')}** | "
                f"{row.get('exception_type', 'OTHER')} | "
                f"Invoice **{row.get('invoice_id', 'N/A')}** | "
                f"Exposure **{_money(row.get('invoice_amount', 0))}** | "
                f"Severity **{row.get('severity', 'N/A')}**"
            )
        return "\n".join(lines)

    if "exception" in q and ("how much" in q or "money" in q or "value" in q):
        # Verify the exception artifact exists before answering. Evaluation
        # output alone is not enough to answer an operational exception query.
        data = tool("get_exceptions")
        if data.get("error"):
            return f"I couldn't retrieve the exception records: {data['error']}"
        rows = data.get("exceptions", [])
        if not rows:
            return "No stored exception records are currently available."
        total = 0.0
        for row in rows:
            # Prefer invoice exposure, then fall back to amount difference.
            value = row.get("invoice_amount")
            if value is None:
                value = row.get("amount_difference")
            try:
                total += abs(float(value or 0))
            except (TypeError, ValueError):
                continue
        return (
            f"The stored exception records contain **{_money(total)}** in exception exposure "
            f"across **{len(rows)}** exception records."
        )

    if "most common" in q or "exception summary" in q or ("common" in q and "problem" in q):
        data = tool("get_exception_summary")
        if data.get("error"):
            return f"I couldn't retrieve the exception summary: {data['error']}"
        rows = data.get("summary") or data.get("by_type") or []
        if not rows:
            return "No exception summary is available."
        lines = ["### Exception summary"]
        for row in rows:
            value = row.get("value", row.get("exception_value", 0))
            lines.append(f"- **{row.get('exception_type', 'OTHER')}**: {row.get('count', 0)} cases, {_money(value)} exposure")
        return "\n".join(lines)

    invoice_match = re.search(r"\bINV(?:[-_][A-Z0-9-]+|\d[A-Z0-9-]*)\b", user_message, re.IGNORECASE)
    if invoice_match:
        invoice_id = invoice_match.group(0).upper()
        data = tool("get_invoice_details", invoice_id=invoice_id)
        if not data.get("found"):
            return f"No invoice record was found for **{invoice_id}** in the available finance data."
        lines = [
            f"### Invoice {invoice_id}",
            f"- Customer: **{data.get('invoice', {}).get('customer_name', 'N/A')}**",
            f"- Invoice amount: **{_money(data.get('invoice', {}).get('amount'))}**",
            f"- Payment: **{data.get('matched_payment', {}).get('payment_id', 'None')}**",
            f"- Payment amount: **{_money(data.get('matched_payment', {}).get('amount'))}**",
            f"- Status: **{data.get('reconciliation', {}).get('status', 'N/A')}**",
            f"- Confidence: **{data.get('reconciliation', {}).get('confidence_score', 'N/A')}**",
        ]
        exceptions = data.get("exceptions", [])
        if exceptions:
            lines.append(f"- Exceptions: **{len(exceptions)}**")
            for exc in exceptions[:3]:
                lines.append(f"  - {exc.get('exception_type', 'OTHER')}: {exc.get('reason', 'No reason recorded.')}")
        return "\n".join(lines)

    payment_match = re.search(r"\b(?:PMT|PAY)(?:[-_][A-Z0-9-]+|\d[A-Z0-9-]*)\b", user_message, re.IGNORECASE)
    if payment_match:
        payment_id = payment_match.group(0).upper()
        data = tool("get_payment_details", payment_id=payment_id)
        if not data.get("found"):
            return f"No payment record was found for **{payment_id}** in the available finance data."
        return (
            f"### Payment {payment_id}\n"
            f"- Amount: **{_money(data.get('payment', {}).get('amount'))}**\n"
            f"- Date: **{data.get('payment', {}).get('date', 'N/A')}**\n"
            f"- Associated invoices: **{', '.join(data.get('invoice_ids', [])) or 'None'}**\n"
            f"- Status: **{data.get('status', 'N/A')}**\n"
            f"- Confidence: **{data.get('confidence_score', 'N/A')}**"
        )

    if "discrepanc" in q and ("payment" in q or "amount difference" in q or "variance" in q):
        amount_match = re.search(r"(?:>|above|over|exceed(?:ing)?|greater than)\s*[₹$]?\s*([\d,]+(?:\.\d+)?)", user_message, re.IGNORECASE)
        threshold = float(amount_match.group(1).replace(",", "")) if amount_match else 1000.0
        data = tool("get_exceptions")
        if data.get("error"):
            return f"I couldn't retrieve payment discrepancies: {data['error']}"
        rows = []
        for r in data.get("exceptions", []):
            try:
                difference = abs(float(r.get("amount_difference", 0) or 0))
            except (TypeError, ValueError):
                continue
            if difference > threshold:
                rows.append(r)
        if not rows:
            return f"No stored exceptions with an absolute amount difference above {_money(threshold)} were found."
        lines = [f"### Payment discrepancies above {_money(threshold)}"]
        for row in rows:
            lines.append(f"- **{row.get('invoice_id', 'N/A')}** / **{row.get('payment_id', 'N/A')}**: {_money(row.get('amount_difference', 0))} | {row.get('exception_type', 'OTHER')}")
        return "\n".join(lines)

    if "search" in q or "find" in q:
        query = re.sub(r"^(search|find)\s+", "", user_message, flags=re.IGNORECASE).strip()
        data = tool("search_transactions", query=query, limit=10)
        rows = data.get("results", [])
        if not rows:
            return f"No transactions matching **{query}** were found."
        lines = [f"### Search results for {query}"]
        for row in rows:
            lines.append(f"- {row.get('record_type', 'record')}: **{row.get('invoice_id', row.get('payment_id', 'N/A'))}** | {row.get('customer_name', '')} | {_money(row.get('amount', 0))}")
        return "\n".join(lines)

    return (
        "I'm running in **DEMO_MODE**, so no paid API call is being made. "
        "I can answer common questions using the same read-only finance tools. "
        "Try asking about reconciliation status, unresolved invoices, an invoice/payment ID, largest exceptions, exception summary, or cash position."
    )

def ask_finance_controller(
    user_message: str,
    *,
    previous_response_id: str | None = None,
    base_dir: Path | str | None = None,
) -> tuple[str, str | None]:
    """Answer one user turn using read-only finance tools.

    FINANCE_ASSISTANT_MODE=demo (default) uses a deterministic local learning
    assistant and never calls the OpenAI API. Set it to openai to enable the
    live LLM orchestration layer when API credits are available.
    """
    base_path = Path(base_dir or Path(__file__).resolve().parents[1])
    blocked = _blocked_financial_action(user_message)
    if blocked:
        return blocked, None
    mode = os.getenv("FINANCE_ASSISTANT_MODE", "demo").casefold()
    if mode == "demo":
        return _demo_answer(user_message, base_dir=base_path), None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Configure the environment variable before using the Finance Controller assistant.")

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The OpenAI Python package is unavailable. Install requirements.txt or use FINANCE_ASSISTANT_MODE=demo.") from exc

    client = OpenAI(api_key=api_key)

    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "tools": TOOLS,
        "input": user_message,
    }
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    response = client.responses.create(**kwargs)

    # Resolve tool calls until the model produces a final text response.
    for _ in range(8):
        calls = _function_calls(response)
        if not calls:
            text = (response.output_text or "").strip()
            if not text:
                text = "I could not produce a response from the available finance data."
            return text, response.id

        tool_outputs = []
        for call in calls:
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = _dispatch(call.name, arguments, base_path)
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

        response = client.responses.create(
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
            tools=TOOLS,
            previous_response_id=response.id,
            input=tool_outputs,
        )

    raise RuntimeError("The assistant exceeded the maximum number of finance-tool steps for this question.")
