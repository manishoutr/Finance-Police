import pandas as pd
import pytest

from human_review import approve_match, reject_match, get_audit_log, get_operational_status


def make_project(tmp_path):
    (tmp_path / "outputs").mkdir()
    pd.DataFrame([{
        "invoice_id": "INV-001",
        "payment_id": "PMT-001",
        "invoice_amount": 1000.0,
        "payment_amount": 950.0,
        "amount_difference": 50.0,
        "customer_similarity": 92.0,
        "date_difference": 2,
        "reference_match": True,
        "confidence_score": 82.5,
        "status": "REVIEW",
        "reason": "Amount differs slightly.",
    }]).to_csv(tmp_path / "outputs" / "reconciliation_results.csv", index=False)
    return tmp_path


def test_approve_match_logs_and_preserves_reconciliation(tmp_path):
    base = make_project(tmp_path)
    before = pd.read_csv(base / "outputs" / "reconciliation_results.csv")
    record = approve_match("INV-001", "PMT-001", "Verified remittance advice.", base)
    after = pd.read_csv(base / "outputs" / "reconciliation_results.csv")
    audit = get_audit_log(base)

    assert record["previous_status"] == "REVIEW"
    assert record["new_status"] == "APPROVED"
    assert record["decision"] == "APPROVE MATCH"
    assert get_operational_status("INV-001", "PMT-001", base) == "APPROVED"
    pd.testing.assert_frame_equal(before, after)
    assert len(audit) == 1
    assert audit.iloc[0]["reason"] == "Verified remittance advice."


def test_reject_match_logs_and_preserves_reconciliation(tmp_path):
    base = make_project(tmp_path)
    before = pd.read_csv(base / "outputs" / "reconciliation_results.csv")
    record = reject_match("INV-001", "PMT-001", "Bank statement does not support the payment.", base)
    after = pd.read_csv(base / "outputs" / "reconciliation_results.csv")

    assert record["new_status"] == "REJECTED"
    assert record["decision"] == "REJECT MATCH"
    assert get_operational_status("INV-001", "PMT-001", base) == "REJECTED"
    pd.testing.assert_frame_equal(before, after)


def test_only_review_can_be_decided(tmp_path):
    base = make_project(tmp_path)
    recon = pd.read_csv(base / "outputs" / "reconciliation_results.csv")
    recon.loc[0, "status"] = "MATCHED"
    recon.to_csv(base / "outputs" / "reconciliation_results.csv", index=False)
    with pytest.raises(ValueError, match="only allowed.*REVIEW"):
        approve_match("INV-001", "PMT-001", "Checked.", base)


def test_reason_is_required(tmp_path):
    base = make_project(tmp_path)
    with pytest.raises(ValueError, match="reason is required"):
        approve_match("INV-001", "PMT-001", "", base)


def test_cannot_decide_same_case_twice(tmp_path):
    base = make_project(tmp_path)
    approve_match("INV-001", "PMT-001", "Verified.", base)
    with pytest.raises(ValueError, match="already has a human decision"):
        reject_match("INV-001", "PMT-001", "Changed my mind.", base)
