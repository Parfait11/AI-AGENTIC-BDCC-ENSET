import pytest

from src.tools import compute_loan_payment, compute_savings_projection


def test_compute_savings_projection_with_interest():
    result = compute_savings_projection.invoke({
        "initial_amount": 1000,
        "monthly_contribution": 100,
        "annual_rate_percent": 6,
        "years": 1,
    })
    assert result["future_value"] == pytest.approx(2295.23, abs=0.01)
    assert result["total_contributed"] == pytest.approx(2200.00, abs=0.01)
    assert result["interest_earned"] == pytest.approx(95.23, abs=0.01)


def test_compute_savings_projection_zero_rate():
    result = compute_savings_projection.invoke({
        "initial_amount": 1000,
        "monthly_contribution": 100,
        "annual_rate_percent": 0,
        "years": 1,
    })
    assert result["future_value"] == pytest.approx(2200.00, abs=0.01)
    assert result["interest_earned"] == pytest.approx(0.0, abs=0.01)


def test_compute_loan_payment_with_interest():
    result = compute_loan_payment.invoke({
        "principal": 10000,
        "annual_rate_percent": 6,
        "years": 1,
    })
    assert result["monthly_payment"] == pytest.approx(860.66, abs=0.01)
    assert result["total_paid"] == pytest.approx(10327.97, abs=0.01)
    assert result["total_interest"] == pytest.approx(327.97, abs=0.01)


def test_compute_loan_payment_zero_rate():
    result = compute_loan_payment.invoke({
        "principal": 12000,
        "annual_rate_percent": 0,
        "years": 1,
    })
    assert result["monthly_payment"] == pytest.approx(1000.00, abs=0.01)
    assert result["total_interest"] == pytest.approx(0.0, abs=0.01)
