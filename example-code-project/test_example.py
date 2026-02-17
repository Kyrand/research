"""Tests for example_module."""

from example_module import analyze_data, format_report


def test_analyze_data_basic():
    result = analyze_data([1, 2, 3, 4, 5])
    assert result["count"] == 5
    assert result["total"] == 15
    assert result["average"] == 3.0
    assert result["min"] == 1
    assert result["max"] == 5


def test_analyze_data_empty():
    result = analyze_data([])
    assert result["count"] == 0
    assert result["total"] == 0
    assert result["average"] == 0


def test_format_report():
    stats = analyze_data([10, 20, 30])
    report = format_report(stats)
    assert "Items analyzed: 3" in report
    assert "Total: 60" in report
    assert "Average: 20.00" in report


def test_format_report_empty():
    stats = analyze_data([])
    report = format_report(stats)
    assert report == "No data to report."
