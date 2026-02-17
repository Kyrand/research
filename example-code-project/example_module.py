"""Example module demonstrating a code-heavy research project."""


def analyze_data(items):
    """Analyze a list of items and return summary statistics."""
    if not items:
        return {"count": 0, "total": 0, "average": 0}

    total = sum(items)
    count = len(items)
    return {
        "count": count,
        "total": total,
        "average": total / count,
        "min": min(items),
        "max": max(items),
    }


def format_report(stats):
    """Format analysis statistics into a readable report string."""
    if stats["count"] == 0:
        return "No data to report."

    lines = [
        f"Items analyzed: {stats['count']}",
        f"Total: {stats['total']}",
        f"Average: {stats['average']:.2f}",
        f"Range: {stats['min']} - {stats['max']}",
    ]
    return "\n".join(lines)
