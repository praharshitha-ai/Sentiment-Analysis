HTML_DASHBOARD = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sentiment Dashboard</title>
</head>
<body>
  <h1>Sentiment Dashboard Moved</h1>
  <p>Use <code>streamlit run streamlit_app.py</code> from the project root.</p>
</body>
</html>
"""


def get_dashboard_html() -> str:
    """Return a migration message for legacy callers."""
    return HTML_DASHBOARD


if __name__ == "__main__":
    print(HTML_DASHBOARD)
