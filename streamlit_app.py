from io import StringIO
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch

from sentiment_analysis.predict import SentimentPredictor

SENTIMENT_ORDER = ["positive", "neutral", "negative"]
SENTIMENT_COLORS = {
    "positive": "#1B9E77",
    "neutral": "#E6AB02",
    "negative": "#D95F02",
}


def normalize_sentiment(value: str) -> str:
    token = str(value).strip().lower()
    if token in {"pos", "positive", "1"}:
        return "positive"
    if token in {"neu", "neutral", "0"}:
        return "neutral"
    if token in {"neg", "negative", "-1"}:
        return "negative"
    return "neutral"


@st.cache_resource(show_spinner=False)
def load_predictor(model_type: str) -> SentimentPredictor:
    return SentimentPredictor(model_type=model_type)


def build_distribution_charts(counts: dict) -> tuple:
    values = [counts.get(label, 0) for label in SENTIMENT_ORDER]
    colors = [SENTIMENT_COLORS[label] for label in SENTIMENT_ORDER]
    total = sum(values)
    labels = [label.title() for label in SENTIMENT_ORDER]
    pie_values = []
    pie_labels = []
    pie_colors = []
    for label, value, color in zip(labels, values, colors):
        if value > 0:
            pie_values.append(value)
            pie_labels.append(label)
            pie_colors.append(color)

    pie_fig, pie_ax = plt.subplots(figsize=(4.8, 4.8))
    if sum(pie_values) == 0:
        pie_ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        pie_ax.axis("off")
    else:
        def _pie_label(pct):
            count = int(round((pct / 100.0) * sum(pie_values)))
            return f"{pct:.1f}%\n(n={count})"

        pie_ax.pie(
            pie_values,
            labels=None,
            colors=pie_colors,
            autopct=_pie_label,
            startangle=90,
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        )
        legend_handles = [Patch(facecolor=color, edgecolor="white") for color in colors]
        legend_labels = [f"{lbl} (n={val})" for lbl, val in zip(labels, values)]
        pie_ax.legend(
            legend_handles,
            legend_labels,
            title=None,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=3,
            frameon=False,
        )
    pie_ax.set_title("Sentiment Distribution (%)")
    pie_ax.set_ylabel("")
    pie_fig.subplots_adjust(top=0.86, bottom=0.22)

    bar_fig, bar_ax = plt.subplots(figsize=(7.2, 4.2))
    bars = bar_ax.bar(
        labels,
        values,
        color=colors,
    )
    bar_ax.set_title("Sentiment Distribution (Counts)")
    bar_ax.set_xlabel("Sentiment Class")
    bar_ax.set_ylabel("Number of Queries")
    bar_ax.grid(axis="y", linestyle="--", alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total * 100.0) if total else 0.0
        bar_ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
        )

    return pie_fig, bar_fig


def percentage(part: int, whole: int) -> float:
    return (part / whole * 100.0) if whole else 0.0


def infer_text_column(df: pd.DataFrame) -> str:
    candidates = ["text", "review", "review_text", "content", "comment", "feedback"]
    lowered = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return df.columns[0]


def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen = {}
    unique = []
    for col in df.columns:
        count = seen.get(col, 0)
        if count == 0:
            unique.append(col)
        else:
            unique.append(f"{col}_{count}")
        seen[col] = count + 1
    out = df.copy()
    out.columns = unique
    return out


def collect_texts_from_ui() -> tuple[list[str], pd.DataFrame]:
    source = st.radio("Data source", ["Manual input", "CSV upload"], horizontal=True)
    if source == "Manual input":
        raw_text = st.text_area(
            "Enter one review per line",
            placeholder="Excellent battery life\nAverage camera quality\nVery disappointed with performance",
            height=180,
        )
        texts = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return texts, pd.DataFrame({"text": texts})

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is None:
        return [], pd.DataFrame()

    csv_bytes = uploaded.read()
    df = pd.read_csv(StringIO(csv_bytes.decode("utf-8", errors="ignore")))
    if df.empty:
        return [], df

    default_column = infer_text_column(df)
    column_index = list(df.columns).index(default_column)
    text_column = st.selectbox("Select text column", df.columns.tolist(), index=column_index)
    filtered = df[df[text_column].notna()].copy()
    texts = filtered[text_column].astype(str).str.strip()
    filtered = filtered[texts != ""].copy()
    texts = filtered[text_column].astype(str).tolist()
    return texts, filtered


def render_header() -> None:
    st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
    st.title("Sentiment Analysis Dashboard")


def render_results(results_df: pd.DataFrame, model_name: str) -> None:
    total = len(results_df)
    counts = {label: int((results_df["sentiment"] == label).sum()) for label in SENTIMENT_ORDER}
    avg_confidence = float(results_df["confidence"].mean()) if total else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Queries", f"{total}")
    k2.metric("Positive Share", f"{percentage(counts['positive'], total):.1f}%")
    k3.metric("Neutral Share", f"{percentage(counts['neutral'], total):.1f}%")
    k4.metric("Negative Share", f"{percentage(counts['negative'], total):.1f}%")

    c1, c2 = st.columns([1, 1])
    pie_fig, bar_fig = build_distribution_charts(counts)
    with c1:
        st.pyplot(pie_fig)
    with c2:
        st.pyplot(bar_fig)

    st.subheader("Prediction Results Table")
    display_df = make_unique_columns(results_df.copy())
    display_df["confidence"] = display_df["confidence"].round(4)
    st.dataframe(display_df, use_container_width=True)

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv_data,
        file_name="sentiment_predictions.csv",
        mime="text/csv",
    )

    with st.expander("Model Details"):
        st.write(f"Model: `{model_name}`")
        st.write(f"Average confidence: `{avg_confidence:.3f}`")
        low_conf = display_df[display_df["confidence"] < 0.60]
        st.write(f"Low-confidence predictions (< 0.60): `{len(low_conf)}`")

    st.subheader("Neutral Results")
    neutral_df = display_df[display_df["sentiment"] == "neutral"].copy()
    if neutral_df.empty:
        st.info("No neutral results found for this analysis batch.")
    else:
        st.dataframe(neutral_df, use_container_width=True)

    st.subheader("Query-Level Sentiment Breakdown")
    for idx, row in display_df.head(100).iterrows():
        s = row["sentiment"]
        badge = "Positive" if s == "positive" else ("Neutral" if s == "neutral" else "Negative")
        score_line = (
            f"pos={row.get('score_positive', 0.0):.3f} | "
            f"neu={row.get('score_neutral', 0.0):.3f} | "
            f"neg={row.get('score_negative', 0.0):.3f}"
        )
        source = row.get("source", "model")
        text_value = str(row.get("text", row.get("review", row.get("review_text", ""))))
        with st.expander(f"Query {idx + 1} | Sentiment: {badge} | Confidence: {row['confidence']:.3f}"):
            st.write(text_value)
            st.caption(f"Model probability signal: {score_line}")
            st.caption(f"Decision source: `{source}` | Lexical compound score: `{row.get('vader_compound', 0.0):.3f}`")


def append_query_log(results_df: pd.DataFrame) -> None:
    if "query_log" not in st.session_state:
        st.session_state["query_log"] = []

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text_col = None
    for candidate in ["text", "review", "review_text", "content", "comment", "feedback"]:
        if candidate in results_df.columns:
            text_col = candidate
            break

    if text_col is None:
        text_col = "processed_text" if "processed_text" in results_df.columns else None
    if text_col is None:
        return

    for _, row in results_df.iterrows():
        st.session_state["query_log"].append(
            {
                "timestamp": now,
                "query": str(row.get(text_col, "")),
                "sentiment": str(row.get("sentiment", "neutral")),
                "confidence": float(row.get("confidence", 0.0)),
                "model": str(row.get("model", "")),
                "source": str(row.get("source", "model")),
            }
        )


def render_query_log() -> None:
    st.subheader("Query Log")
    if "query_log" not in st.session_state:
        st.session_state["query_log"] = []

    if not st.session_state["query_log"]:
        st.caption("No queries logged yet. Run analysis to populate the log.")
        return

    log_df = pd.DataFrame(st.session_state["query_log"])
    log_df["confidence"] = log_df["confidence"].round(4)
    st.dataframe(log_df, use_container_width=True)

    log_csv = log_df.to_csv(index=False).encode("utf-8")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            label="Download Query Log CSV",
            data=log_csv,
            file_name="query_log.csv",
            mime="text/csv",
        )
    with c2:
        if st.button("Clear Query Log", use_container_width=True):
            st.session_state["query_log"] = []
            st.rerun()


def main() -> None:
    render_header()

    available_models = SentimentPredictor.available_model_types()
    st.sidebar.header("Classification Model")
    selected_model = st.sidebar.selectbox(
        "Choose model",
        options=available_models,
        index=0,
        help="`best` uses the training report winner or best available fallback.",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Submission Mode")
    st.sidebar.caption("Use `best` for final demo. Switch models to show comparative behavior.")

    try:
        predictor = load_predictor(selected_model)
    except Exception as exc:
        st.error(f"Failed to load model artifacts. Details: {exc}")
        st.stop()

    texts, source_df = collect_texts_from_ui()

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not texts:
            st.warning("No valid text found. Add manual input or upload a CSV with a populated text column.")
            st.stop()

        with st.spinner("Analyzing sentiment..."):
            raw_results = predictor.predict_batch(texts)

        results_df = pd.DataFrame(raw_results)
        results_df["sentiment"] = results_df["sentiment"].apply(normalize_sentiment)

        if not source_df.empty and len(source_df) == len(results_df):
            aligned_source = source_df.reset_index(drop=True)
            overlapping = [col for col in aligned_source.columns if col in results_df.columns]
            if overlapping:
                aligned_source = aligned_source.drop(columns=overlapping)
            results_df = pd.concat([aligned_source, results_df], axis=1)

        append_query_log(results_df)
        render_results(results_df, predictor.model_type)
        st.success("Analysis completed successfully. You can review query-level details and download the result file.")

    render_query_log()


if __name__ == "__main__":
    main()
