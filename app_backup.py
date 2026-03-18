"""
app.py — Auto Data Pipeline Tool
=================================
HOW THE PIPELINE CONNECTS — STEP BY STEP:

  USER uploads CSV
       ↓
  STEP 1: load_data()         — reads CSV into a DataFrame
       ↓
  STEP 2: handle_missing_values() — fills NaN with mean/mode
       ↓
  STEP 3: add_noise()         — adds small random noise
       ↓
  STEP 4: validate_data()     — removes duplicates
       ↓
  STEP 5: normalize_data()    — scales numbers to [0, 1]
       ↓
  STEP 6: remove_outliers()   — drops outlier rows (IQR)
       ↓
  STEP 7: select_features()   — drops low-variance columns
       ↓
  STEP 8: split_data()        — 80/20 train/test split
       ↓
  RESULT shown in browser (first 5 rows + summary log)
"""

import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

app.secret_key = "pipeline_secret_key"

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ─────────────────────────────────────────────────────────────
# IMPORT EACH PIPELINE FUNCTION INDIVIDUALLY
# This is where app.py connects to your pipeline/ folder.
# Each function is imported from its own file.
# ─────────────────────────────────────────────────────────────

from pipeline.missing_values  import handle_missing_values  # Step 2
from pipeline.noise            import add_noise              # Step 3
from pipeline.validation       import validate_data          # Step 4
from pipeline.normalization    import normalize_data         # Step 5
from pipeline.outliers         import remove_outliers        # Step 6
from pipeline.feature_selection import select_features       # Step 7
from pipeline.split            import split_data             # Step 8


# ─────────────────────────────────────────────────────────────
# THE PIPELINE — connects all 7 functions in order
#
# Each function:
#   - receives the DataFrame from the previous step
#   - returns (updated_dataframe, message_string)
#   - the message goes into the summary log shown in the UI
# ─────────────────────────────────────────────────────────────

def run_pipeline(df):
    """
    Runs the full 7-step pipeline on a DataFrame.

    Returns:
        df      — the fully processed DataFrame
        summary — list of dicts, one per step, shown in the UI
    """

    summary = []   # collects one log entry per step

    # ── Step 1: Handle Missing Values ────────────────────────
    # Fills NaN: mean for numbers, mode for text columns
    df, msg = handle_missing_values(df)
    summary.append({"step": 1, "name": "handle_missing_values", "message": msg})
    print(f"[pipeline] Step 1 done → {msg}")

    # ── Step 2: Add Noise ─────────────────────────────────────
    # Adds tiny Gaussian noise to numeric columns (1% of std)
    # Randomly swaps 5% of categorical values
    df, msg = add_noise(df)
    summary.append({"step": 2, "name": "add_noise", "message": msg})
    print(f"[pipeline] Step 2 done → {msg}")

    # ── Step 3: Validate Data ─────────────────────────────────
    # Removes exact duplicate rows
    df, msg = validate_data(df)
    summary.append({"step": 3, "name": "validate_data", "message": msg})
    print(f"[pipeline] Step 3 done → {msg}")

    # ── Step 4: Normalize Data ────────────────────────────────
    # Scales all numeric columns to range [0, 1]
    # Categorical columns are NOT touched
    df, msg = normalize_data(df)
    summary.append({"step": 4, "name": "normalize_data", "message": msg})
    print(f"[pipeline] Step 4 done → {msg}")

    # ── Step 5: Remove Outliers ───────────────────────────────
    # Drops rows where any numeric value is outside
    # Q1 - 1.5*IQR  or  Q3 + 1.5*IQR  (IQR method)
    df, msg = remove_outliers(df)
    summary.append({"step": 5, "name": "remove_outliers", "message": msg})
    print(f"[pipeline] Step 5 done → {msg}")

    # ── Step 6: Select Features ───────────────────────────────
    # Drops numeric columns with near-zero variance
    # (columns that are almost the same value in every row)
    df, msg = select_features(df)
    summary.append({"step": 6, "name": "select_features", "message": msg})
    print(f"[pipeline] Step 6 done → {msg}")

    # ── Step 7: Split Data ────────────────────────────────────
    # Splits into 80% train / 20% test
    # Returns only the training set for preview
    df, msg = split_data(df)
    summary.append({"step": 7, "name": "split_data", "message": msg})
    print(f"[pipeline] Step 7 done → {msg}")

    print(f"[pipeline] All steps complete. Final shape: {df.shape}")
    return df, summary


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def allowed_file(filename):
    """Returns True only for .csv files."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "csv"


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """
    GET /  →  show the upload form (empty state).
    No data is passed — the template shows the empty state card.
    """
    return render_template(
        "index.html",
        preview=None,
        summary=None,
        original_shape=None,
        processed_shape=None,
    )


@app.route("/upload", methods=["POST"])
def upload():
    """
    POST /upload  →  receive the CSV, run the pipeline, show results.

    Flow:
      1. Validate the uploaded file
      2. Save it to uploads/
      3. Read it into a pandas DataFrame       ← load_data equivalent
      4. Pass DataFrame through run_pipeline() ← all 7 steps
      5. Render index.html with results
    """

    # ── 1. Validate ───────────────────────────────────────────
    if "file" not in request.files:
        flash("No file was included in the request.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("Please select a CSV file before clicking Run Pipeline.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(f"'{file.filename}' is not a CSV file. Please upload a .csv file.")
        return redirect(url_for("index"))

    # ── 2. Save ───────────────────────────────────────────────
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)
    print(f"[upload] File saved → {save_path}")

    # ── 3. Load into DataFrame ────────────────────────────────
    try:
        df = pd.read_csv(save_path)
        print(f"[upload] CSV loaded → {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"[upload] Columns: {list(df.columns)}")
    except Exception as e:
        flash(f"Could not read the CSV file: {e}")
        return redirect(url_for("index"))

    # Store original shape BEFORE pipeline changes it
    original_shape = df.shape

    # ── 4. Run the pipeline ───────────────────────────────────
    # This is the single call that chains all 7 steps
    try:
        processed_df, summary = run_pipeline(df.copy())
    except Exception as e:
        flash(f"Pipeline error at step: {e}")
        return redirect(url_for("index"))

    processed_shape = processed_df.shape

    print(f"[upload] Original shape : {original_shape}")
    print(f"[upload] Processed shape: {processed_shape}")

    # ── 5. Build HTML table preview of first 5 rows ───────────
    # pandas .to_html() converts the DataFrame to an HTML <table>
    # classes="data-table" applies our CSS table styles
    preview_html = processed_df.head(5).to_html(
        classes="data-table",
        index=False,      # don't show row numbers
        border=0,         # no HTML border attribute (CSS handles it)
    )

    # ── 6. Render results ─────────────────────────────────────
    return render_template(
        "index.html",
        filename=file.filename,
        original_shape=original_shape,    # e.g. (20, 5)
        processed_shape=processed_shape,  # e.g. (13, 4)
        preview=preview_html,             # HTML string → {{ preview|safe }}
        summary=summary,                  # list of step dicts → {% for item %}
    )


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Auto Data Pipeline Tool")
    print("=" * 50)
    print(f"  Templates : {app.template_folder}")
    print(f"  Static    : {app.static_folder}")
    print(f"  Uploads   : {UPLOAD_FOLDER}")
    print(f"  URL       : http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)