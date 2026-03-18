import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR,"templates"), static_folder=os.path.join(BASE_DIR,"static"))
app.secret_key = "pipeline_secret_key"
UPLOAD_FOLDER = os.path.join(BASE_DIR,"uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR,"outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

from pipeline.missing_values    import handle_missing_values
from pipeline.noise             import add_noise
from pipeline.validation        import validate_data
from pipeline.normalization     import normalize_data
from pipeline.outliers          import remove_outliers
from pipeline.feature_selection import select_features
from pipeline.split             import split_data

def run_pipeline(df):
    summary = []
    step_rows = []
    for i, step in enumerate([handle_missing_values,add_noise,validate_data,normalize_data,remove_outliers,select_features,split_data],1):
        df, msg = step(df)
        summary.append({"step":i,"name":step.__name__,"message":msg})
        step_rows.append(len(df))
        print(f"[pipeline] Step {i} -> {msg}")
    return df, summary, step_rows

def build_chart_data(orig, proc, summary, step_rows):
    num_cols = proc.select_dtypes(include='number').columns.tolist()
    return {
        "original_rows":   int(orig.shape[0]),
        "processed_rows":  int(proc.shape[0]),
        "numeric_cols":    len(num_cols),
        "categorical_cols":len(proc.select_dtypes(exclude='number').columns),
        "step_labels":     [s["name"].replace("_"," ") for s in summary],
        "step_rows":       step_rows,
        "col_names":       num_cols,
        "col_means":       [round(float(proc[c].mean()),4) for c in num_cols],
    }

def allowed_file(f): return "." in f and f.rsplit(".",1)[1].lower()=="csv"

@app.route("/",methods=["GET"])
def index():
    return render_template("index.html",preview=None,summary=None,original_shape=None,processed_shape=None,download_ready=False,output_filename=None,chart_data=None)

@app.route("/upload",methods=["POST"])
def upload():
    if "file" not in request.files: flash("No file."); return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename=="": flash("Select a CSV."); return redirect(url_for("index"))
    if not allowed_file(file.filename): flash("Only .csv files."); return redirect(url_for("index"))
    save_path = os.path.join(app.config["UPLOAD_FOLDER"],file.filename)
    file.save(save_path)
    try:
        df = pd.read_csv(save_path)
        print(f"[upload] {df.shape[0]} rows x {df.shape[1]} cols")
    except Exception as e:
        flash(f"Could not read CSV: {e}"); return redirect(url_for("index"))
    orig = df.copy()
    orig_shape = df.shape
    try:
        proc, summary, step_rows = run_pipeline(df.copy())
    except Exception as e:
        flash(f"Pipeline error: {e}"); return redirect(url_for("index"))
    out_name = "processed_" + file.filename
    out_path = os.path.join(app.config["OUTPUT_FOLDER"],out_name)
    proc.to_csv(out_path,index=False)
    print(f"[upload] Saved -> {out_path}")
    chart_data = build_chart_data(orig,proc,summary,step_rows)
    preview = proc.head(5).to_html(classes="data-table",index=False,border=0)
    return render_template("index.html",filename=file.filename,output_filename=out_name,original_shape=orig_shape,processed_shape=proc.shape,preview=preview,summary=summary,download_ready=True,chart_data=chart_data)

@app.route("/download/<filename>")
def download(filename):
    p = os.path.join(app.config["OUTPUT_FOLDER"],filename)
    if not os.path.exists(p): flash("File not found."); return redirect(url_for("index"))
    return send_file(p,mimetype="text/csv",as_attachment=True,download_name=filename)

if __name__=="__main__":
    print("="*50)
    print("  Auto Data Pipeline Tool")
    print(f"  URL : http://127.0.0.1:5000")
    print("="*50)
    app.run(debug=True,port=5000)
