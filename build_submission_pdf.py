import argparse
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER

ROOT = Path(__file__).resolve().parent
SHOTS = ROOT / "reports" / "screenshots"
OUT = ROOT / "reports" / "Final_Submission.pdf"

def build(github_url):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=19, leading=24, spaceAfter=12))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
    story = []
    story += [
        Paragraph("End-to-End Data Management Pipeline for a Recommendation System", styles["CoverTitle"]),
        Paragraph("Data Management for Machine Learning – Assignment 2", styles["Heading2"]),
        Paragraph(f"<b>GitHub repository:</b> <link href='{github_url}' color='blue'>{github_url}</link>", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("1. Objective and dataset", styles["Heading2"]),
        Paragraph("The project implements a reproducible recommendation data pipeline using the public MovieLens Latest Small dataset. It ingests user, movie and rating-interaction data; validates data quality; prepares chronological train/test data; creates reusable features; trains two recommenders; evaluates them; and orchestrates the workflow.", styles["BodyText"]),
        Paragraph("MovieLens provides movie metadata and timestamped ratings. Since the dataset does not provide demographic user profiles, the users entity is derived from distinct observed user identifiers and records its source. No demographic attributes are fabricated.", styles["BodyText"]),
        Paragraph("2. Architecture and implementation", styles["Heading2"]),
        Paragraph("The pipeline stages are: ingestion → validation → preparation → transformation → feature management → model training → model evaluation → orchestration. Python and Pandas handle ingestion and data processing; validation checks cover nulls, duplicates, types, rating range, timestamp validity and referential integrity. Invalid interactions are separated into an error CSV and validation results are saved as JSON and CSV.", styles["BodyText"]),
        Paragraph("A chronological per-user holdout is used: for users with at least two interactions, the latest interaction is assigned to test and earlier interactions are retained for training. User activity, item popularity and a user-item matrix are generated. Reusable user, item and interaction feature tables are documented in a feature catalog.", styles["BodyText"]),
        Paragraph("3. Models and evaluation", styles["Heading2"]),
        Paragraph("Two models are trained: (1) a popularity-based baseline ranked by training interaction count and average rating, and (2) personalized item-based collaborative filtering using cosine similarity between item interaction vectors. Ranking relevance is defined as a held-out rating of at least 4.0. Evaluation uses K=5 and reports Precision@5, Recall@5, NDCG@5, catalog coverage, and intra-list genre diversity. Explicit rating predictions are additionally evaluated using RMSE and MAE.", styles["BodyText"]),
        Paragraph("4. Experiment tracking and orchestration", styles["Heading2"]),
        Paragraph("MLflow records model names, K, relevance threshold, training/test sizes, evaluation metrics and the validation report artifact. A dependency-aware Python workflow executes stages in sequence, stops when a dependency fails, and writes a JSON execution summary. The README documents environment setup, execution, output locations and the MLflow UI.", styles["BodyText"]),
        Paragraph("5. Results", styles["Heading2"]),
    ]
    results_path = ROOT / "reports" / "model_comparison.csv"
    if results_path.exists():
        import pandas as pd
        df = pd.read_csv(results_path, index_col=0).reset_index(names="Model")
        data = [list(df.columns)] + df.round(4).astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#D9EAF7")),
            ("GRID",(0,0),(-1,-1),0.4,colors.grey),
            ("FONTSIZE",(0,0),(-1,-1),7),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F5F7FA")])
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("Run the pipeline to populate reports/model_comparison.csv. Rebuild this PDF after successful execution.", styles["BodyText"]))
    story += [
        Spacer(1,6),
        Paragraph("Interpretation: compare the baseline and personalized model across ranking quality, catalog reach, recommendation variety and rating prediction error. Do not claim one model is superior unless the recorded results support that conclusion.", styles["Small"]),
        Paragraph("6. Limitations", styles["Heading2"]),
        Paragraph("MovieLens Latest Small is a compact benchmark and does not represent all production environments. The latest-one-event holdout is a simple temporal evaluation design. Genre-based diversity is a proxy for variety. The dense item-similarity matrix may require a sparse or approximate-neighbor implementation for substantially larger catalogs.", styles["BodyText"]),
        Paragraph("7. Reproducibility", styles["Heading2"]),
        Paragraph("From the repository root, install dependencies with pip install -r requirements.txt and execute python run_pipeline.py. The workflow downloads the public dataset on first run, writes intermediate and final outputs to the documented folders, tracks experiments in MLflow, and records orchestration status in logs/pipeline_run.json.", styles["BodyText"]),
        PageBreak(),
        Paragraph("Execution Evidence – Screenshots", styles["Heading1"]),
        Paragraph("The following appendix contains screenshots captured by the student from the actual execution environment. Each image should show the corresponding successful stage or output.", styles["BodyText"]),
        Spacer(1,8)
    ]
    required = [
        ("01_ingestion.png","1. Data ingestion"),
        ("02_validation.png","2. Data validation"),
        ("03_preparation.png","3. Data preparation and temporal split"),
        ("04_transformation.png","4. Data transformation"),
        ("05_feature_management.png","5. Feature management"),
        ("06_model_training_evaluation.png","6. Model training and evaluation"),
        ("07_mlflow_runs.png","7. MLflow experiment tracking"),
        ("08_orchestration_success.png","8. Successful orchestration run"),
        ("09_github_repository.png","9. Public GitHub repository")
    ]
    missing = []
    for filename, caption in required:
        path = SHOTS / filename
        story.append(Paragraph(caption, styles["Heading3"]))
        if path.exists():
            img = Image(str(path))
            max_w, max_h = 170*mm, 210*mm
            scale = min(max_w/img.imageWidth, max_h/img.imageHeight, 1)
            img.drawWidth = img.imageWidth*scale
            img.drawHeight = img.imageHeight*scale
            story.append(img)
        else:
            missing.append(filename)
            story.append(Paragraph(f"<i>Screenshot not supplied: {filename}</i>", styles["Small"]))
        story.append(Spacer(1,8))
    if missing:
        story.append(Paragraph("Evidence still required: " + ", ".join(missing), styles["Small"]))
    doc.build(story)
    print(f"Created: {OUT}")
    if missing:
        print("WARNING: Missing screenshots:", ", ".join(missing))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-url", required=True, help="Public GitHub repository URL")
    args = parser.parse_args()
    build(args.github_url)
