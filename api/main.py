"""FastAPI backend for OmicsInsight — exposes the analysis pipeline as REST endpoints."""

import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from omicsinsight.config import PipelineConfig
from omicsinsight.io import load_json
from omicsinsight.pipeline import run_pipeline
from omicsinsight.utils import setup_logging

app = FastAPI(
    title="OmicsInsight API",
    description="Transcriptomics analysis pipeline — REST interface",
    version="1.0.0",
)

logger = setup_logging()

# Serve outputs/ as static files so browser can load PNGs directly
_outputs_dir = Path("outputs")
_outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(_outputs_dir)), name="outputs")

# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>OmicsInsight</title>
<style>
  body{font-family:system-ui,sans-serif;background:#f5f7fa;margin:0;padding:24px;}
  h1{color:#1a3a5c;margin-bottom:4px;}
  .subtitle{color:#666;margin-bottom:24px;font-size:.95rem;}
  .card{background:#fff;border-radius:10px;padding:24px;max-width:700px;box-shadow:0 2px 8px rgba(0,0,0,.08);}
  label{display:block;margin-top:14px;font-weight:600;font-size:.9rem;color:#333;}
  input,select{width:100%;box-sizing:border-box;padding:8px 10px;border:1px solid #ccc;border-radius:6px;font-size:.9rem;margin-top:4px;}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
  button{margin-top:20px;width:100%;padding:11px;background:#1a3a5c;color:#fff;border:none;border-radius:6px;font-size:1rem;cursor:pointer;}
  button:hover{background:#25507a;}
  button:disabled{background:#999;cursor:not-allowed;}
  #status{margin-top:16px;padding:12px;border-radius:6px;font-size:.9rem;display:none;}
  .ok{background:#e6f4ea;color:#1e6e35;border:1px solid #a8d5b5;}
  .err{background:#fdecea;color:#a32a2a;border:1px solid #f5c2c2;}
  .info{background:#e8f0fe;color:#1a3a5c;border:1px solid #b0c4ef;}
  #results{margin-top:24px;display:none;}
  #results h2{color:#1a3a5c;margin-bottom:12px;}
  .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:18px;}
  .metric{background:#f0f4ff;border-radius:8px;padding:12px;text-align:center;}
  .metric .val{font-size:1.5rem;font-weight:700;color:#1a3a5c;}
  .metric .lbl{font-size:.78rem;color:#555;margin-top:2px;}
  .plots{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
  .plots img{width:100%;border-radius:8px;border:1px solid #ddd;}
  .plots img.broken{display:none;}
  .section{margin-top:18px;}
  .section h3{color:#1a3a5c;font-size:.95rem;margin-bottom:8px;}
  table{width:100%;border-collapse:collapse;font-size:.83rem;}
  th{background:#1a3a5c;color:#fff;padding:6px 10px;text-align:left;}
  td{padding:5px 10px;border-bottom:1px solid #eee;}
  tr:hover td{background:#f5f7ff;}
  a.dl{display:inline-block;margin-top:8px;margin-right:8px;padding:5px 12px;background:#e8f0fe;border-radius:5px;text-decoration:none;color:#1a3a5c;font-size:.83rem;}
  a.dl:hover{background:#c5d8ff;}
</style>
</head>
<body>
<h1>OmicsInsight</h1>
<p class="subtitle">Transcriptomics analysis pipeline &mdash; plant expression data</p>

<div class="card">
  <label>Count matrix path
    <input id="counts_path" value="dataset/GSE124666/GSE124666_NGS_000247_countData.txt"/>
  </label>
  <label>Metadata path
    <input id="metadata_path" value="dataset/GSE124666/GSE124666_series_matrix.txt"/>
  </label>
  <label>Output directory
    <input id="output_dir" value="outputs/run_ui"/>
  </label>
  <label>Target column (for classification)
    <input id="target_column" value="treatment"/>
  </label>
  <div class="row">
    <div>
      <label>Max features
        <input id="max_features" type="number" value="500"/>
      </label>
    </div>
    <div>
      <label>Number of clusters
        <input id="n_clusters" type="number" value="3"/>
      </label>
    </div>
  </div>
  <div class="row">
    <div>
      <label>Log2 transform
        <select id="log_transform"><option value="true" selected>Yes</option><option value="false">No</option></select>
      </label>
    </div>
    <div>
      <label>UMAP
        <select id="umap_enabled"><option value="false" selected>Disabled</option><option value="true">Enabled</option></select>
      </label>
    </div>
  </div>
  <button id="runBtn" onclick="runAnalysis()">&#9654; Run Analysis</button>
  <div id="status"></div>
</div>

<div id="results" class="card">
  <h2>Results</h2>
  <div class="metrics" id="metricsRow"></div>
  <div class="plots" id="plotsRow"></div>
  <div class="section" id="classSection"></div>
  <div class="section" id="topFeatSection"></div>
  <div id="dlLinks"></div>
</div>

<script>
async function runAnalysis() {
  const btn = document.getElementById('runBtn');
  const st  = document.getElementById('status');
  const res = document.getElementById('results');
  res.style.display = 'none';
  btn.disabled = true;
  btn.textContent = '⏳ Running… (this may take 1-2 minutes)';
  st.className = 'info'; st.style.display = 'block';
  st.textContent = 'Pipeline started — parsing, preprocessing, PCA, clustering, classification…';

  const payload = {
    counts_path:    document.getElementById('counts_path').value,
    metadata_path:  document.getElementById('metadata_path').value,
    output_dir:     document.getElementById('output_dir').value,
    target_column:  document.getElementById('target_column').value,
    max_features:   parseInt(document.getElementById('max_features').value),
    n_clusters:     parseInt(document.getElementById('n_clusters').value),
    log_transform:  document.getElementById('log_transform').value === 'true',
    umap_enabled:   document.getElementById('umap_enabled').value === 'true',
  };

  try {
    const r = await fetch('/analyze', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) { throw new Error(data.detail || 'Pipeline error'); }

    st.className = 'ok';
    st.textContent = `✔ Completed — run ID: ${data.run_id}  |  ${data.n_samples} samples  |  ${data.n_features} features`;
    btn.textContent = '▶ Run Analysis';
    btn.disabled = false;

    // Fetch full summary
    const sr = await fetch(`/results/${data.run_id}`);
    const summary = await sr.json();
    renderResults(summary, data.output_dir);
  } catch(e) {
    st.className = 'err';
    st.textContent = '✖ ' + e.message;
    btn.textContent = '▶ Run Analysis';
    btn.disabled = false;
  }
}

function renderResults(s, outDir) {
  const res = document.getElementById('results');
  res.style.display = 'block';

  // Metrics row
  const km = s.clustering?.KMeans || {};
  const lr = s.classification?.LogisticRegression || {};
  const rf = s.classification?.RandomForest || {};
  const pca = s.pca?.explained_variance_ratio || [];
  document.getElementById('metricsRow').innerHTML = [
    ['Samples', s.dataset?.n_samples],
    ['Features', s.dataset?.n_features_after_preprocessing],
    ['PC1 var%', pca[0] ? (pca[0]*100).toFixed(1)+'%' : '—'],
    ['KMeans sil.', km.silhouette_score ?? '—'],
    ['LR accuracy', lr.accuracy ?? '—'],
    ['RF accuracy', rf.accuracy ?? '—'],
  ].map(([l,v])=>`<div class="metric"><div class="val">${v}</div><div class="lbl">${l}</div></div>`).join('');

  // Plots
  const slash = outDir.replace(/\\\\/g, '/');
  document.getElementById('plotsRow').innerHTML = ['pca_scatter','umap_scatter','cluster_heatmap'].map(name=>{
    const src = `/${slash}/${name}.png`;
    return `<img src="${src}" alt="${name}" onerror="this.classList.add('broken')"/>`;
  }).join('');

  // Classification table
  let classHtml = '';
  if (s.classification) {
    classHtml = '<h3>Classification (LOO-CV)</h3><table><tr><th>Model</th><th>Accuracy</th><th>Macro F1</th></tr>';
    for (const [name, m] of Object.entries(s.classification)) {
      classHtml += `<tr><td>${name}</td><td>${m.accuracy}</td><td>${m.macro_f1}</td></tr>`;
    }
    classHtml += '</table>';
  }
  document.getElementById('classSection').innerHTML = classHtml;

  // Top features
  let featHtml = '';
  const feats = s.top_features?.features || [];
  if (feats.length) {
    featHtml = '<h3>Top Ranked Features</h3><table><tr><th>#</th><th>Feature</th><th>Avg Rank</th><th>Variance Rank</th></tr>';
    feats.slice(0,10).forEach((f,i)=>{
      featHtml += `<tr><td>${i+1}</td><td>${f.feature}</td><td>${f.avg_rank?.toFixed(1)??'—'}</td><td>${f.variance_rank??'—'}</td></tr>`;
    });
    featHtml += '</table>';
  }
  document.getElementById('topFeatSection').innerHTML = featHtml;

  // Download links
  const dl = document.getElementById('dlLinks');
  dl.innerHTML = '<br>' + ['report.md','analysis_summary.json','ranked_features.csv','cluster_labels.csv','pca_components.csv'].map(f=>
    `<a class="dl" href="/${slash}/${f}" download>⬇ ${f}</a>`
  ).join('');
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui() -> HTMLResponse:
    """Browser-based analysis UI."""
    return HTMLResponse(content=_HTML)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    counts_path: str
    metadata_path: str
    output_dir: Optional[str] = None
    target_column: str = "treatment"
    sample_id_column: str = "sample_id"
    max_features: int = 500
    n_clusters: int = 3
    log_transform: bool = True
    min_total_count: int = 10
    random_state: int = 42
    umap_enabled: bool = True


class AnalyzeResponse(BaseModel):
    run_id: str
    status: str
    output_dir: str
    n_samples: int
    n_features: int


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Basic liveness check."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the full analysis pipeline on the given local files."""
    run_id = uuid.uuid4().hex[:12]
    output_dir = request.output_dir or f"outputs/{run_id}"

    # Validate file paths exist
    counts_path = Path(request.counts_path).resolve()
    metadata_path = Path(request.metadata_path).resolve()

    if not counts_path.is_file():
        raise HTTPException(status_code=400, detail=f"Counts file not found: {counts_path}")
    if not metadata_path.is_file():
        raise HTTPException(status_code=400, detail=f"Metadata file not found: {metadata_path}")

    config = PipelineConfig(
        counts_path=str(counts_path),
        metadata_path=str(metadata_path),
        output_dir=output_dir,
        target_column=request.target_column,
        sample_id_column=request.sample_id_column,
        max_features=request.max_features,
        n_clusters=request.n_clusters,
        log_transform=request.log_transform,
        min_total_count=request.min_total_count,
        random_state=request.random_state,
        umap_enabled=request.umap_enabled,
    )

    try:
        summary = run_pipeline(config)
    except Exception as exc:
        logger.error("Pipeline failed for run %s: %s", run_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    return AnalyzeResponse(
        run_id=run_id,
        status="completed",
        output_dir=output_dir,
        n_samples=summary["dataset"]["n_samples"],
        n_features=summary["dataset"]["n_features_after_preprocessing"],
    )


@app.get("/results/{run_id}")
def get_results(run_id: str) -> dict:
    """Retrieve the analysis summary for a completed run."""
    # Validate run_id to prevent path traversal
    if not re.match(r"^[a-zA-Z0-9_-]+$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format.")

    summary_path = Path("outputs") / run_id / "analysis_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for run_id: {run_id}")

    return load_json(str(summary_path))
