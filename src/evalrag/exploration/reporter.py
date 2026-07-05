"""Format experiment results for display."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evalrag.core.types import ExperimentResult


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


class Reporter:
    """Renders experiment results in various formats."""

    @staticmethod
    def to_table(results: list[ExperimentResult]) -> str:
        if not results:
            return "No results."
        all_metrics: list[str] = []
        for r in results:
            for m in r.mean_scores:
                if m not in all_metrics:
                    all_metrics.append(m)

        col_widths = [max(20, max((len(r.name) for r in results), default=4))]
        col_widths += [max(10, len(m)) for m in all_metrics]

        header = "Name".ljust(col_widths[0])
        for i, m in enumerate(all_metrics):
            header += " | " + m.ljust(col_widths[i + 1])
        lines = [header, "-" * len(header)]

        for r in results:
            row = r.name.ljust(col_widths[0])
            for i, m in enumerate(all_metrics):
                val = r.mean_scores.get(m, float("nan"))
                row += " | " + f"{val:.4f}".ljust(col_widths[i + 1])
            lines.append(row)
        return "\n".join(lines)

    @staticmethod
    def to_json(
        results: list[ExperimentResult],
        output_dir: str = ".",
        context: dict | None = None,
    ) -> str:
        data = _build_report_data(results, context)
        path = Path(output_dir) / f"{_timestamp()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, indent=2, default=str)
        path.write_text(content)
        return str(path)

    @staticmethod
    def to_html(
        results: list[ExperimentResult],
        output_dir: str = ".",
        context: dict | None = None,
    ) -> str:
        data = _build_report_data(results, context)
        html = _render_html(data)
        path = Path(output_dir) / f"{_timestamp()}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        return str(path)


def _build_report_data(results: list[ExperimentResult], context: dict | None = None) -> dict:
    all_metrics: list[str] = []
    for r in results:
        for m in r.mean_scores:
            if m not in all_metrics:
                all_metrics.append(m)

    experiments = []
    for r in results:
        per_query = []
        for qr in r.query_results:
            per_query.append({
                "question": qr.query,
                "answer": qr.answer[:300],
                "ground_truth": qr.ground_truth,
                "scores": {s.metric: s.value for s in qr.scores},
                "num_chunks": len(qr.retrieval.chunks),
            })
        experiments.append({
            "name": r.name,
            "timestamp": r.timestamp.isoformat(),
            "mean_scores": r.mean_scores,
            "ranking": [{"metric": rr.metric, "value": rr.value} for rr in r.ranking_results],
            "per_query": per_query,
        })

    data = {
        "generated": datetime.now().isoformat(),
        "metrics": all_metrics,
        "experiments": experiments,
    }
    if context:
        data["context"] = context
    return data


def _render_html(data: dict) -> str:
    json_data = json.dumps(data, default=str)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EvalRAG Report</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root {{
  --bg: #ffffff; --bg-card: #f8f9fa; --bg-hover: #e9ecef;
  --text: #212529; --text-muted: #6c757d; --border: #dee2e6;
  --accent: #4361ee; --accent-light: #e8ecff; --grid: #e9ecef;
  --tooltip-bg: #212529; --tooltip-text: #f8f9fa;
  --green: #10b981; --green-bg: #ecfdf5; --green-border: #a7f3d0;
  --red: #ef4444; --red-bg: #fef2f2; --amber: #f59e0b;
  --bar-colors: #4361ee, #f72585, #4cc9f0, #7209b7, #3a0ca3, #f77f00;
  --winner-glow: 0 0 20px rgba(16,185,129,0.15);
  --card-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
[data-theme="dark"] {{
  --bg: #0a0a14; --bg-card: #12122a; --bg-hover: #1a1a3e;
  --text: #e0e0e0; --text-muted: #8888aa; --border: #2a2a4a;
  --accent: #7b93f5; --accent-light: #1a1a3e; --grid: #1e1e3a;
  --tooltip-bg: #e0e0e0; --tooltip-text: #0a0a14;
  --green: #34d399; --green-bg: #064e3b; --green-border: #065f46;
  --red: #f87171; --red-bg: #450a0a; --amber: #fbbf24;
  --bar-colors: #7b93f5, #f72585, #4cc9f0, #9b59b6, #6c5ce7, #f77f00;
  --winner-glow: 0 0 20px rgba(52,211,153,0.15);
  --card-shadow: 0 1px 3px rgba(0,0,0,0.2);
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.6;
  transition: background 0.3s, color 0.3s;
}}
.container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
.subtitle {{ color: var(--text-muted); margin-bottom: 1.5rem; font-size: 0.9rem; }}
.theme-toggle {{
  position: fixed; top: 1rem; right: 1rem; z-index: 100;
  width: 44px; height: 44px; border-radius: 50%;
  border: 1px solid var(--border); background: var(--bg-card);
  color: var(--text); cursor: pointer; font-size: 1.2rem;
  display: flex; align-items: center; justify-content: center;
  transition: all 0.3s;
}}
.theme-toggle:hover {{ background: var(--bg-hover); }}

/* Executive Summary */
.exec-summary {{
  margin-bottom: 2rem; padding: 1.5rem;
  background: var(--green-bg); border: 1px solid var(--green-border);
  border-radius: 12px; box-shadow: var(--winner-glow);
}}
.exec-summary h2 {{ font-size: 1rem; color: var(--green); margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }}
.exec-insight {{ font-size: 1.1rem; margin-bottom: 1rem; line-height: 1.5; }}
.exec-insight strong {{ color: var(--green); }}

.scorecards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-top: 1rem; }}
.scorecard {{
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.2rem; text-align: center;
  transition: all 0.3s; box-shadow: var(--card-shadow);
}}
.scorecard.winner {{ border-color: var(--green); box-shadow: var(--winner-glow); }}
.scorecard .name {{ font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }}
.scorecard .f1-value {{ font-size: 2.2rem; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.scorecard .f1-label {{ font-size: 0.7rem; color: var(--text-muted); margin-top: 0.2rem; }}
.scorecard .sub-metrics {{ display: flex; justify-content: center; gap: 1.2rem; margin-top: 0.6rem; font-size: 0.75rem; color: var(--text-muted); }}
.scorecard .sub-metrics span {{ display: flex; flex-direction: column; align-items: center; }}
.scorecard .sub-metrics .val {{ font-size: 0.9rem; font-weight: 600; color: var(--text); }}
.badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }}
.badge-winner {{ background: var(--green); color: #fff; }}
.badge-runner {{ background: var(--amber); color: #fff; }}

/* Section header */
.section-divider {{ margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }}
.section-divider h2 {{ font-size: 1rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}

.card {{
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
  transition: background 0.3s, border-color 0.3s; box-shadow: var(--card-shadow);
}}
.card h3 {{ font-size: 1.05rem; margin-bottom: 1rem; font-weight: 600; }}
.chart-container {{ width: 100%; min-height: 350px; }}
.tooltip {{
  position: absolute; padding: 8px 12px; border-radius: 6px;
  background: var(--tooltip-bg); color: var(--tooltip-text);
  font-size: 0.8rem; pointer-events: none; opacity: 0;
  transition: opacity 0.2s; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
th, td {{ padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ color: var(--text-muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
tr:hover td {{ background: var(--bg-hover); }}
.score-bar {{ height: 6px; border-radius: 3px; background: var(--accent); transition: width 0.6s ease-out; }}
.score-cell {{ display: flex; align-items: center; gap: 0.5rem; }}
.score-cell .value {{ min-width: 3rem; text-align: right; font-variant-numeric: tabular-nums; }}
.score-cell .bar-bg {{ flex: 1; height: 6px; border-radius: 3px; background: var(--grid); }}
.tabs {{ display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }}
.tab {{
  padding: 0.4rem 1rem; border-radius: 6px; border: 1px solid var(--border);
  background: var(--bg); color: var(--text-muted); cursor: pointer;
  font-size: 0.85rem; transition: all 0.2s;
}}
.tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
.tab:hover:not(.active) {{ background: var(--bg-hover); }}

details {{ margin-top: 0.5rem; }}
details summary {{ cursor: pointer; color: var(--text-muted); font-size: 0.85rem; padding: 0.3rem 0; }}
details summary:hover {{ color: var(--text); }}

.context-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0; }}
@media (max-width: 768px) {{ .context-grid {{ grid-template-columns: 1fr; }} }}
.doc-card {{ padding: 1rem; background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; }}
.doc-card h4 {{ font-size: 0.85rem; margin-bottom: 0.5rem; }}
.doc-card p {{ font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.3rem; }}
.doc-card a {{ color: var(--accent); text-decoration: none; font-size: 0.8rem; word-break: break-all; }}
.doc-card a:hover {{ text-decoration: underline; }}
.dataset-table {{ font-size: 0.8rem; }}
.dataset-table td {{ vertical-align: top; }}
.dataset-table .q-type {{ display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.65rem; font-weight: 600; text-transform: uppercase; }}
.q-keyword {{ background: var(--accent-light); color: var(--accent); }}
.q-semantic {{ background: var(--green-bg); color: var(--green); }}
</style>
</head>
<body>
<script>
(function(){{
  var t = localStorage.getItem('evalrag-theme') ||
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', t);
}})();
</script>
<button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme">
  <span id="theme-icon">&#9790;</span>
</button>
<div class="container">
  <h1>EvalRAG Report</h1>
  <p class="subtitle" id="report-time"></p>

  <!-- EXECUTIVE SUMMARY -->
  <div class="exec-summary" id="exec-summary"></div>

  <!-- SCORECARDS -->
  <div class="scorecards" id="scorecards"></div>

  <!-- CONTEXT: Documents & Dataset -->
  <div id="context-section"></div>

  <!-- DETAILS SECTION -->
  <div class="section-divider"><h2>Detailed Analysis</h2></div>

  <div class="card">
    <h3>F1 Score Comparison</h3>
    <div id="f1-chart" class="chart-container" style="min-height:280px;"></div>
  </div>

  <div class="card">
    <h3>All Metrics Comparison</h3>
    <div id="bar-chart" class="chart-container"></div>
  </div>

  <div class="card" id="ranking-card" style="display:none;">
    <h3>Ranking Metrics</h3>
    <div id="ranking-chart" class="chart-container"></div>
  </div>

  <div class="card" id="embedding-card" style="display:none;">
    <h3>Embedding Space Projection (PCA)</h3>
    <p style="font-size:0.8rem;color:var(--text-muted);margin-bottom:0.75rem;">Chunks and queries projected to 2D via PCA. Lines connect queries to their top retrieved chunks. Closer = more semantically similar.</p>
    <div class="tabs" id="projection-tabs"></div>
    <div id="embedding-scatter" class="chart-container" style="min-height:500px;"></div>
  </div>

  <div class="card" id="heatmap-card" style="display:none;">
    <h3>Query &times; Chunk Similarity Heatmap</h3>
    <p style="font-size:0.8rem;color:var(--text-muted);margin-bottom:0.75rem;">Cosine similarity between each query embedding and each chunk embedding. Brighter = higher similarity.</p>
    <div id="sim-heatmap" class="chart-container" style="min-height:500px;overflow-x:auto;"></div>
  </div>

  <div class="card">
    <h3>Per-Query Breakdown</h3>
    <div class="tabs" id="query-tabs"></div>
    <div id="query-table"></div>
  </div>

  <div class="card">
    <h3>Score Details</h3>
    <div id="details-table"></div>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
const DATA = {json_data};

// -- Theme ------------------------------------------------------------------
function getTheme() {{
  return localStorage.getItem('evalrag-theme') ||
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
}}
function applyTheme(t) {{
  document.documentElement.setAttribute('data-theme', t);
  document.getElementById('theme-icon').innerHTML = t === 'dark' ? '&#9788;' : '&#9790;';
  localStorage.setItem('evalrag-theme', t);
}}
function toggleTheme() {{
  applyTheme(getTheme() === 'dark' ? 'light' : 'dark');
  renderAll();
}}
applyTheme(getTheme());

function getColors() {{
  return getComputedStyle(document.documentElement).getPropertyValue('--bar-colors').split(',').map(s => s.trim());
}}
function css(v) {{ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }}
function tip() {{ return d3.select('#tooltip'); }}

document.getElementById('report-time').textContent =
  'Generated ' + new Date(DATA.generated).toLocaleString();

// -- Helpers: extract key metrics per experiment ----------------------------
function getExpMetrics(exp) {{
  const r = (m) => {{
    const found = (exp.ranking || []).find(x => x.metric === m);
    return found ? found.value : null;
  }};
  const f1_5 = r('f1@5'), f1_3 = r('f1@3'), f1_1 = r('f1@1');
  const f1 = f1_5 ?? f1_3 ?? f1_1 ?? null;
  const p5 = r('precision@5'), r5 = r('recall@5');
  const mrr = r('mrr'), map = r('map');
  const overlap = exp.mean_scores['answer_overlap'] ?? null;
  return {{ f1, precision: p5, recall: r5, mrr, map, overlap, name: exp.name }};
}}

// -- Executive Summary ------------------------------------------------------
function renderExecSummary() {{
  const el = document.getElementById('exec-summary');
  const mets = DATA.experiments.map(getExpMetrics);
  const withF1 = mets.filter(m => m.f1 !== null);

  if (!withF1.length) {{
    el.style.display = 'none';
    return;
  }}

  withF1.sort((a, b) => b.f1 - a.f1);
  const winner = withF1[0];
  const others = withF1.slice(1);
  const lift = others.length ? ((winner.f1 - others[0].f1) / (others[0].f1 || 1) * 100) : 0;

  let insight = `<strong>${{winner.name}}</strong> achieves the highest F1@5 score of <strong>${{(winner.f1*100).toFixed(1)}}%</strong>`;
  if (others.length && lift > 0) {{
    insight += `, outperforming ${{others.map(o => o.name).join(' and ')}} by <strong>${{lift.toFixed(0)}}%</strong>`;
  }}
  insight += '. ';
  if (winner.mrr !== null) insight += `First relevant result appears at rank ${{(1/winner.mrr).toFixed(1)}} on average (MRR ${{(winner.mrr*100).toFixed(0)}}%).`;

  el.innerHTML = `<h2>Executive Summary</h2><p class="exec-insight">${{insight}}</p>`;
}}

// -- Scorecards -------------------------------------------------------------
function renderScorecards() {{
  const el = document.getElementById('scorecards');
  el.innerHTML = '';
  const mets = DATA.experiments.map(getExpMetrics);
  const colors = getColors();
  const withF1 = mets.filter(m => m.f1 !== null);
  const maxF1 = withF1.length ? Math.max(...withF1.map(m => m.f1)) : 0;

  mets.forEach((m, i) => {{
    const isWinner = m.f1 !== null && m.f1 === maxF1 && withF1.length > 1;
    const f1Pct = m.f1 !== null ? (m.f1 * 100).toFixed(1) : '--';
    const f1Color = m.f1 !== null ? (m.f1 >= 0.7 ? css('--green') : m.f1 >= 0.4 ? css('--amber') : css('--red')) : css('--text-muted');

    let badge = '';
    if (isWinner) badge = '<span class="badge badge-winner">Winner</span>';
    else if (m.f1 !== null && withF1.indexOf(m) === 1) badge = '<span class="badge badge-runner">Runner-up</span>';

    let sub = '';
    if (m.precision !== null) sub += `<span><span class="val">${{(m.precision*100).toFixed(0)}}%</span>Precision</span>`;
    if (m.recall !== null) sub += `<span><span class="val">${{(m.recall*100).toFixed(0)}}%</span>Recall</span>`;
    if (m.mrr !== null) sub += `<span><span class="val">${{(m.mrr*100).toFixed(0)}}%</span>MRR</span>`;
    if (m.overlap !== null) sub += `<span><span class="val">${{(m.overlap*100).toFixed(0)}}%</span>Overlap</span>`;

    el.innerHTML += `
      <div class="scorecard ${{isWinner ? 'winner' : ''}}">
        <div class="name" style="color:${{colors[i % colors.length]}}">${{m.name}} ${{badge}}</div>
        <div class="f1-value" style="color:${{f1Color}}">${{f1Pct}}%</div>
        <div class="f1-label">F1 Score</div>
        <div class="sub-metrics">${{sub}}</div>
      </div>`;
  }});
}}

// -- F1 Horizontal Bar Chart ------------------------------------------------
function renderF1Chart() {{
  const el = document.getElementById('f1-chart');
  el.innerHTML = '';
  const mets = DATA.experiments.map(getExpMetrics).filter(m => m.f1 !== null);
  if (!mets.length) {{ el.innerHTML = '<p style="color:var(--text-muted)">No F1 data</p>'; return; }}
  const colors = getColors();

  const margin = {{top: 10, right: 60, bottom: 30, left: 180}};
  const barH = 44, gap = 12;
  const height = mets.length * (barH + gap) - gap;
  const width = el.clientWidth - margin.left - margin.right;

  const svg = d3.select(el).append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);

  const x = d3.scaleLinear().domain([0, 1]).range([0, width]);
  const y = d3.scaleBand().domain(mets.map(m => m.name)).range([0, height]).padding(0.25);

  svg.append('g').call(d3.axisBottom(x).ticks(5).tickFormat(d => (d*100)+'%'))
    .attr('transform', `translate(0,${{height}})`)
    .call(g => {{ g.selectAll('text').attr('fill', css('--text-muted')); g.select('.domain').attr('stroke', css('--grid')); }});

  svg.append('g').call(d3.axisLeft(y))
    .call(g => {{ g.selectAll('text').attr('fill', css('--text')).style('font-size','0.85rem').style('font-weight','600'); g.select('.domain').remove(); g.selectAll('line').remove(); }});

  [0.25, 0.5, 0.75, 1.0].forEach(v => {{
    svg.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y1', 0).attr('y2', height)
      .attr('stroke', css('--grid')).attr('stroke-dasharray', '3,3').attr('opacity', 0.5);
  }});

  mets.forEach((m, i) => {{
    const ei = DATA.experiments.findIndex(e => e.name === m.name);
    svg.append('rect')
      .attr('x', 0).attr('y', y(m.name)).attr('width', 0).attr('height', y.bandwidth())
      .attr('rx', 6).attr('fill', colors[ei % colors.length])
      .on('mouseover', function(event) {{
        tip().style('opacity', 1).html(`<strong>${{m.name}}</strong><br>F1: ${{(m.f1*100).toFixed(1)}}%<br>P: ${{m.precision !== null ? (m.precision*100).toFixed(1)+'%' : '--'}}<br>R: ${{m.recall !== null ? (m.recall*100).toFixed(1)+'%' : '--'}}`);
      }})
      .on('mousemove', function(event) {{
        tip().style('left', (event.pageX+12)+'px').style('top', (event.pageY-28)+'px');
      }})
      .on('mouseout', function() {{ tip().style('opacity', 0); }})
      .transition().duration(800).delay(i * 120)
      .attr('width', x(m.f1));

    svg.append('text')
      .attr('x', x(m.f1) + 8).attr('y', y(m.name) + y.bandwidth()/2)
      .attr('dominant-baseline', 'middle')
      .attr('fill', css('--text')).style('font-size', '0.9rem').style('font-weight', '700')
      .style('font-variant-numeric', 'tabular-nums')
      .text((m.f1*100).toFixed(1) + '%')
      .style('opacity', 0).transition().duration(400).delay(i * 120 + 600).style('opacity', 1);
  }});
}}

// -- Full Bar Chart ---------------------------------------------------------
function renderBarChart() {{
  const el = document.getElementById('bar-chart');
  el.innerHTML = '';
  const exps = DATA.experiments;
  const metrics = DATA.metrics;
  if (!metrics.length) return;

  const margin = {{top: 20, right: 20, bottom: 80, left: 50}};
  const width = el.clientWidth - margin.left - margin.right;
  const height = 320 - margin.top - margin.bottom;
  const colors = getColors();

  const svg = d3.select(el).append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);

  const x0 = d3.scaleBand().domain(metrics).range([0, width]).padding(0.2);
  const x1 = d3.scaleBand().domain(exps.map(e => e.name)).range([0, x0.bandwidth()]).padding(0.05);
  const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

  svg.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')))
    .call(g => {{ g.selectAll('text').attr('fill', css('--text-muted')); g.selectAll('line').attr('stroke', css('--grid')); g.select('.domain').attr('stroke', css('--grid')); }});

  svg.append('g').attr('transform', `translate(0,${{height}})`)
    .call(d3.axisBottom(x0))
    .call(g => {{ g.selectAll('text').attr('fill', css('--text-muted')).attr('transform','rotate(-35)').style('text-anchor','end'); g.select('.domain').attr('stroke', css('--grid')); }});

  y.ticks(5).forEach(v => {{
    svg.append('line').attr('x1',0).attr('x2',width).attr('y1',y(v)).attr('y2',y(v))
      .attr('stroke', css('--grid')).attr('stroke-dasharray','3,3').attr('opacity',0.5);
  }});

  metrics.forEach(metric => {{
    exps.forEach((exp, ei) => {{
      const val = exp.mean_scores[metric] || 0;
      svg.append('rect')
        .attr('x', x0(metric)+x1(exp.name)).attr('y', y(0))
        .attr('width', x1.bandwidth()).attr('height', 0).attr('rx', 3)
        .attr('fill', colors[ei % colors.length])
        .on('mouseover', function(event) {{
          tip().style('opacity',1).html(`<strong>${{exp.name}}</strong><br>${{metric}}: ${{(val*100).toFixed(1)}}%`);
        }})
        .on('mousemove', function(event) {{
          tip().style('left',(event.pageX+12)+'px').style('top',(event.pageY-28)+'px');
        }})
        .on('mouseout', function() {{ tip().style('opacity',0); }})
        .transition().duration(600).delay(ei*80)
        .attr('y', y(val)).attr('height', height - y(val));
    }});
  }});

  const legend = svg.append('g').attr('transform', `translate(${{width - exps.length * 140}}, -10)`);
  exps.forEach((exp, i) => {{
    const g = legend.append('g').attr('transform', `translate(${{i*140}},0)`);
    g.append('rect').attr('width',12).attr('height',12).attr('rx',2).attr('fill', colors[i % colors.length]);
    g.append('text').attr('x',16).attr('y',10).text(exp.name).attr('fill', css('--text-muted')).style('font-size','0.75rem');
  }});
}}

// -- Ranking Table ----------------------------------------------------------
function renderRanking() {{
  const card = document.getElementById('ranking-card');
  const allRanking = DATA.experiments.flatMap(e => e.ranking || []);
  if (!allRanking.length) {{ card.style.display = 'none'; return; }}
  card.style.display = '';

  const el = document.getElementById('ranking-chart');
  el.innerHTML = '';
  const colors = getColors();

  let html = '<table><thead><tr><th>Metric</th>';
  DATA.experiments.forEach((e, i) => html += `<th style="color:${{colors[i % colors.length]}}">${{e.name}}</th>`);
  html += '</tr></thead><tbody>';

  const metricNames = [...new Set(allRanking.map(r => r.metric))];
  // Show F1 and key metrics first
  const priority = ['f1@5','f1@3','f1@1','mrr','map'];
  metricNames.sort((a,b) => {{
    const ai = priority.indexOf(a), bi = priority.indexOf(b);
    if (ai >= 0 && bi >= 0) return ai - bi;
    if (ai >= 0) return -1;
    if (bi >= 0) return 1;
    return a.localeCompare(b);
  }});

  metricNames.forEach(m => {{
    const isKey = priority.includes(m);
    html += `<tr${{isKey ? ' style="font-weight:600"' : ''}}><td>${{m}}</td>`;
    const vals = DATA.experiments.map(e => {{
      const r = (e.ranking || []).find(x => x.metric === m);
      return r ? r.value : 0;
    }});
    const maxV = Math.max(...vals);
    vals.forEach((v, i) => {{
      const best = v === maxV && maxV > 0 ? `color:${{css('--green')}};font-weight:700` : '';
      html += `<td><div class="score-cell" style="${{best}}"><span class="value">${{(v*100).toFixed(1)}}%</span><div class="bar-bg"><div class="score-bar" style="width:${{v*100}}%"></div></div></div></td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody></table>';
  el.innerHTML = html;
}}

// -- Per-Query Table --------------------------------------------------------
let activeQueryExp = 0;
function renderQueryTabs() {{
  const el = document.getElementById('query-tabs');
  el.innerHTML = '';
  DATA.experiments.forEach((exp, i) => {{
    const btn = document.createElement('button');
    btn.className = 'tab' + (i === activeQueryExp ? ' active' : '');
    btn.textContent = exp.name;
    btn.onclick = () => {{ activeQueryExp = i; renderQueryTabs(); renderQueryTable(); }};
    el.appendChild(btn);
  }});
}}
function renderQueryTable() {{
  const el = document.getElementById('query-table');
  const exp = DATA.experiments[activeQueryExp];
  if (!exp.per_query.length) {{ el.innerHTML = '<p style="color:var(--text-muted)">No queries</p>'; return; }}
  const metrics = Object.keys(exp.per_query[0].scores || {{}});
  let html = '<table><thead><tr><th>Question</th><th>Chunks</th>';
  metrics.forEach(m => html += `<th>${{m}}</th>`);
  html += '</tr></thead><tbody>';
  exp.per_query.forEach(q => {{
    html += `<tr><td><details><summary>${{q.question}}</summary><p style="font-size:0.8rem;color:var(--text-muted);padding:0.5rem 0;max-width:600px">${{q.ground_truth}}</p></details></td><td>${{q.num_chunks}}</td>`;
    metrics.forEach(m => {{
      const v = q.scores[m] || 0;
      const c = v >= 0.7 ? css('--green') : v >= 0.4 ? css('--amber') : css('--red');
      html += `<td><div class="score-cell"><span class="value" style="color:${{c}}">${{(v*100).toFixed(0)}}%</span><div class="bar-bg"><div class="score-bar" style="width:${{v*100}}%;background:${{c}}"></div></div></div></td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody></table>';
  el.innerHTML = html;
}}

// -- Details Table ----------------------------------------------------------
function renderDetails() {{
  const el = document.getElementById('details-table');
  const metrics = DATA.metrics;
  const colors = getColors();
  let html = '<table><thead><tr><th>Experiment</th>';
  metrics.forEach(m => html += `<th>${{m}}</th>`);
  html += '</tr></thead><tbody>';
  DATA.experiments.forEach((exp, i) => {{
    html += `<tr><td style="color:${{colors[i % colors.length]}};font-weight:600">${{exp.name}}</td>`;
    metrics.forEach(m => {{
      const v = exp.mean_scores[m] || 0;
      html += `<td><div class="score-cell"><span class="value">${{(v*100).toFixed(1)}}%</span><div class="bar-bg"><div class="score-bar" style="width:${{v*100}}%"></div></div></div></td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody></table>';
  el.innerHTML = html;
}}

// -- Embedding Scatter Plot -------------------------------------------------
let activeProjectionExp = null;
function renderProjectionTabs() {{
  const el = document.getElementById('projection-tabs');
  const viz = (DATA.context || {{}}).embedding_viz;
  if (!viz) return;
  el.innerHTML = '';
  const expNames = Object.keys(viz.retrieval_edges || {{}});
  if (!activeProjectionExp && expNames.length) activeProjectionExp = expNames[expNames.length - 1];
  expNames.forEach(name => {{
    const btn = document.createElement('button');
    btn.className = 'tab' + (name === activeProjectionExp ? ' active' : '');
    btn.textContent = name;
    btn.onclick = () => {{ activeProjectionExp = name; renderProjectionTabs(); renderEmbeddingScatter(); }};
    el.appendChild(btn);
  }});
}}

function renderEmbeddingScatter() {{
  const card = document.getElementById('embedding-card');
  const viz = (DATA.context || {{}}).embedding_viz;
  if (!viz || !viz.chunk_points) {{ card.style.display = 'none'; return; }}
  card.style.display = '';

  const el = document.getElementById('embedding-scatter');
  el.innerHTML = '';

  const margin = {{top: 30, right: 30, bottom: 40, left: 50}};
  const width = Math.min(el.clientWidth, 900) - margin.left - margin.right;
  const height = 460 - margin.top - margin.bottom;

  const allX = [...viz.chunk_points.map(p => p.x), ...viz.query_points.map(p => p.x)];
  const allY = [...viz.chunk_points.map(p => p.y), ...viz.query_points.map(p => p.y)];
  const pad = 0.08;
  const xExt = d3.extent(allX), yExt = d3.extent(allY);
  const xPad = (xExt[1] - xExt[0]) * pad, yPad = (yExt[1] - yExt[0]) * pad;

  const x = d3.scaleLinear().domain([xExt[0] - xPad, xExt[1] + xPad]).range([0, width]);
  const y = d3.scaleLinear().domain([yExt[0] - yPad, yExt[1] + yPad]).range([height, 0]);

  const svg = d3.select(el).append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);

  // Grid
  svg.append('g').call(d3.axisBottom(x).ticks(6).tickFormat('')).attr('transform', `translate(0,${{height}})`)
    .call(g => {{ g.select('.domain').attr('stroke', css('--grid')); g.selectAll('line').attr('stroke', css('--grid')); }});
  svg.append('g').call(d3.axisLeft(y).ticks(6).tickFormat(''))
    .call(g => {{ g.select('.domain').attr('stroke', css('--grid')); g.selectAll('line').attr('stroke', css('--grid')); }});

  svg.append('text').attr('x', width/2).attr('y', height + 32).attr('text-anchor', 'middle')
    .attr('fill', css('--text-muted')).style('font-size', '0.7rem').text('PC1');
  svg.append('text').attr('x', -height/2).attr('y', -35).attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle').attr('fill', css('--text-muted')).style('font-size', '0.7rem').text('PC2');

  // Retrieval edges
  const edges = (viz.retrieval_edges || {{}})[activeProjectionExp] || [];
  const colors = getColors();
  const expIdx = Object.keys(viz.retrieval_edges || {{}}).indexOf(activeProjectionExp);
  const edgeColor = colors[expIdx % colors.length] || css('--accent');

  edges.forEach((chunkIdxs, qi) => {{
    const qp = viz.query_points[qi];
    if (!qp) return;
    chunkIdxs.forEach((ci, rank) => {{
      const cp = viz.chunk_points[ci];
      if (!cp) return;
      svg.append('line')
        .attr('x1', x(qp.x)).attr('y1', y(qp.y))
        .attr('x2', x(cp.x)).attr('y2', y(cp.y))
        .attr('stroke', edgeColor).attr('stroke-width', rank === 0 ? 1.5 : 0.8)
        .attr('stroke-opacity', rank === 0 ? 0.5 : 0.2)
        .attr('stroke-dasharray', rank > 0 ? '3,3' : 'none');
    }});
  }});

  // Chunk dots
  viz.chunk_points.forEach(p => {{
    svg.append('circle').attr('cx', x(p.x)).attr('cy', y(p.y)).attr('r', 5)
      .attr('fill', css('--text-muted')).attr('fill-opacity', 0.4).attr('stroke', css('--border')).attr('stroke-width', 0.5)
      .on('mouseover', function(event) {{
        tip().style('opacity', 1).html(`<strong>Chunk ${{p.index}}</strong><br>${{p.preview}}...`);
      }})
      .on('mousemove', function(event) {{ tip().style('left',(event.pageX+12)+'px').style('top',(event.pageY-28)+'px'); }})
      .on('mouseout', function() {{ tip().style('opacity', 0); }});
  }});

  // Query dots
  viz.query_points.forEach(p => {{
    const c = p.type === 'keyword' ? css('--accent') : css('--green');
    svg.append('circle').attr('cx', x(p.x)).attr('cy', y(p.y)).attr('r', 7)
      .attr('fill', c).attr('stroke', '#fff').attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .on('mouseover', function(event) {{
        d3.select(this).attr('r', 10);
        tip().style('opacity', 1).html(`<strong>Q${{p.index + 1}}</strong> (${{p.type}})<br>${{p.question}}`);
      }})
      .on('mousemove', function(event) {{ tip().style('left',(event.pageX+12)+'px').style('top',(event.pageY-28)+'px'); }})
      .on('mouseout', function() {{ d3.select(this).attr('r', 7); tip().style('opacity', 0); }});
    svg.append('text').attr('x', x(p.x)).attr('y', y(p.y) - 10)
      .attr('text-anchor', 'middle').attr('fill', c)
      .style('font-size', '0.6rem').style('font-weight', '700').text('Q' + (p.index + 1));
  }});

  // Legend
  const lg = svg.append('g').attr('transform', `translate(${{width - 200}}, 0)`);
  [[css('--accent'), 'Keyword query'], [css('--green'), 'Semantic query'], [css('--text-muted'), 'Document chunk']].forEach(([c, label], i) => {{
    lg.append('circle').attr('cx', 0).attr('cy', i * 18).attr('r', 5).attr('fill', c).attr('fill-opacity', c === css('--text-muted') ? 0.4 : 1);
    lg.append('text').attr('x', 12).attr('y', i * 18 + 4).text(label).attr('fill', css('--text-muted')).style('font-size', '0.7rem');
  }});
}}

// -- Similarity Heatmap -----------------------------------------------------
function renderHeatmap() {{
  const card = document.getElementById('heatmap-card');
  const viz = (DATA.context || {{}}).embedding_viz;
  if (!viz || !viz.similarity_matrix) {{ card.style.display = 'none'; return; }}
  card.style.display = '';

  const el = document.getElementById('sim-heatmap');
  el.innerHTML = '';

  const matrix = viz.similarity_matrix;
  const qLabels = viz.query_labels;
  const cLabels = viz.chunk_labels;
  const nQ = matrix.length, nC = matrix[0].length;

  const cellW = Math.max(16, Math.min(28, (el.clientWidth - 220) / nC));
  const cellH = Math.max(16, Math.min(24, 400 / nQ));
  const margin = {{top: 40, right: 20, bottom: 20, left: 200}};
  const width = nC * cellW;
  const height = nQ * cellH;

  const svg = d3.select(el).append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);

  // Color scale
  const isDark = getTheme() === 'dark';
  const colorScale = d3.scaleSequential()
    .domain([0, 1])
    .interpolator(isDark ? d3.interpolateInferno : d3.interpolateYlOrRd);

  // Cells
  matrix.forEach((row, qi) => {{
    row.forEach((val, ci) => {{
      svg.append('rect')
        .attr('x', ci * cellW).attr('y', qi * cellH)
        .attr('width', cellW - 1).attr('height', cellH - 1)
        .attr('rx', 2).attr('fill', colorScale(val))
        .on('mouseover', function(event) {{
          tip().style('opacity', 1)
            .html(`<strong>Q${{qi+1}} &rarr; C${{ci}}</strong><br>Similarity: ${{(val*100).toFixed(1)}}%<br><em>${{qLabels[qi]}}</em>`);
        }})
        .on('mousemove', function(event) {{ tip().style('left',(event.pageX+12)+'px').style('top',(event.pageY-28)+'px'); }})
        .on('mouseout', function() {{ tip().style('opacity', 0); }});
    }});
  }});

  // Query labels (left)
  qLabels.forEach((label, i) => {{
    svg.append('text').attr('x', -6).attr('y', i * cellH + cellH / 2)
      .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
      .attr('fill', css('--text-muted')).style('font-size', '0.6rem')
      .text(label.length > 30 ? label.slice(0, 28) + '..' : label);
  }});

  // Chunk labels (top)
  cLabels.forEach((label, i) => {{
    svg.append('text').attr('x', i * cellW + cellW / 2).attr('y', -6)
      .attr('text-anchor', 'middle').attr('fill', css('--text-muted')).style('font-size', '0.6rem')
      .text(label);
  }});

  // Color legend
  const legendW = 200, legendH = 10;
  const lg = svg.append('g').attr('transform', `translate(${{width / 2 - legendW / 2}}, ${{height + 6}})`);
  const defs = svg.append('defs');
  const grad = defs.append('linearGradient').attr('id', 'hm-grad');
  d3.range(0, 1.01, 0.1).forEach(t => {{
    grad.append('stop').attr('offset', `${{t*100}}%`).attr('stop-color', colorScale(t));
  }});
  lg.append('rect').attr('width', legendW).attr('height', legendH).attr('rx', 3).style('fill', 'url(#hm-grad)');
  lg.append('text').attr('x', 0).attr('y', legendH + 12).text('0%').attr('fill', css('--text-muted')).style('font-size', '0.6rem');
  lg.append('text').attr('x', legendW).attr('y', legendH + 12).attr('text-anchor', 'end').text('100%').attr('fill', css('--text-muted')).style('font-size', '0.6rem');
  lg.append('text').attr('x', legendW / 2).attr('y', legendH + 12).attr('text-anchor', 'middle').text('Cosine Similarity').attr('fill', css('--text-muted')).style('font-size', '0.6rem');
}}

// -- Context Section --------------------------------------------------------
function renderContext() {{
  const el = document.getElementById('context-section');
  const ctx = DATA.context;
  if (!ctx) {{ el.style.display = 'none'; return; }}

  let html = '<div class="section-divider"><h2>Evaluation Context</h2></div>';
  html += '<div class="context-grid">';

  // Documents card
  if (ctx.documents && ctx.documents.length) {{
    html += '<div class="doc-card"><h4>Source Documents</h4>';
    ctx.documents.forEach(d => {{
      html += `<p><strong>${{d.name}}</strong>`;
      if (d.chunks) html += ` &mdash; ${{d.chunks}} chunks`;
      if (d.chars) html += ` (${{Math.round(d.chars/1000)}}k chars)`;
      html += '</p>';
      if (d.summary) html += `<p>${{d.summary}}</p>`;
      if (d.path) html += `<a href="file://${{d.path}}">${{d.path}}</a>`;
    }});
    html += '</div>';
  }}

  // Dataset summary card
  if (ctx.dataset) {{
    const ds = ctx.dataset;
    html += '<div class="doc-card"><h4>Evaluation Dataset</h4>';
    html += `<p><strong>${{ds.total}} questions</strong>`;
    if (ds.keyword_count !== undefined) html += ` &mdash; ${{ds.keyword_count}} keyword, ${{ds.semantic_count}} semantic`;
    html += '</p>';
    if (ds.summary) html += `<p>${{ds.summary}}</p>`;
    if (ds.path) html += `<a href="file://${{ds.path}}">${{ds.path}}</a>`;
    html += '</div>';
  }}
  html += '</div>';

  // Q&A table
  if (ctx.questions && ctx.questions.length) {{
    html += '<div class="card"><h3>Questions &amp; Expected Answers</h3>';
    html += '<table class="dataset-table"><thead><tr><th>#</th><th>Type</th><th>Question</th><th>Expected Answer (Ground Truth)</th></tr></thead><tbody>';
    ctx.questions.forEach((q, i) => {{
      const typeCls = q.type === 'keyword' ? 'q-keyword' : 'q-semantic';
      html += `<tr><td>${{i+1}}</td><td><span class="q-type ${{typeCls}}">${{q.type}}</span></td><td>${{q.question}}</td><td style="color:var(--text-muted);font-size:0.78rem">${{q.ground_truth}}</td></tr>`;
    }});
    html += '</tbody></table></div>';
  }}

  el.innerHTML = html;
}}

// -- Render All -------------------------------------------------------------
function renderAll() {{
  renderExecSummary();
  renderScorecards();
  renderContext();
  renderF1Chart();
  renderBarChart();
  renderRanking();
  renderQueryTabs();
  renderQueryTable();
  renderDetails();
  renderProjectionTabs();
  renderEmbeddingScatter();
  renderHeatmap();
}}
renderAll();
window.addEventListener('resize', renderAll);
</script>
</body>
</html>"""
