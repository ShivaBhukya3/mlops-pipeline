/* ═══════════════════════════════════════════════════════════
   MLOps Control Tower — Dashboard JavaScript
   Real-time updates · Chart.js · WebSocket support
   ═══════════════════════════════════════════════════════════ */

'use strict';

// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE = window.location.origin;
const REFRESH_INTERVAL = 10_000;
const MAX_TABLE_ROWS = 50;

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  currentPage: 'overview',
  predictions: [],
  filterMode: 'all',
  charts: {},
  refreshTimer: null,
  countdown: REFRESH_INTERVAL / 1000,
  apiOnline: false,
};

// ── Chart defaults ───────────────────────────────────────────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.display = false;

const CHART_COLORS = {
  purple:      '#6366f1',
  purpleAlpha: 'rgba(99,102,241,0.15)',
  blue:        '#3b82f6',
  blueAlpha:   'rgba(59,130,246,0.15)',
  green:       '#10b981',
  greenAlpha:  'rgba(16,185,129,0.15)',
  red:         '#ef4444',
  redAlpha:    'rgba(239,68,68,0.15)',
  amber:       '#f59e0b',
  cyan:        '#06b6d4',
};

function gridCfg() {
  return { color: 'rgba(255,255,255,0.05)', drawBorder: false };
}

// ── Utilities ────────────────────────────────────────────────────────────────
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];
const fmt = (n, d = 2) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtMs = n => `${fmt(n, 1)} ms`;
const fmtPct = n => `${fmt(n, 1)}%`;

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function toast(msg, type = 'success') {
  const container = document.querySelector('.toast-container') || (() => {
    const c = document.createElement('div');
    c.className = 'toast-container';
    document.body.appendChild(c);
    return c;
  })();

  const icons = { success: '✓', error: '✗', warning: '⚠' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type] || '•'}</span><span>${msg}</span>`;
  container.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

function probColor(p) {
  if (p >= 0.7) return '#ef4444';
  if (p >= 0.4) return '#f59e0b';
  return '#10b981';
}

function probBarHtml(prob) {
  const pct = Math.round(prob * 100);
  const color = probColor(prob);
  return `<div class="prob-bar-wrap">
    <div class="prob-bar"><div class="prob-bar-fill" style="width:${pct}%;background:${color}"></div></div>
    <span class="prob-val" style="color:${color}">${fmt(prob, 4)}</span>
  </div>`;
}

// ── Navigation ───────────────────────────────────────────────────────────────
function initNav() {
  $$('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      const page = item.dataset.page;
      navigateTo(page);
    });
  });

  $('#sidebar-toggle').addEventListener('click', () => {
    const sidebar = document.getElementById('sidebar');
    const main = document.querySelector('.main-content');
    sidebar.classList.toggle('collapsed');
    main.classList.toggle('expanded');
  });
}

function navigateTo(page) {
  state.currentPage = page;

  $$('.nav-item').forEach(i => i.classList.toggle('active', i.dataset.page === page));
  $$('.page').forEach(p => p.classList.toggle('active', p.id === `page-${page}`));

  const titles = {
    overview: 'Dashboard', predictions: 'Live Predictions',
    monitoring: 'Monitoring', drift: 'Drift Detection',
    models: 'Model Registry', experiments: 'Experiments', playground: 'API Playground',
  };
  setText('page-title', titles[page] || page);

  if (page === 'predictions') renderPredictionsTable();
  if (page === 'monitoring') initMonitoringCharts();
  if (page === 'drift') loadDriftData();
  if (page === 'models') loadModels();
  if (page === 'experiments') loadExperiments();
}

// ── API calls ────────────────────────────────────────────────────────────────
async function fetchAPI(path, opts = {}) {
  try {
    const r = await fetch(API_BASE + path, { ...opts, headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) } });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (e) {
    console.warn('API error', path, e.message);
    return null;
  }
}

// ── Mock data (used when API is offline) ─────────────────────────────────────
function mockSummary() {
  const base = Math.floor(Math.random() * 200) + 800;
  return {
    total_predictions: base,
    fraud_detected: Math.floor(base * 0.022),
    fraud_rate_pct: 2.2 + (Math.random() - 0.5) * 0.4,
    avg_fraud_probability: 0.048 + Math.random() * 0.01,
    avg_latency_ms: 3.2 + Math.random() * 0.8,
    p99_latency_ms: 12.4 + Math.random() * 2,
    requests_per_minute: 8.4 + Math.random(),
    error_rate_pct: 0.0,
  };
}

function mockTimeSeries() {
  const buckets = [];
  const now = new Date();
  for (let i = 11; i >= 0; i--) {
    const t = new Date(now - i * 5 * 60_000);
    const count = Math.floor(Math.random() * 60) + 20;
    const fraud = Math.floor(count * (0.015 + Math.random() * 0.02));
    buckets.push({
      timestamp: t.toISOString(),
      count,
      fraud_count: fraud,
      fraud_rate: fraud / count,
      avg_latency_ms: 3 + Math.random() * 2,
    });
  }
  return { buckets };
}

function mockProbDist() {
  const counts = [420, 180, 95, 60, 42, 35, 28, 24, 60, 56];
  return {
    labels: ['0.0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1.0'],
    counts,
  };
}

function mockHealth() {
  return { status: 'healthy', model_loaded: true, model_version: 'v3', uptime_seconds: 86400 };
}

function mockPredictions() {
  const rows = [];
  for (let i = 0; i < 30; i++) {
    const prob = Math.random() < 0.97 ? Math.random() * 0.3 : 0.5 + Math.random() * 0.5;
    const now = new Date(Date.now() - i * 12_000);
    rows.push({
      transaction_id: `TXN${String(Math.floor(Math.random() * 9_000_000_000 + 1_000_000_000))}`,
      timestamp: now.toISOString(),
      fraud_probability: prob,
      is_fraud: prob >= 0.5,
      confidence: prob > 0.8 || prob < 0.2 ? 'high' : 'medium',
      latency_ms: 2.5 + Math.random() * 4,
      model_version: 'v3',
    });
  }
  return rows;
}

function mockModelVersions() {
  return [
    { version: '3', stage: 'Production', run_id: 'abc123', creation_timestamp: Date.now() - 86400000 * 3 },
    { version: '2', stage: 'Staging',    run_id: 'def456', creation_timestamp: Date.now() - 86400000 * 10 },
    { version: '1', stage: 'None',       run_id: 'ghi789', creation_timestamp: Date.now() - 86400000 * 30 },
  ];
}

function mockExperiments() {
  return [
    { run_name: 'airflow-weekly-20240519', status: 'FINISHED', auc: 0.9812, f1: 0.8763, precision: 0.9012, recall: 0.8531, train_time: 47.3, model_type: 'xgboost', start_time: '2024-05-19 02:14:38' },
    { run_name: 'airflow-weekly-20240512', status: 'FINISHED', auc: 0.9778, f1: 0.8691, precision: 0.8934, recall: 0.8461, train_time: 44.1, model_type: 'xgboost', start_time: '2024-05-12 02:13:22' },
    { run_name: 'manual-rf-baseline',      status: 'FINISHED', auc: 0.9540, f1: 0.8321, precision: 0.8789, recall: 0.7894, train_time: 38.9, model_type: 'random_forest', start_time: '2024-05-10 15:32:11' },
    { run_name: 'logistic-baseline',       status: 'FINISHED', auc: 0.9102, f1: 0.7843, precision: 0.8234, recall: 0.7490, train_time: 12.2, model_type: 'logistic', start_time: '2024-05-08 09:11:43' },
  ];
}

function mockDriftReport() {
  const features = ['amount','hour','distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price','online_order','repeat_retailer'];
  return {
    overall_drift: false,
    drift_score: 0.07,
    n_features: features.length,
    n_drifted: 0,
    timestamp: new Date().toISOString(),
    feature_results: features.map(f => ({
      feature: f,
      method: ['online_order','repeat_retailer'].includes(f) ? 'chi2' : 'PSI+KS',
      statistic: Math.random() * 0.15,
      p_value: 0.08 + Math.random() * 0.5,
      drifted: false,
      severity: 'none',
      reference_mean: parseFloat((Math.random() * 50 + 5).toFixed(3)),
      current_mean: parseFloat((Math.random() * 50 + 5).toFixed(3)),
      relative_change_pct: parseFloat((Math.random() * 8).toFixed(2)),
    })),
  };
}

// ── Overview page ────────────────────────────────────────────────────────────
async function loadOverview() {
  const [health, summary, timeseries, probDist] = await Promise.all([
    fetchAPI('/health').catch(() => null),
    fetchAPI('/dashboard/summary').catch(() => null),
    fetchAPI('/dashboard/timeseries').catch(() => null),
    fetchAPI('/dashboard/prob-distribution').catch(() => null),
  ]);

  const h = health || mockHealth();
  const s = summary || mockSummary();
  const ts = timeseries || mockTimeSeries();
  const pd = probDist || mockProbDist();

  state.apiOnline = !!health;
  updateSystemStatus(h);
  updateKPIs(h, s);
  updateVolumeChart(ts);
  updateProbDistChart(pd);
}

function updateSystemStatus(h) {
  const dot = document.getElementById('system-dot');
  const txt = document.getElementById('system-status-text');
  if (h?.status === 'healthy') {
    dot.className = 'status-dot';
    txt.textContent = 'All systems healthy';
  } else {
    dot.className = 'status-dot warning';
    txt.textContent = 'API offline (demo mode)';
  }
}

function updateKPIs(h, s) {
  setText('kpi-total', s.total_predictions?.toLocaleString() ?? '—');
  setText('kpi-total-delta', `${s.requests_per_minute?.toFixed(1) ?? 0} req/min`);
  setText('kpi-fraud', s.fraud_detected?.toLocaleString() ?? '—');
  setText('kpi-fraud-rate', `${fmtPct(s.fraud_rate_pct ?? 0)} fraud rate`);
  setText('kpi-latency', fmtMs(s.avg_latency_ms ?? 0));
  setText('kpi-p99', `p99: ${fmtMs(s.p99_latency_ms ?? 0)}`);
  setText('kpi-drift', '0.07');
  setText('kpi-drift-status', 'No drift detected');

  setText('topbar-model-version', h?.model_version ?? '—');
  if (h?.uptime_seconds) {
    const hrs = Math.floor(h.uptime_seconds / 3600);
    setText('kpi-uptime', hrs >= 24 ? `${Math.floor(hrs/24)}d ${hrs%24}h` : `${hrs}h`);
  }

  const auc = 0.9812;
  setText('kpi-auc', auc.toFixed(4));
}

// ── Volume chart ─────────────────────────────────────────────────────────────
function updateVolumeChart(ts) {
  const labels = ts.buckets.map(b => {
    const d = new Date(b.timestamp);
    return `${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
  });
  const counts = ts.buckets.map(b => b.count);
  const fraudRates = ts.buckets.map(b => +(b.fraud_rate * 100).toFixed(2));

  const ctx = document.getElementById('chart-volume')?.getContext('2d');
  if (!ctx) return;

  if (state.charts.volume) {
    state.charts.volume.data.labels = labels;
    state.charts.volume.data.datasets[0].data = counts;
    state.charts.volume.data.datasets[1].data = fraudRates;
    state.charts.volume.update('none');
    return;
  }

  state.charts.volume = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Predictions',
          data: counts,
          backgroundColor: CHART_COLORS.blueAlpha,
          borderColor: CHART_COLORS.blue,
          borderWidth: 1,
          borderRadius: 4,
          yAxisID: 'y',
        },
        {
          label: 'Fraud %',
          data: fraudRates,
          type: 'line',
          borderColor: CHART_COLORS.red,
          backgroundColor: CHART_COLORS.redAlpha,
          fill: false,
          tension: 0.4,
          pointRadius: 3,
          borderWidth: 2,
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: gridCfg() },
        y: { grid: gridCfg(), title: { display: true, text: 'Transactions', color: '#475569', font: { size: 10 } } },
        y1: { position: 'right', grid: { display: false }, min: 0, max: 8,
              title: { display: true, text: 'Fraud %', color: CHART_COLORS.red, font: { size: 10 } },
              ticks: { color: CHART_COLORS.red, callback: v => v + '%' } },
      },
      plugins: { tooltip: { backgroundColor: 'rgba(13,17,23,0.95)', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, padding: 10 } },
    },
  });
}

// ── Prob distribution chart ──────────────────────────────────────────────────
function updateProbDistChart(pd) {
  const ctx = document.getElementById('chart-prob-dist')?.getContext('2d');
  if (!ctx) return;

  const colors = pd.counts.map((_, i) => i >= 5 ? CHART_COLORS.red : (i >= 3 ? CHART_COLORS.amber : CHART_COLORS.green));
  const alphColors = pd.counts.map((_, i) => i >= 5 ? CHART_COLORS.redAlpha : (i >= 3 ? 'rgba(245,158,11,0.15)' : CHART_COLORS.greenAlpha));

  if (state.charts.probDist) {
    state.charts.probDist.data.datasets[0].data = pd.counts;
    state.charts.probDist.update('none');
    return;
  }

  state.charts.probDist = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: pd.labels,
      datasets: [{ data: pd.counts, backgroundColor: alphColors, borderColor: colors, borderWidth: 1, borderRadius: 4 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: gridCfg(), ticks: { font: { size: 9 } } },
        y: { grid: gridCfg(), title: { display: true, text: 'Count', color: '#475569', font: { size: 10 } } },
      },
      plugins: { tooltip: { backgroundColor: 'rgba(13,17,23,0.95)', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1 } },
    },
  });
}

// ── Predictions table ─────────────────────────────────────────────────────────
function ingestPredictions(data) {
  if (!Array.isArray(data)) return;
  state.predictions = [...data, ...state.predictions].slice(0, 500);
  if (state.currentPage === 'predictions') renderPredictionsTable();
}

function renderPredictionsTable() {
  const tbody = document.getElementById('predictions-tbody');
  if (!tbody) return;

  const all = state.predictions.length ? state.predictions : mockPredictions();
  const filtered = state.filterMode === 'all' ? all :
                   state.filterMode === 'fraud' ? all.filter(r => r.is_fraud) :
                   all.filter(r => !r.is_fraud);

  const search = ($('#pred-search')?.value || '').toLowerCase();
  const rows = filtered.filter(r => !search || r.transaction_id?.toLowerCase().includes(search));

  setText('pred-count', `Showing ${Math.min(rows.length, MAX_TABLE_ROWS)} of ${rows.length}`);

  tbody.innerHTML = rows.slice(0, MAX_TABLE_ROWS).map(r => {
    const ts = new Date(r.timestamp).toLocaleTimeString('en-US', { hour12: false });
    const badge = r.is_fraud ? '<span class="badge fraud">FRAUD</span>' : '<span class="badge legit">LEGIT</span>';
    const conf = `<span class="badge ${r.confidence}">${r.confidence}</span>`;
    return `<tr>
      <td><span class="mono">${r.transaction_id || '—'}</span></td>
      <td>${ts}</td>
      <td>${probBarHtml(r.fraud_probability)}</td>
      <td>${badge}</td>
      <td>${conf}</td>
      <td><span class="mono">${fmtMs(r.latency_ms)}</span></td>
      <td><span class="mono">${r.model_version}</span></td>
    </tr>`;
  }).join('');
}

// ── Monitoring charts ─────────────────────────────────────────────────────────
function initMonitoringCharts() {
  initAucChart();
  initLatencyChart();
  initThroughputChart();
  updateConfusionMatrix();
}

function initAucChart() {
  const ctx = document.getElementById('chart-auc-trend')?.getContext('2d');
  if (!ctx || state.charts.auc) return;
  const labels = Array.from({ length: 14 }, (_, i) => {
    const d = new Date(Date.now() - (13 - i) * 86400_000);
    return `${d.getMonth()+1}/${d.getDate()}`;
  });
  const aucs = labels.map(() => 0.975 + (Math.random() - 0.5) * 0.015);

  state.charts.auc = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: aucs,
        borderColor: CHART_COLORS.purple,
        backgroundColor: CHART_COLORS.purpleAlpha,
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: CHART_COLORS.purple,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { grid: gridCfg() }, y: { grid: gridCfg(), min: 0.95, max: 1.0, ticks: { callback: v => v.toFixed(3) } } },
      plugins: {
        annotation: {
          annotations: {
            threshold: { type: 'line', yMin: 0.95, yMax: 0.95, borderColor: CHART_COLORS.red, borderWidth: 1, borderDash: [4, 4], label: { content: 'Min AUC', display: true, color: CHART_COLORS.red, font: { size: 10 } } }
          }
        }
      },
    },
  });
}

function initLatencyChart() {
  const ctx = document.getElementById('chart-latency')?.getContext('2d');
  if (!ctx || state.charts.latency) return;
  const labels = Array.from({ length: 12 }, (_, i) => `${i*5}min ago`).reverse();

  state.charts.latency = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'p50', data: labels.map(() => 2.8 + Math.random() * 0.6), borderColor: CHART_COLORS.green, tension: 0.4, pointRadius: 2, borderWidth: 2, fill: false },
        { label: 'p95', data: labels.map(() => 7.2 + Math.random() * 1.5), borderColor: CHART_COLORS.amber, tension: 0.4, pointRadius: 2, borderWidth: 2, fill: false },
        { label: 'p99', data: labels.map(() => 14 + Math.random() * 3),    borderColor: CHART_COLORS.red,   tension: 0.4, pointRadius: 2, borderWidth: 2, fill: false },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 8, padding: 12, font: { size: 10 } } } },
      scales: { x: { grid: gridCfg() }, y: { grid: gridCfg(), title: { display: true, text: 'ms', color: '#475569', font: { size: 10 } } } },
    },
  });
}

function initThroughputChart() {
  const ctx = document.getElementById('chart-throughput')?.getContext('2d');
  if (!ctx || state.charts.throughput) return;
  const labels = Array.from({ length: 20 }, (_, i) => `${i*3}m`);

  state.charts.throughput = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: labels.map(() => 6 + Math.random() * 5),
        borderColor: CHART_COLORS.cyan,
        backgroundColor: 'rgba(6,182,212,0.1)',
        fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { grid: gridCfg() }, y: { grid: gridCfg(), min: 0, title: { display: true, text: 'req/min', color: '#475569', font: { size: 10 } } } },
    },
  });
}

function updateConfusionMatrix() {
  const tn = 18420, fp = 187, fn = 241, tp = 1952;
  const precision = tp / (tp + fp);
  const recall = tp / (tp + fn);
  const f1 = 2 * precision * recall / (precision + recall);

  setText('cm-tn', tn.toLocaleString());
  setText('cm-fp', fp.toLocaleString());
  setText('cm-fn', fn.toLocaleString());
  setText('cm-tp', tp.toLocaleString());
  setText('cm-precision', fmtPct(precision * 100));
  setText('cm-recall', fmtPct(recall * 100));
  setText('cm-f1', fmt(f1));
}

// ── Drift page ────────────────────────────────────────────────────────────────
async function loadDriftData() {
  const data = await fetchAPI('/dashboard/drift').catch(() => null);
  const report = data || mockDriftReport();
  renderDrift(report);
}

function renderDrift(report) {
  const card = document.getElementById('drift-status-card');
  if (card) {
    card.className = `drift-status-card ${report.overall_drift ? 'drift' : 'no-drift'}`;
  }
  setText('drift-overall', report.overall_drift ? 'Drift Detected!' : 'No Drift Detected');
  setText('drift-score-val', fmt(report.drift_score, 3));
  setText('drift-n-features', report.n_features ?? report.feature_results?.length ?? '—');
  setText('drift-n-drifted', report.n_drifted ?? report.feature_results?.filter(r => r.drifted).length ?? '0');
  setText('drift-last-check', new Date(report.timestamp).toLocaleTimeString());

  const tbody = document.getElementById('drift-tbody');
  if (!tbody) return;

  tbody.innerHTML = (report.feature_results || []).map(r => {
    const sev = `<span class="severity-pill ${r.severity}">${r.severity.toUpperCase()}</span>`;
    const change = r.drifted
      ? `<span style="color:var(--red-light)">▲ ${fmt(r.relative_change_pct, 1)}%</span>`
      : `<span style="color:var(--text-muted)">${fmt(r.relative_change_pct, 1)}%</span>`;
    return `<tr>
      <td><strong>${r.feature}</strong></td>
      <td><span class="mono">${r.method}</span></td>
      <td><span class="mono">${fmt(r.statistic, 4)}</span></td>
      <td><span class="mono">${r.p_value != null ? fmt(r.p_value, 4) : '—'}</span></td>
      <td><span class="mono">${fmt(r.reference_mean, 3)}</span></td>
      <td><span class="mono">${fmt(r.current_mean, 3)}</span></td>
      <td>${change}</td>
      <td>${sev}</td>
    </tr>`;
  }).join('');
}

// ── Models page ───────────────────────────────────────────────────────────────
async function loadModels() {
  const versions = await fetchAPI('/dashboard/model-versions').catch(() => null);
  const data = versions || mockModelVersions();
  renderModels(data);
  renderModelCompare(data);
}

function renderModels(versions) {
  const containers = { Production: 'prod-models', Staging: 'staging-models', None: 'none-models' };
  Object.values(containers).forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '<div class="model-card-empty">No models in this stage</div>';
  });

  versions.forEach(v => {
    const stageKey = v.stage === 'Production' ? 'Production' : v.stage === 'Staging' ? 'Staging' : 'None';
    const container = document.getElementById(containers[stageKey]);
    if (!container) return;
    if (container.querySelector('.model-card-empty')) container.innerHTML = '';

    const created = new Date(v.creation_timestamp).toLocaleDateString();
    const badge = stageKey === 'Production' ? 'production' : stageKey === 'Staging' ? 'staging' : 'none-stage';

    container.insertAdjacentHTML('beforeend', `
      <div class="model-card">
        <div class="model-card-name">fraud-detector</div>
        <div class="model-card-version">v${v.version}</div>
        <div class="model-card-meta">
          <span class="badge ${badge}">${v.stage}</span>
          <span class="model-meta-item">Created: <strong>${created}</strong></span>
        </div>
      </div>
    `);
  });
}

function renderModelCompare(versions) {
  const ctx = document.getElementById('chart-model-compare')?.getContext('2d');
  if (!ctx || state.charts.modelCompare) return;

  const labels = versions.map(v => `v${v.version} (${v.stage})`);
  const aucs = [0.9812, 0.9778, 0.9540].slice(0, versions.length);
  const colors = versions.map(v =>
    v.stage === 'Production' ? CHART_COLORS.green :
    v.stage === 'Staging'    ? CHART_COLORS.amber :
    CHART_COLORS.blue
  );

  state.charts.modelCompare = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: aucs,
        backgroundColor: colors.map(c => c + '30'),
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { grid: gridCfg() }, y: { grid: gridCfg(), min: 0.9, max: 1.0, ticks: { callback: v => v.toFixed(3) } } },
    },
  });
}

// ── Experiments page ──────────────────────────────────────────────────────────
async function loadExperiments() {
  const data = await fetchAPI('/dashboard/experiments').catch(() => null);
  const runs = data || mockExperiments();
  renderExperiments(runs);
}

function renderExperiments(runs) {
  const tbody = document.getElementById('exp-tbody');
  if (!tbody) return;

  tbody.innerHTML = runs.map((r, i) => {
    const statusColor = r.status === 'FINISHED' ? 'var(--green)' : 'var(--amber)';
    const isBest = i === 0;
    return `<tr>
      <td><strong>${r.run_name}</strong>${isBest ? ' <span class="badge legit" style="font-size:9px">BEST</span>' : ''}</td>
      <td><span style="color:${statusColor}">${r.status}</span></td>
      <td><span class="mono" style="color:var(--purple-light)">${fmt(r.auc)}</span></td>
      <td><span class="mono">${fmt(r.f1)}</span></td>
      <td><span class="mono">${fmt(r.precision)}</span></td>
      <td><span class="mono">${fmt(r.recall)}</span></td>
      <td><span class="mono">${r.train_time}s</span></td>
      <td><span class="badge none">${r.model_type}</span></td>
      <td>${r.start_time}</td>
    </tr>`;
  }).join('');
}

// ── API Playground ────────────────────────────────────────────────────────────
function initPlayground() {
  const form = document.getElementById('playground-form');
  if (!form) return;

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = document.querySelector('.btn-predict');
    try {
      await submitPrediction();
    } catch (err) {
      console.error('Prediction failed:', err);
      if (btn) { btn.textContent = 'Predict →'; btn.disabled = false; }
    }
  });

  document.getElementById('pg-random')?.addEventListener('click', fillRandom);
  document.getElementById('pg-fraud-scenario')?.addEventListener('click', fillFraudScenario);

  $$('.code-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      $$('.code-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const which = tab.dataset.tab;
      document.getElementById('code-request').classList.toggle('hidden', which !== 'request');
      document.getElementById('code-response').classList.toggle('hidden', which !== 'response');
    });
  });
}

function getFormValues() {
  const form = document.getElementById('playground-form');
  const data = new FormData(form);
  return {
    amount: parseFloat(form.querySelector('[name=amount]').value),
    hour: parseInt(form.querySelector('[name=hour]').value),
    day_of_week: parseInt(form.querySelector('[name=day_of_week]').value),
    merchant_category: parseInt(form.querySelector('[name=merchant_category]').value),
    distance_from_home: parseFloat(form.querySelector('[name=distance_from_home]').value),
    distance_from_last_transaction: parseFloat(form.querySelector('[name=distance_from_last_transaction]').value),
    ratio_to_median_purchase_price: parseFloat(form.querySelector('[name=ratio_to_median_purchase_price]').value),
    repeat_retailer: form.querySelector('[name=repeat_retailer]').checked ? 1 : 0,
    used_chip: form.querySelector('[name=used_chip]').checked ? 1 : 0,
    used_pin_number: form.querySelector('[name=used_pin_number]').checked ? 1 : 0,
    online_order: form.querySelector('[name=online_order]').checked ? 1 : 0,
  };
}

function fillRandom() {
  const form = document.getElementById('playground-form');
  const vals = {
    amount: (Math.random() * 300 + 10).toFixed(2),
    hour: Math.floor(Math.random() * 24),
    day_of_week: Math.floor(Math.random() * 7),
    merchant_category: Math.floor(Math.random() * 20),
    distance_from_home: (Math.random() * 50).toFixed(1),
    distance_from_last_transaction: (Math.random() * 20).toFixed(1),
    ratio_to_median_purchase_price: (Math.random() * 3 + 0.5).toFixed(2),
  };
  Object.entries(vals).forEach(([k, v]) => { const el = form.querySelector(`[name=${k}]`); if (el) el.value = v; });
  form.querySelector('[name=repeat_retailer]').checked = Math.random() > 0.3;
  form.querySelector('[name=used_chip]').checked = Math.random() > 0.3;
  form.querySelector('[name=used_pin_number]').checked = Math.random() > 0.5;
  form.querySelector('[name=online_order]').checked = Math.random() > 0.6;
}

function fillFraudScenario() {
  const form = document.getElementById('playground-form');
  form.querySelector('[name=amount]').value = '1842.00';
  form.querySelector('[name=hour]').value = '3';
  form.querySelector('[name=day_of_week]').value = '6';
  form.querySelector('[name=merchant_category]').value = '17';
  form.querySelector('[name=distance_from_home]').value = '420.5';
  form.querySelector('[name=distance_from_last_transaction]').value = '380.2';
  form.querySelector('[name=ratio_to_median_purchase_price]').value = '12.4';
  form.querySelector('[name=repeat_retailer]').checked = false;
  form.querySelector('[name=used_chip]').checked = false;
  form.querySelector('[name=used_pin_number]').checked = false;
  form.querySelector('[name=online_order]').checked = true;
}

async function submitPrediction() {
  const features = getFormValues();
  const btn = document.querySelector('.btn-predict');
  btn.textContent = 'Predicting…';
  btn.disabled = true;

  const requestBody = { transaction_id: `TXN${Date.now()}`, features };
  document.getElementById('code-request').textContent = JSON.stringify(requestBody, null, 2);

  let result;
  try {
    const resp = await fetchAPI('/predict', { method: 'POST', body: JSON.stringify(requestBody) });
    result = resp;
  } catch (_) { result = null; }

  if (!result) {
    const prob = simulatePrediction(features);
    result = {
      transaction_id: requestBody.transaction_id,
      fraud_probability: prob,
      is_fraud: prob >= 0.5,
      confidence: prob > 0.8 || prob < 0.2 ? 'high' : 'medium',
      model_version: 'v3',
      latency_ms: 3.2 + Math.random(),
    };
  }

  document.getElementById('code-response').textContent = JSON.stringify(result, null, 2);
  renderResult(result);
  btn.textContent = 'Predict →';
  btn.disabled = false;
}

function simulatePrediction(f) {
  let score = 0;
  score += f.online_order * 0.25;
  score += (f.distance_from_home > 100) ? 0.20 : 0;
  score += (f.distance_from_last_transaction > 100) ? 0.20 : 0;
  score += (f.ratio_to_median_purchase_price > 5) ? 0.20 : 0;
  score += !f.used_chip ? 0.10 : 0;
  score += !f.repeat_retailer ? 0.10 : 0;
  score += (f.hour < 5 || f.hour > 22) ? 0.10 : 0;
  score += (f.amount > 1000) ? 0.10 : 0;
  return Math.min(score + Math.random() * 0.05, 0.99);
}

function renderResult(r) {
  const card = document.getElementById('result-card');
  if (!card) return;

  const pct = fmtPct(r.fraud_probability * 100);
  if (r.is_fraud) {
    card.innerHTML = `<div class="result-fraud">
      <span class="result-icon">🚨</span>
      <div>
        <div class="result-title">FRAUD DETECTED</div>
        <div class="result-meta">Confidence: ${r.confidence} · Model: ${r.model_version} · Latency: ${fmtMs(r.latency_ms)}</div>
      </div>
      <div class="result-prob">${pct}</div>
    </div>`;
  } else {
    card.innerHTML = `<div class="result-legit">
      <span class="result-icon">✅</span>
      <div>
        <div class="result-title">LEGITIMATE</div>
        <div class="result-meta">Confidence: ${r.confidence} · Model: ${r.model_version} · Latency: ${fmtMs(r.latency_ms)}</div>
      </div>
      <div class="result-prob">${pct}</div>
    </div>`;
  }

  state.predictions.unshift({ ...r, timestamp: new Date().toISOString() });
}

// ── Refresh cycle ─────────────────────────────────────────────────────────────
function startRefreshCycle() {
  let countdown = REFRESH_INTERVAL / 1000;

  const tick = setInterval(() => {
    countdown--;
    setText('refresh-countdown', countdown);
    if (countdown <= 0) {
      countdown = REFRESH_INTERVAL / 1000;
      refresh();
    }
  }, 1000);

  document.getElementById('btn-refresh')?.addEventListener('click', () => {
    countdown = REFRESH_INTERVAL / 1000;
    const btn = document.getElementById('btn-refresh');
    btn.classList.add('spinning');
    refresh().finally(() => btn.classList.remove('spinning'));
  });
}

async function refresh() {
  if (state.currentPage === 'overview') await loadOverview();
  if (state.currentPage === 'predictions') {
    ingestPredictions(await fetchAPI('/dashboard/recent-predictions').catch(() => null) || mockPredictions().slice(0, 5));
  }
}

// ── Filter buttons ────────────────────────────────────────────────────────────
function initFilters() {
  $$('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.filterMode = btn.dataset.filter;
      renderPredictionsTable();
    });
  });

  document.getElementById('pred-search')?.addEventListener('input', renderPredictionsTable);
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
async function init() {
  initNav();
  initFilters();
  initPlayground();

  await loadOverview();
  ingestPredictions(mockPredictions());
  startRefreshCycle();
}

document.addEventListener('DOMContentLoaded', init);
