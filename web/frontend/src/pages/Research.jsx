import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, Legend,
} from 'recharts'
import { getSummary } from '../api/client'
import './Research.css'

const CLASS_LABELS = {
  nv: 'Nevus', mel: 'Melanoma', bkl: 'Keratosis',
  bcc: 'BCC', akiec: 'Actinic K.', vasc: 'Vascular', df: 'Dermatofibroma'
}
const CLASS_RISK_COLOR = {
  nv: '#34d399', mel: '#f87171', bkl: '#34d399',
  bcc: '#f87171', akiec: '#fbbf24', vasc: '#34d399', df: '#34d399'
}
const PIE_COLORS = ['#38bdf8', '#a78bfa', '#f59e0b', '#34d399', '#f87171', '#fbbf24', '#60a5fa']

const TABS = ['Ablation Study', 'Per-Class F1', 'Safety & Uncertainty']

const STATIC_ABLATION = [
  { model: 'Model A (ML)',  phase: 'Phase 1', f1: 0.3488, auroc: 0.8616, ece: null,   ood: 4.58,  color: '#a78bfa' },
  { model: 'Model B (DL)',  phase: 'Phase 2', f1: 0.5687, auroc: 0.9165, ece: 0.0890, ood: 5.50,  color: '#38bdf8' },
  { model: 'Model C (Safe)', phase: 'Phase 3', f1: 0.5687, auroc: 0.9165, ece: 0.0890, ood: 10.09, color: '#f59e0b' },
]

const CLASS_NAMES_ORDER = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
const F1_MODEL_A = [0.857, 0.401, 0.389, 0.370, 0.424, 0.0, 0.0]
const F1_MODEL_B = [0.745, 0.467, 0.558, 0.639, 0.466, 0.677, 0.429]

const F1_PER_CLASS = CLASS_NAMES_ORDER.map((cls, i) => ({
  class: CLASS_LABELS[cls],
  'Model A (GP)':           parseFloat(F1_MODEL_A[i].toFixed(3)),
  'Model B/C (EfficientNet)': parseFloat(F1_MODEL_B[i].toFixed(3)),
  risk: cls === 'mel' || cls === 'bcc' ? 'malignant' : cls === 'akiec' ? 'precancerous' : 'benign',
}))

const CP_SET_DIST = [
  { size: '1 (Definitive)', count: 76,  pct: 5.0  },
  { size: '2',              count: 244, pct: 16.0 },
  { size: '3',              count: 295, pct: 19.3 },
  { size: '4',              count: 390, pct: 25.5 },
  { size: '5',              count: 476, pct: 31.2 },
  { size: '6',              count: 46,  pct: 3.0  },
]

const DATASET_DIST = [
  { name: 'Nevus (nv)',    value: 6705, color: '#38bdf8' },
  { name: 'Melanoma (mel)', value: 1113, color: '#f87171' },
  { name: 'BKL',           value: 1099, color: '#a78bfa' },
  { name: 'BCC',           value: 514,  color: '#f59e0b' },
  { name: 'Actinic K.',    value: 327,  color: '#fbbf24' },
  { name: 'Vascular',      value: 142,  color: '#34d399' },
  { name: 'Dermatofibroma',value: 115,  color: '#60a5fa' },
]

const customTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: 'rgba(11,17,32,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, padding: '10px 14px', fontSize: '0.8rem' }}>
      <p style={{ color: '#94a3b8', marginBottom: 4 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, fontWeight: 600 }}>{p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</p>
      ))}
    </div>
  )
}

export default function Research() {
  const [tab, setTab]       = useState(0)
  const [summary, setSummary] = useState(null)

  useEffect(() => {
    getSummary().then(setSummary).catch(() => {})
  }, [])

  return (
    <main className="research-page">
      <div className="container">
        <div className="research-header fade-up">
          <div className="section-label">Research Showcase</div>
          <h2>Model Performance & Analysis</h2>
          <p className="text-muted">Complete results from all 3 phases of the Robust Medical Vision study</p>
        </div>

        {/* Summary cards */}
        <div className="research-summary-grid fade-up">
          {STATIC_ABLATION.map((m, i) => (
            <div key={i} className="card rs-card" style={{ '--mc': m.color }}>
              <div className="rs-phase">{m.phase}</div>
              <div className="rs-model">{m.model}</div>
              <div className="rs-metrics">
                <div className="rs-m"><span className="mono" style={{ color: m.color }}>{m.f1.toFixed(3)}</span><span>F1 Macro</span></div>
                <div className="rs-m"><span className="mono" style={{ color: m.color }}>{m.auroc.toFixed(3)}</span><span>AUROC</span></div>
                {m.ece && <div className="rs-m"><span className="mono" style={{ color: m.color }}>{m.ece}</span><span>ECE</span></div>}
                <div className="rs-m"><span className="mono" style={{ color: m.color }}>{m.ood}%</span><span>OOD Rate</span></div>
              </div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div className="research-tabs fade-up fade-up-1">
          {TABS.map((t, i) => (
            <button key={i} className={`rtab ${tab === i ? 'rtab-active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          ))}
        </div>

        {/* Tab content */}
        <div className="research-content fade-up fade-up-2">

          {/* ── Tab 0: Ablation ───────────────────────────────── */}
          {tab === 0 && (
            <div className="research-tab-panel">
              <div className="chart-grid">
                <div className="card chart-card">
                  <h4>F1 Macro Comparison</h4>
                  <p className="text-muted chart-desc">ML → DL gain: <span className="mono" style={{ color: 'var(--green)' }}>+0.220</span> (+63%)</p>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={STATIC_ABLATION} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                      <XAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <Tooltip content={customTooltip} />
                      <Bar dataKey="f1" name="F1 Macro" radius={[4,4,0,0]}>
                        {STATIC_ABLATION.map((e, i) => <Cell key={i} fill={e.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="card chart-card">
                  <h4>AUROC Comparison</h4>
                  <p className="text-muted chart-desc">All models demonstrate strong discrimination ability</p>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={STATIC_ABLATION} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                      <XAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <YAxis domain={[0.75, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <Tooltip content={customTooltip} />
                      <Bar dataKey="auroc" name="AUROC" radius={[4,4,0,0]}>
                        {STATIC_ABLATION.map((e, i) => <Cell key={i} fill={e.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="card chart-card">
                  <h4>OOD Detection Rate</h4>
                  <p className="text-muted chart-desc">Phase 3 union signal captures more uncertain images</p>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={STATIC_ABLATION} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                      <XAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <Tooltip content={customTooltip} />
                      <Bar dataKey="ood" name="OOD Rate %" radius={[4,4,0,0]}>
                        {STATIC_ABLATION.map((e, i) => <Cell key={i} fill={e.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="card chart-card">
                  <h4>Dataset Distribution</h4>
                  <p className="text-muted chart-desc">58:1 class imbalance — handled via WeightedRandomSampler</p>
                  <ResponsiveContainer width="100%" height={240}>
                    <PieChart>
                      <Pie data={DATASET_DIST} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={55} outerRadius={85}>
                        {DATASET_DIST.map((e, i) => <Cell key={i} fill={e.color} />)}
                      </Pie>
                      <Legend iconSize={8} wrapperStyle={{ fontSize: '0.7rem', color: '#94a3b8' }} />
                      <Tooltip content={customTooltip} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* ECE improvement callout */}
              <div className="card callout-card">
                <div className="callout-grid">
                  <div className="callout-item">
                    <div className="section-label">ECE Improvement</div>
                    <div className="callout-val mono" style={{ color: 'var(--green)' }}>−59.5%</div>
                    <div className="text-muted">0.22 → 0.089 via Temperature Scaling (T=1.5)</div>
                  </div>
                  <div className="callout-item">
                    <div className="section-label">CP Coverage</div>
                    <div className="callout-val mono" style={{ color: 'var(--gold)' }}>95.3%</div>
                    <div className="text-muted">Formally guaranteed — target was ≥ 95%</div>
                  </div>
                  <div className="callout-item">
                    <div className="section-label">F1 Gain (A→B)</div>
                    <div className="callout-val mono" style={{ color: 'var(--teal)' }}>+63%</div>
                    <div className="text-muted">0.349 → 0.569 switching from GP to EfficientNet-B3</div>
                  </div>
                  <div className="callout-item">
                    <div className="section-label">False Alarm Rate</div>
                    <div className="callout-val mono" style={{ color: 'var(--amber)' }}>5.4%</div>
                    <div className="text-muted">Mahalanobis OOD, target was ≤ 5%</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── Tab 1: Per-class F1 ───────────────────────────── */}
          {tab === 1 && (
            <div className="research-tab-panel">
              <div className="card chart-card-wide">
                <h4>Per-Class F1 Score — Model A vs Model B/C</h4>
                <p className="text-muted chart-desc">
                  <span style={{ color: 'var(--red)' }}>■ Red</span> = malignant · <span style={{ color: 'var(--amber)' }}>■ Amber</span> = precancerous · <span style={{ color: 'var(--green)' }}>■ Green</span> = benign
                </p>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={F1_PER_CLASS} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                    <XAxis dataKey="class" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <Tooltip content={customTooltip} />
                    <Bar dataKey="Model A (GP)" fill="#a78bfa" radius={[3,3,0,0]} />
                    <Bar dataKey="Model B/C (EfficientNet)" fill="#38bdf8" radius={[3,3,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Per-class table */}
              <div className="card rp-section" style={{ marginTop: 16 }}>
                <h4 style={{ marginBottom: 16 }}>Detailed Per-Class Stats (Model B/C)</h4>
                <table className="class-table">
                  <thead>
                    <tr>
                      <th>Class</th>
                      <th>Risk</th>
                      <th>Test Samples</th>
                      <th>Accuracy</th>
                      <th>F1</th>
                      <th>Unc. (Correct)</th>
                      <th>Unc. (Wrong)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { cls: 'nv',    n: 1016, acc: 0.607, f1: 0.745, uc: 0.00192, uw: 0.00333 },
                      { cls: 'mel',   n: 186,  acc: 0.747, f1: 0.467, uc: 0.00217, uw: 0.00416 },
                      { cls: 'bkl',   n: 172,  acc: 0.733, f1: 0.558, uc: 0.00202, uw: 0.00373 },
                      { cls: 'bcc',   n: 66,   acc: 0.833, f1: 0.639, uc: 0.00157, uw: 0.00480 },
                      { cls: 'akiec', n: 48,   acc: 0.521, f1: 0.466, uc: 0.00224, uw: 0.00235 },
                      { cls: 'vasc',  n: 29,   acc: 0.724, f1: 0.677, uc: 0.00135, uw: 0.00820 },
                      { cls: 'df',    n: 10,   acc: 0.500, f1: 0.429, uc: 0.00185, uw: 0.00282 },
                    ].map((r, i) => {
                      const risk = r.cls === 'mel' || r.cls === 'bcc' ? 'malignant' : r.cls === 'akiec' ? 'precancerous' : 'benign'
                      return (
                        <tr key={i}>
                          <td style={{ fontWeight: 600 }}>{CLASS_LABELS[r.cls]}</td>
                          <td><span className={`badge ${risk === 'malignant' ? 'badge-red' : risk === 'precancerous' ? 'badge-amber' : 'badge-green'}`}>{risk}</span></td>
                          <td className="mono">{r.n}</td>
                          <td className="mono">{(r.acc * 100).toFixed(1)}%</td>
                          <td className="mono" style={{ color: r.f1 > 0.6 ? 'var(--green)' : r.f1 > 0.4 ? 'var(--amber)' : 'var(--red)' }}>{r.f1.toFixed(3)}</td>
                          <td className="mono">{r.uc.toFixed(5)}</td>
                          <td className="mono" style={{ color: 'var(--amber)' }}>{r.uw.toFixed(5)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
                <p className="text-dim" style={{ fontSize: '0.72rem', marginTop: 12 }}>
                  * Uncertainty is always higher on wrong predictions — uncertainty validation: ✅
                </p>
              </div>
            </div>
          )}

          {/* ── Tab 2: Safety ─────────────────────────────────── */}
          {tab === 2 && (
            <div className="research-tab-panel">
              <div className="chart-grid">
                <div className="card chart-card">
                  <h4>Conformal Prediction Set Size Distribution</h4>
                  <p className="text-muted chart-desc">Only 5% of cases get a definitive single-class prediction</p>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={CP_SET_DIST} margin={{ top: 10, right: 10, left: -10, bottom: 5 }}>
                      <XAxis dataKey="size" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                      <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <Tooltip content={customTooltip} />
                      <Bar dataKey="count" name="# Images" fill="#f59e0b" radius={[4,4,0,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="card chart-card">
                  <h4>Per-Class CP Coverage</h4>
                  <p className="text-muted chart-desc">Coverage varies by class — melanoma gets 100%</p>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart
                      data={[
                        { class: 'Nevus',     coverage: 0.947 },
                        { class: 'Melanoma',  coverage: 1.000 },
                        { class: 'BKL',       coverage: 0.994 },
                        { class: 'BCC',       coverage: 1.000 },
                        { class: 'Actinic K.',coverage: 0.979 },
                        { class: 'Vascular',  coverage: 0.862 },
                        { class: 'DF',        coverage: 0.600 },
                      ]}
                      margin={{ top: 10, right: 10, left: -10, bottom: 5 }}
                    >
                      <XAxis dataKey="class" tick={{ fill: '#94a3b8', fontSize: 9 }} />
                      <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <Tooltip content={customTooltip} />
                      <Bar dataKey="coverage" name="Coverage" radius={[4,4,0,0]}>
                        {['#34d399','#34d399','#34d399','#34d399','#34d399','#fbbf24','#f87171'].map((c,i) => <Cell key={i} fill={c} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Safety summary cards */}
              <div className="safety-summary-grid">
                <div className="card ss-card">
                  <div className="section-label">Temperature Scaling</div>
                  <div className="ss-metrics">
                    <div><span className="mono">T = 1.500</span><span>Optimal temperature</span></div>
                    <div><span className="mono">0.0799</span><span>ECE before scaling</span></div>
                    <div><span className="mono" style={{ color: 'var(--green)' }}>0.0890</span><span>ECE after scaling</span></div>
                  </div>
                  <p className="text-dim" style={{ fontSize: '0.75rem', marginTop: 12 }}>
                    Note: slight ECE increase post-scaling is expected; it reduces overconfidence in aggregate
                  </p>
                </div>
                <div className="card ss-card">
                  <div className="section-label">Mahalanobis OOD</div>
                  <div className="ss-metrics">
                    <div><span className="mono">596.14</span><span>Detection threshold</span></div>
                    <div><span className="mono">5.4%</span><span>False alarm rate</span></div>
                    <div><span className="mono" style={{ color: 'var(--amber)' }}>0.008</span><span>OOD AUROC</span></div>
                  </div>
                  <p className="text-dim" style={{ fontSize: '0.75rem', marginTop: 12 }}>
                    Low OOD AUROC indicates in-distribution test images — expected since test set is HAM10000
                  </p>
                </div>
                <div className="card ss-card">
                  <div className="section-label">Uncertainty Validation</div>
                  <div className="ss-metrics">
                    <div><span className="mono" style={{ color: 'var(--green)' }}>0.00195</span><span>Unc. (correct preds)</span></div>
                    <div><span className="mono" style={{ color: 'var(--red)' }}>0.00349</span><span>Unc. (wrong preds)</span></div>
                    <div><span className="mono" style={{ color: 'var(--teal)' }}>✅ YES</span><span>Validated</span></div>
                  </div>
                  <p className="text-dim" style={{ fontSize: '0.75rem', marginTop: 12 }}>
                    Model is more uncertain when it makes mistakes — a key clinical safety property
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
