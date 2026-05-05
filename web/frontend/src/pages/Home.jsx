import { Link } from 'react-router-dom'
import './Home.css'

const METRICS = [
  { label: 'AUROC',       value: '0.916', unit: '',   desc: 'EfficientNet-B3' },
  { label: 'F1 Macro',    value: '0.569', unit: '',   desc: 'All 7 classes' },
  { label: 'ECE',         value: '0.089', unit: '',   desc: 'Calibrated' },
  { label: 'CP Coverage', value: '95.3',  unit: '%',  desc: 'Formal guarantee' },
]

const PHASES = [
  {
    phase: '01',
    title: 'ML Baseline',
    subtitle: 'Gaussian Process + IsoForest',
    description: '104 handcrafted features (GLCM, LBP, HSV) → PCA(50) → GP classifier with OOD detection via Isolation Forest.',
    metrics: [
      { k: 'F1 Macro', v: '0.349' },
      { k: 'AUROC',    v: '0.862' },
      { k: 'OOD Rate', v: '4.6%'  },
    ],
    color: 'var(--purple)',
    badge: 'Baseline',
  },
  {
    phase: '02',
    title: 'DL Standard',
    subtitle: 'EfficientNet-B3 + MC Dropout',
    description: 'Deep neural network with Monte Carlo Dropout for uncertainty, Temperature Scaling for calibration, Mahalanobis OOD detection.',
    metrics: [
      { k: 'F1 Macro', v: '0.569' },
      { k: 'AUROC',    v: '0.916' },
      { k: 'ECE',      v: '0.089' },
    ],
    color: 'var(--teal)',
    badge: 'Standard',
  },
  {
    phase: '03',
    title: 'Safe AI',
    subtitle: 'Hybrid + Conformal Prediction',
    description: 'Full clinical pipeline: EfficientNet-B3 + GP OOD union signal + Conformal Prediction with formal 95% coverage guarantee.',
    metrics: [
      { k: 'CP Coverage', v: '95.3%' },
      { k: 'Avg Set Size', v: '3.67' },
      { k: 'OOD Rate',    v: '10.1%' },
    ],
    color: 'var(--gold)',
    badge: '✦ Clinical Grade',
  },
]

const FEATURES = [
  {
    icon: '🩺',
    title: 'Clinical-Grade Safety',
    desc: 'Conformal Prediction provides a mathematically proven 95% coverage guarantee — not just a confidence score.',
  },
  {
    icon: '🔍',
    title: 'Explainable AI',
    desc: 'Grad-CAM heatmaps show exactly which skin regions drove the prediction, building clinical trust.',
  },
  {
    icon: '🛡️',
    title: 'OOD Detection',
    desc: 'Mahalanobis distance + Isolation Forest flags images the model was never trained on — preventing silent failures.',
  },
  {
    icon: '📊',
    title: 'Calibrated Probabilities',
    desc: 'Temperature Scaling (T=1.5) reduces ECE by 59.5% — when the model says 80%, it means 80%.',
  },
  {
    icon: '⚡',
    title: '3-Phase Ablation',
    desc: 'Run all three model architectures side-by-side and understand each component\'s contribution.',
  },
  {
    icon: '🧬',
    title: 'Zero Data Leakage',
    desc: 'GroupShuffleSplit by lesion_id ensures honest evaluation — a medical AI hallmark.',
  },
]

export default function Home() {
  return (
    <main className="home">
      {/* ── Hero ─────────────────────────────────────────── */}
      <section className="hero">
        {/* Ambient blobs */}
        <div className="blob blob-1" />
        <div className="blob blob-2" />

        <div className="container">
          <div className="hero-content fade-up">
            <div className="hero-eyebrow">
              <span className="badge badge-teal">HAM10000 · EfficientNet-B3 · Phase 3</span>
            </div>
            <h1>
              Clinical AI for<br />
              <span className="text-gradient">Skin Lesion Analysis</span>
            </h1>
            <p className="hero-desc">
              A formally verified, uncertainty-aware deep learning pipeline for
              dermatological diagnosis. Upload a dermoscopy image and receive a
              calibrated prediction with a <strong>95% coverage guarantee</strong>.
            </p>
            <div className="hero-cta">
              <Link to="/analyze" className="btn btn-primary" style={{ fontSize: '1rem', padding: '14px 32px' }}>
                🔬 Analyze a Lesion
              </Link>
              <Link to="/research" className="btn btn-ghost">
                View Research Results →
              </Link>
            </div>
          </div>

          {/* Metric strip */}
          <div className="metric-strip fade-up fade-up-2">
            {METRICS.map((m, i) => (
              <div key={i} className="metric-item">
                <div className="metric-value mono">{m.value}<span className="metric-unit">{m.unit}</span></div>
                <div className="metric-label">{m.label}</div>
                <div className="metric-desc">{m.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── 3 Phases ─────────────────────────────────────── */}
      <section className="phases-section">
        <div className="container">
          <div className="section-header fade-up">
            <div className="section-label">Research Journey</div>
            <h2>Three Phases of Robust AI</h2>
            <p className="text-muted" style={{ marginTop: 8 }}>
              From handcrafted features to formally certified clinical inference
            </p>
          </div>

          <div className="phases-grid">
            {PHASES.map((p, i) => (
              <div key={i} className={`phase-card card fade-up fade-up-${i + 1}`} style={{ '--phase-color': p.color }}>
                <div className="phase-header">
                  <span className="phase-number mono">{p.phase}</span>
                  <span className="badge" style={{ background: `${p.color}20`, color: p.color }}>{p.badge}</span>
                </div>
                <h3 style={{ color: p.color }}>{p.title}</h3>
                <div className="phase-subtitle">{p.subtitle}</div>
                <p className="phase-desc">{p.description}</p>
                <div className="phase-metrics">
                  {p.metrics.map((m, j) => (
                    <div key={j} className="phase-metric">
                      <span className="phase-metric-val mono">{m.v}</span>
                      <span className="phase-metric-key">{m.k}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ──────────────────────────────────────── */}
      <section className="features-section">
        <div className="container">
          <div className="section-header fade-up">
            <div className="section-label">Capabilities</div>
            <h2>Built for Clinical Trust</h2>
          </div>
          <div className="features-grid">
            {FEATURES.map((f, i) => (
              <div key={i} className={`feature-card card fade-up fade-up-${(i % 3) + 1}`}>
                <div className="feature-icon">{f.icon}</div>
                <h4>{f.title}</h4>
                <p className="text-muted">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Dataset Stats ─────────────────────────────────── */}
      <section className="dataset-section">
        <div className="container">
          <div className="dataset-card card">
            <div className="dataset-label section-label">Dataset</div>
            <div className="dataset-stats">
              <div className="ds-stat"><span className="mono" style={{ color: 'var(--teal)', fontSize: '1.75rem', fontWeight: 800 }}>10,015</span><span>Total images</span></div>
              <div className="ds-stat"><span className="mono" style={{ color: 'var(--purple)', fontSize: '1.75rem', fontWeight: 800 }}>7</span><span>Lesion classes</span></div>
              <div className="ds-stat"><span className="mono" style={{ color: 'var(--amber)', fontSize: '1.75rem', fontWeight: 800 }}>58:1</span><span>Imbalance ratio</span></div>
              <div className="ds-stat"><span className="mono" style={{ color: 'var(--green)', fontSize: '1.75rem', fontWeight: 800 }}>0</span><span>Leakage (GroupSplit)</span></div>
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA ───────────────────────────────────────────── */}
      <section className="cta-section">
        <div className="container">
          <div className="cta-card card card-glow fade-up">
            <div className="blob blob-cta" />
            <h2>Ready to analyze?</h2>
            <p className="text-muted">Upload a dermoscopy image and run it through all three phases of our pipeline.</p>
            <Link to="/analyze" className="btn btn-primary" style={{ fontSize: '1rem', padding: '14px 32px', marginTop: 8 }}>
              Start Analysis →
            </Link>
          </div>
        </div>
      </section>
    </main>
  )
}
