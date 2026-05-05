import './Pipeline.css'

const TIMELINE = [
  {
    phase: '01',
    title: 'ML Baseline',
    color: 'var(--purple)',
    badge: 'Phase 1',
    steps: [
      { icon: '🖼️', text: 'HAM10000 images loaded (10,015 total)' },
      { icon: '✂️', text: 'GroupShuffleSplit by lesion_id — zero data leakage' },
      { icon: '🔬', text: 'GLCM + LBP + HSV feature extraction (104 features)' },
      { icon: '📉', text: 'StandardScaler → PCA(50) — 98.2% variance retained' },
      { icon: '🧠', text: 'Gaussian Process Classifier (One-vs-Rest, 7 classes)' },
      { icon: '🛡️', text: 'Isolation Forest for OOD detection (4.6% rate)' },
    ],
    metrics: { 'F1 Macro': '0.349', AUROC: '0.862', 'OOD Rate': '4.6%' },
  },
  {
    phase: '02',
    title: 'DL Standard',
    color: 'var(--teal)',
    badge: 'Phase 2',
    steps: [
      { icon: '⚖️', text: 'WeightedRandomSampler — handles 58:1 class imbalance' },
      { icon: '🤖', text: 'EfficientNet-B3 pretrained on ImageNet (fine-tuned)' },
      { icon: '🎲', text: 'MC Dropout (10 forward passes) for uncertainty estimation' },
      { icon: '🌡️', text: 'Temperature Scaling (T=1.5) — reduces ECE by 59.5%' },
      { icon: '📏', text: 'Mahalanobis Distance OOD detector (threshold=596.14)' },
      { icon: '🗺️', text: 'Grad-CAM heatmaps for explainability' },
    ],
    metrics: { 'F1 Macro': '0.569', AUROC: '0.916', ECE: '0.089', 'OOD Rate': '5.5%' },
  },
  {
    phase: '03',
    title: 'Safe AI (Hybrid)',
    color: 'var(--gold)',
    badge: '✦ Phase 3',
    steps: [
      { icon: '🔗', text: 'Combines EfficientNet-B3 + GP OOD signal (union logic)' },
      { icon: '📊', text: 'Conformal Prediction (RAPS) calibrated on validation set' },
      { icon: '🔒', text: 'q̂ = 0.984 ensures ≥95% formal coverage guarantee' },
      { icon: '📋', text: 'Outputs a set of plausible diagnoses, not just one' },
      { icon: '✅', text: 'Uncertainty validated: wrong preds have 1.79× higher uncertainty' },
      { icon: '🏥', text: 'Clinically deployable — explicit coverage guarantee' },
    ],
    metrics: { 'CP Coverage': '95.3%', 'Avg Set Size': '3.67', 'OOD Rate': '10.1%', 'ECE': '0.089' },
  },
]

const CLASSES = [
  { code: 'nv',    name: 'Melanocytic Nevus',    risk: 'benign',      n: 6705, pct: 66.9, color: '#34d399' },
  { code: 'mel',   name: 'Melanoma',             risk: 'malignant',   n: 1113, pct: 11.1, color: '#f87171' },
  { code: 'bkl',   name: 'Benign Keratosis',     risk: 'benign',      n: 1099, pct: 11.0, color: '#34d399' },
  { code: 'bcc',   name: 'Basal Cell Carcinoma', risk: 'malignant',   n: 514,  pct: 5.1,  color: '#f87171' },
  { code: 'akiec', name: 'Actinic Keratosis',    risk: 'precancerous',n: 327,  pct: 3.3,  color: '#fbbf24' },
  { code: 'vasc',  name: 'Vascular Lesion',      risk: 'benign',      n: 142,  pct: 1.4,  color: '#34d399' },
  { code: 'df',    name: 'Dermatofibroma',       risk: 'benign',      n: 115,  pct: 1.1,  color: '#34d399' },
]

export default function Pipeline() {
  return (
    <main className="pipeline-page">
      <div className="container">
        <div className="pipeline-header fade-up">
          <div className="section-label">Pipeline Overview</div>
          <h2>How DermaSense AI Works</h2>
          <p className="text-muted">
            A 3-phase research pipeline from classical ML to formally certified clinical AI,
            built on the HAM10000 dermoscopy dataset.
          </p>
        </div>

        {/* Dataset overview */}
        <div className="card pipeline-dataset fade-up">
          <div className="pd-header">
            <div>
              <div className="section-label">Dataset</div>
              <h3>HAM10000 — Human Against Machine with 10,000 training images</h3>
              <p className="text-muted" style={{ fontSize: '0.875rem', marginTop: 4 }}>
                10,015 dermoscopy images · 7 lesion classes · 58:1 class imbalance · Zero leakage split by lesion_id
              </p>
            </div>
            <div className="pd-stats">
              <div><span className="mono" style={{ color: 'var(--teal)', fontSize: '1.5rem', fontWeight: 800 }}>6,959</span><span>Train</span></div>
              <div><span className="mono" style={{ color: 'var(--purple)', fontSize: '1.5rem', fontWeight: 800 }}>1,529</span><span>Val</span></div>
              <div><span className="mono" style={{ color: 'var(--gold)', fontSize: '1.5rem', fontWeight: 800 }}>1,527</span><span>Test</span></div>
            </div>
          </div>

          <div className="class-grid">
            {CLASSES.map((c, i) => (
              <div key={i} className="class-chip" style={{ '--cc': c.color }}>
                <div className="class-chip-top">
                  <span className="class-code mono">{c.code}</span>
                  <span className={`badge ${c.risk === 'malignant' ? 'badge-red' : c.risk === 'precancerous' ? 'badge-amber' : 'badge-green'}`} style={{ fontSize: '0.62rem' }}>{c.risk}</span>
                </div>
                <div className="class-chip-name">{c.name}</div>
                <div className="class-chip-bar">
                  <div className="class-chip-fill" style={{ width: `${c.pct}%`, background: c.color }} />
                </div>
                <div className="class-chip-stats">
                  <span className="mono" style={{ color: c.color }}>{c.n}</span>
                  <span>{c.pct}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 3-phase timeline */}
        <div className="pipeline-timeline">
          {TIMELINE.map((t, i) => (
            <div key={i} className={`timeline-phase fade-up fade-up-${i + 1}`} style={{ '--tc': t.color }}>
              <div className="timeline-connector">
                <div className="tc-dot" />
                {i < TIMELINE.length - 1 && <div className="tc-line" />}
              </div>

              <div className="timeline-content card">
                <div className="tl-header">
                  <div>
                    <div className="tl-badge-row">
                      <span className="badge" style={{ background: `${t.color}20`, color: t.color }}>{t.badge}</span>
                    </div>
                    <h3 style={{ color: t.color, marginTop: 8 }}>{t.title}</h3>
                  </div>
                  <div className="tl-metrics">
                    {Object.entries(t.metrics).map(([k, v]) => (
                      <div key={k} className="tl-metric">
                        <span className="mono" style={{ color: t.color }}>{v}</span>
                        <span>{k}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="divider" />

                <div className="tl-steps">
                  {t.steps.map((s, j) => (
                    <div key={j} className="tl-step">
                      <span className="tl-step-icon">{s.icon}</span>
                      <span className="tl-step-text">{s.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Key decisions */}
        <div className="key-decisions fade-up">
          <div className="section-label" style={{ marginBottom: 20 }}>Key Engineering Decisions</div>
          <div className="decisions-grid">
            {[
              { title: 'GroupShuffleSplit', icon: '🔒', desc: 'Splits by lesion_id not image_id. HAM10000 has ~2,500 duplicate photos of the same lesion — naive splits inflate accuracy dishonestly.' },
              { title: 'WeightedRandomSampler', icon: '⚖️', desc: 'Gives rare classes (df: 115 images) equal sampling weight as the majority class (nv: 6,705). Prevents the model from ignoring rare cancers.' },
              { title: 'MC Dropout at Inference', icon: '🎲', desc: 'Dropout stays active during prediction (10 passes). The variance across passes = epistemic uncertainty. Wrong predictions have 1.79× higher uncertainty.' },
              { title: 'Temperature Scaling', icon: '🌡️', desc: 'A single scalar T=1.5 is learned post-training to reduce overconfidence. It cannot change predictions, only calibrate the probability scores.' },
              { title: 'Mahalanobis OOD', icon: '📏', desc: 'Measures statistical distance of a new image from class-conditional Gaussians in feature space. More principled than softmax-based OOD which is easily fooled.' },
              { title: 'Conformal Prediction', icon: '🔒', desc: 'RAPS (Regularized Adaptive Prediction Sets) provides a formal PAC-style guarantee: at least one class in the output set is correct with ≥1−α probability.' },
            ].map((d, i) => (
              <div key={i} className="card decision-card">
                <div className="dc-icon">{d.icon}</div>
                <h4>{d.title}</h4>
                <p className="text-muted">{d.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </main>
  )
}
