import { useState } from 'react'
import './ResultsPanel.css'
import ClinicalSummary from './ClinicalSummary'

const CLASS_INFO = {
  nv:    { display: 'Melanocytic Nevus',    risk: 'benign',       emoji: '🟢' },
  mel:   { display: 'Melanoma',             risk: 'malignant',    emoji: '🔴' },
  bkl:   { display: 'Benign Keratosis',     risk: 'benign',       emoji: '🟢' },
  bcc:   { display: 'Basal Cell Carcinoma', risk: 'malignant',    emoji: '🔴' },
  akiec: { display: 'Actinic Keratosis',    risk: 'precancerous', emoji: '🟡' },
  vasc:  { display: 'Vascular Lesion',      risk: 'benign',       emoji: '🟢' },
  df:    { display: 'Dermatofibroma',       risk: 'benign',       emoji: '🟢' },
}

const PHASE_LABEL = { 1: 'ML Baseline', 2: 'DL Standard', 3: 'Hybrid Safe AI' }
const PHASE_COLOR = { 1: 'var(--purple)', 2: 'var(--teal)', 3: 'var(--gold)' }

function ProbBar({ label, risk, probability, isTop }) {
  const riskColor = risk === 'malignant' ? 'var(--risk-malignant)'
                  : risk === 'precancerous' ? 'var(--risk-precancer)'
                  : 'var(--risk-benign)'
  return (
    <div className={`prob-row ${isTop ? 'prob-row-top' : ''}`}>
      <div className="prob-bar-header">
        <div className="prob-label">
          <span>{CLASS_INFO[label]?.emoji}</span>
          <span>{CLASS_INFO[label]?.display || label}</span>
          {isTop && <span className="badge badge-teal" style={{ fontSize: '0.65rem' }}>Top</span>}
        </div>
        <span className="mono" style={{ fontSize: '0.9rem', fontWeight: 700, color: riskColor }}>
          {(probability * 100).toFixed(1)}%
        </span>
      </div>
      <div className="prob-bar-track" style={{ height: 8, borderRadius: 99, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
        <div
          className="prob-bar-fill"
          style={{
            width: `${probability * 100}%`,
            height: '100%',
            borderRadius: 99,
            background: isTop
              ? `linear-gradient(90deg, ${riskColor}, ${riskColor}88)`
              : 'rgba(255,255,255,0.15)',
            transition: 'width 1s cubic-bezier(0.16,1,0.3,1)',
          }}
        />
      </div>
    </div>
  )
}

export default function ResultsPanel({ result }) {
  const [showGradcam,  setShowGradcam]  = useState(false)
  const [showDetails,  setShowDetails]  = useState(false)
  const phase = result.phase
  const top   = result.top_prediction
  const info  = CLASS_INFO[top?.class] || {}

  return (
    <div className="results-panel fade-up">

      {/* ── Phase badge ─────────────────────────────────────── */}
      <div className="rp-phase-bar" style={{ '--pc': PHASE_COLOR[phase] }}>
        <div className="rp-phase-label">Phase {phase} — {PHASE_LABEL[phase]}</div>
        {phase === 3 && <span className="badge badge-gold">✦ Clinical Grade</span>}
        {phase === 2 && <span className="badge badge-teal">Standard</span>}
        {phase === 1 && <span className="badge badge-purple">Baseline</span>}
      </div>

      {/* ── 1. CLINICAL SUMMARY — plain English, always visible ─ */}
      <ClinicalSummary result={result} />

      {/* ── 2. Grad-CAM — after summary, prominent ──────────── */}
      {result.gradcam_b64 && (
        <div className="card rp-section">
          <div className="rp-gradcam-header">
            <div>
              <div className="section-label" style={{ marginBottom: 4 }}>Where is the AI looking?</div>
              <p className="text-muted" style={{ fontSize: '0.8rem' }}>
                Red/warm zones = regions that most influenced the prediction.
                Switch to see the Grad-CAM heatmap overlay.
              </p>
            </div>
            <button
              className="btn btn-ghost"
              style={{ padding: '6px 14px', fontSize: '0.8rem' }}
              onClick={() => setShowGradcam(v => !v)}
            >
              {showGradcam ? '← Original' : '🔥 Show Heatmap'}
            </button>
          </div>
          <div className="gradcam-wrap">
            <img
              src={showGradcam
                ? `data:image/png;base64,${result.gradcam_b64}`
                : `data:image/png;base64,${result.original_image_b64}`
              }
              alt={showGradcam ? 'gradcam heatmap' : 'original lesion'}
              className="gradcam-img"
            />
          </div>
        </div>
      )}

      {/* ── 3. ADVANCED DETAILS — collapsed by default ───────── */}
      <div className="rp-details-toggle">
        <button
          className="details-toggle-btn"
          onClick={() => setShowDetails(v => !v)}
        >
          <span>{showDetails ? '▲' : '▼'}</span>
          {showDetails ? 'Hide Technical Details' : 'Show Technical Details'}
          <span className="details-toggle-sub">
            Probability breakdown · Safety metrics · Model outputs
          </span>
        </button>
      </div>

      {showDetails && (
        <div className="rp-details-body">

          {/* Preprocessing */}
          {result.preprocessed_image_b64 && (
            <div className="card rp-section rp-preprocess">
              <div className="section-label" style={{ marginBottom: 12 }}>Preprocessing Preview</div>
              <div className="preprocess-grid">
                <div className="preprocess-img-wrap">
                  <img src={`data:image/png;base64,${result.original_image_b64}`} alt="original" />
                  <div className="preprocess-caption">Original</div>
                </div>
                <div className="preprocess-arrow">→</div>
                <div className="preprocess-img-wrap">
                  <img src={`data:image/png;base64,${result.preprocessed_image_b64}`} alt="preprocessed" />
                  <div className="preprocess-caption">224×224 · Normalized</div>
                </div>
              </div>
              <div className="preprocess-steps">
                {result.preprocessing_steps?.map((s, i) => (
                  <span key={i} className="badge badge-teal" style={{ fontSize: '0.68rem' }}>{s}</span>
                ))}
              </div>
            </div>
          )}

          {/* Full probability breakdown */}
          {result.predictions && (
            <div className="card rp-section">
              <div className="section-label" style={{ marginBottom: 16 }}>Full Probability Distribution (all 7 classes)</div>
              <div className="prob-list">
                {result.predictions.map((p, i) => (
                  <ProbBar key={p.class} label={p.class} risk={p.risk} probability={p.probability} isTop={i === 0} />
                ))}
              </div>
            </div>
          )}

          {/* Conformal Prediction Set technical view */}
          {phase === 3 && result.conformal_set && (
            <div className="card rp-section">
              <div className="rp-cp-header">
                <div>
                  <div className="section-label" style={{ marginBottom: 4 }}>Conformal Prediction Set (Technical)</div>
                  <p className="text-muted" style={{ fontSize: '0.8rem' }}>
                    Formal guarantee: at least one diagnosis in this set is correct with{' '}
                    <strong style={{ color: 'var(--gold)' }}>95% probability</strong>
                  </p>
                </div>
                <span className="badge badge-gold">
                  Coverage {result.conformal_coverage ? (result.conformal_coverage * 100).toFixed(1) : '95.3'}%
                </span>
              </div>
              <div className="cp-set">
                {result.conformal_set.map((c, i) => (
                  <div key={i} className={`cp-item ${c.risk}`}>
                    <span>{CLASS_INFO[c.class]?.emoji}</span>
                    <span className="cp-name">{c.display}</span>
                    <span className={`cp-risk risk-${c.risk}`}>{c.risk}</span>
                  </div>
                ))}
              </div>
              <div className="cp-footer">
                <span className="text-dim" style={{ fontSize: '0.75rem' }}>
                  Avg set size: <span className="mono">{result.avg_set_size?.toFixed(2) ?? '3.67'}</span> · q̂ = 0.984 · α = 0.05
                </span>
              </div>
            </div>
          )}

          {/* Safety & Uncertainty metrics */}
          <div className="card rp-section">
            <div className="section-label" style={{ marginBottom: 16 }}>Safety &amp; Uncertainty Metrics</div>
            <div className="safety-grid">
              {result.uncertainty_score !== undefined && (
                <div className="safety-metric">
                  <div className="sm-val mono">{result.uncertainty_score.toFixed(5)}</div>
                  <div className="sm-key">Entropy Uncertainty</div>
                  <div className="sm-desc">Shannon entropy of mean probs. Higher = more uncertain.</div>
                </div>
              )}
              {result.mc_uncertainty !== undefined && (
                <div className="safety-metric">
                  <div className="sm-val mono" style={{ color: result.mc_uncertainty > 0.01 ? 'var(--amber)' : 'var(--green)' }}>
                    {result.mc_uncertainty.toFixed(6)}
                  </div>
                  <div className="sm-key">MC Dropout Variance</div>
                  <div className="sm-desc">Variance across 10 stochastic forward passes.</div>
                </div>
              )}
              {result.evidential_vacuity !== undefined && (
                <div className="safety-metric">
                  <div className="sm-val mono" style={{ color: result.evidential_vacuity > 0.5 ? 'var(--amber)' : 'var(--green)' }}>
                    {result.evidential_vacuity.toFixed(4)}
                  </div>
                  <div className="sm-key">Evidential Vacuity</div>
                  <div className="sm-desc">Dirichlet: low evidence = rare/unseen input type.</div>
                </div>
              )}
              {result.mahalanobis_distance !== undefined && (
                <div className={`safety-metric ${result.is_ood ? 'sm-danger' : ''}`}>
                  <div className="sm-val mono" style={{ color: result.is_ood ? 'var(--red)' : 'var(--green)' }}>
                    {result.mahalanobis_distance.toFixed(1)}
                  </div>
                  <div className="sm-key">Mahalanobis Distance</div>
                  <div className="sm-desc">
                    Threshold: {result.mahalanobis_threshold}{' '}
                    {result.is_ood ? '— ⚠️ OOD!' : '— ✅ In-distribution'}
                  </div>
                </div>
              )}
              {result.model_metrics && (
                <>
                  <div className="safety-metric">
                    <div className="sm-val mono">{result.model_metrics.f1_macro?.toFixed(4)}</div>
                    <div className="sm-key">F1 Macro (test set)</div>
                    <div className="sm-desc">Model baseline on HAM10000 hold-out.</div>
                  </div>
                  <div className="safety-metric">
                    <div className="sm-val mono">{result.model_metrics.auroc?.toFixed(4)}</div>
                    <div className="sm-key">AUROC (test set)</div>
                    <div className="sm-desc">Discrimination ability across all 7 classes.</div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Phase 1 extras */}
          {phase === 1 && result.feature_description && (
            <div className="card rp-section">
              <div className="section-label" style={{ marginBottom: 8 }}>Feature Extraction</div>
              <p className="mono" style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{result.feature_description}</p>
              {result.n_features && (
                <p className="text-muted" style={{ fontSize: '0.8rem', marginTop: 8 }}>
                  {result.n_features} handcrafted features → PCA(50) → 98.2% variance retained
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
