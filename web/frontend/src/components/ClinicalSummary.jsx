import './ClinicalSummary.css'

/* ─────────────────────────────────────────────────────────
   Derives a simple "clinical status" from the raw API output.
   Status: 'ood' | 'urgent' | 'uncertain' | 'monitor' | 'safe'
   ───────────────────────────────────────────────────────── */
function deriveStatus(result) {
  const { is_ood, top_prediction, uncertainty_score, conformal_set } = result
  const prob      = top_prediction?.probability ?? 0
  const risk      = top_prediction?.risk ?? 'benign'
  const entropy   = uncertainty_score ?? 0
  const cpSize    = conformal_set ? conformal_set.length : 1

  if (is_ood)                                      return 'ood'
  if (risk === 'malignant' && prob > 0.55)         return 'urgent'
  if (risk === 'malignant' || risk === 'precancerous') {
    if (entropy > 1.2 || cpSize >= 4)             return 'uncertain'
    return 'monitor'
  }
  if (entropy > 1.5 || cpSize >= 5)               return 'uncertain'
  return 'safe'
}

const STATUS_CONFIG = {
  ood: {
    color:   'var(--red)',
    bgVar:   'var(--red-dim)',
    icon:    '🚫',
    title:   'Image Quality Warning',
    badge:   'Out of Distribution',
    summary: 'This image looks significantly different from the skin lesion images the AI was trained on. The prediction below may not be reliable.',
    action:  'Do not rely on this AI result. Please consult a dermatologist directly with the original image.',
    actionIcon: '🏥',
    actionLevel: 'danger',
  },
  urgent: {
    color:   'var(--red)',
    bgVar:   'var(--red-dim)',
    icon:    '⚠️',
    title:   'Potentially Malignant Finding',
    badge:   'Urgent Review Recommended',
    summary: 'The AI has identified features consistent with a potentially malignant skin lesion with moderate-to-high confidence.',
    action:  'Schedule an urgent dermatology consultation. Do not use this AI result as a definitive diagnosis.',
    actionIcon: '📅',
    actionLevel: 'danger',
  },
  monitor: {
    color:   'var(--amber)',
    bgVar:   'rgba(251,191,36,0.08)',
    icon:    '⚡',
    title:   'Precancerous or Malignant Possibility',
    badge:   'Specialist Review Recommended',
    summary: 'The AI has detected features that may indicate a pre-cancerous or malignant condition. The confidence is reasonable but not definitive.',
    action:  'Book a dermatology appointment. Bring this report. A biopsy or dermoscopy may be needed.',
    actionIcon: '📋',
    actionLevel: 'warning',
  },
  uncertain: {
    color:   'var(--purple)',
    bgVar:   'rgba(167,139,250,0.08)',
    icon:    '🔮',
    title:   'Ambiguous Finding — Multiple Possibilities',
    badge:   'Highly Uncertain',
    summary: 'The AI sees several plausible diagnoses and cannot distinguish confidently between them. This is expected for complex or overlapping lesion types.',
    action:  'Consult a dermatologist. Provide clinical history, patient age, and lesion evolution for a more accurate diagnosis.',
    actionIcon: '💬',
    actionLevel: 'warning',
  },
  safe: {
    color:   'var(--green)',
    bgVar:   'rgba(52,211,153,0.07)',
    icon:    '✅',
    title:   'Likely Benign Finding',
    badge:   'Low Risk',
    summary: 'The AI has identified features most consistent with a benign skin lesion, with reasonable confidence. This is not a guarantee.',
    action:  'Continue routine skin monitoring. If the lesion changes in size, shape, or color — seek medical advice.',
    actionIcon: '📅',
    actionLevel: 'safe',
  },
}

const CLASS_EMOJI = {
  nv: '🟢', mel: '🔴', bkl: '🟢', bcc: '🔴',
  akiec: '🟡', vasc: '🟢', df: '🟢',
}

function UncertaintyBar({ probability, entropy }) {
  // Convert probability + entropy into a simple 3-tier uncertainty label
  const pct = probability * 100
  let level, label, barColor
  if (pct >= 60 && entropy < 1.0) {
    level = 15; label = 'Low Uncertainty'; barColor = 'var(--green)'
  } else if (pct >= 40 || entropy < 1.5) {
    level = 50; label = 'Moderate Uncertainty'; barColor = 'var(--amber)'
  } else {
    level = 85; label = 'High Uncertainty'; barColor = 'var(--red)'
  }
  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar-header">
        <span className="conf-label">AI Uncertainty</span>
        <span className="conf-level" style={{ color: barColor }}>{label}</span>
      </div>
      <div className="conf-bar-track">
        <div className="conf-bar-fill" style={{ width: `${level}%`, background: barColor }} />
      </div>
      <div className="conf-bar-footer">
        <span>Low</span><span>Moderate</span><span>High</span>
      </div>
    </div>
  )
}

function DifferentialList({ conformalSet, topPrediction }) {
  const items = conformalSet || (topPrediction ? [topPrediction] : [])
  if (!items.length) return null
  return (
    <div className="diff-list">
      <div className="diff-title">
        {conformalSet
          ? `${items.length === 1 ? 'Definitive' : 'Possible'} Diagnos${items.length === 1 ? 'is' : 'es'}`
          : 'Top Diagnosis'
        }
      </div>
      <div className="diff-items">
        {items.map((item, i) => {
          const cls   = item.class || ''
          const name  = item.display || cls
          const risk  = item.risk || 'benign'
          const emoji = CLASS_EMOJI[cls] || '⬜'
          return (
            <div key={i} className={`diff-item risk-border-${risk}`}>
              <span className="diff-emoji">{emoji}</span>
              <div className="diff-info">
                <div className="diff-name">{name}</div>
                <div className={`diff-risk risk-${risk}`}>
                  {risk === 'malignant' ? 'Malignant' : risk === 'precancerous' ? 'Precancerous' : 'Benign'}
                </div>
              </div>
              {item.probability && (
                <span className="diff-prob mono">{(item.probability * 100).toFixed(0)}%</span>
              )}
              {conformalSet && !item.probability && (
                <span className="diff-prob-cp badge badge-gold" style={{ fontSize: '0.65rem' }}>
                  {i === 0 ? 'Most likely' : 'Possible'}
                </span>
              )}
            </div>
          )
        })}
      </div>
      {conformalSet && conformalSet.length > 1 && (
        <p className="diff-cp-note">
          🔒 Formal guarantee: the true diagnosis is in this list with <strong>95% probability</strong>
        </p>
      )}
    </div>
  )
}

export default function ClinicalSummary({ result }) {
  const status = deriveStatus(result)
  const cfg    = STATUS_CONFIG[status]
  const { top_prediction, conformal_set, phase, is_ood } = result
  const entropy = result.uncertainty_score ?? 0

  // For the differential list: prefer CP set (phase 3), else top prediction
  const differentialItems = conformal_set?.length
    ? conformal_set
    : top_prediction ? [top_prediction] : []

  return (
    <div className="clinical-summary" style={{ '--status-color': cfg.color, '--status-bg': cfg.bgVar }}>
      {/* Header banner */}
      <div className="cs-banner">
        <div className="cs-banner-left">
          <span className="cs-icon">{cfg.icon}</span>
          <div>
            <div className="cs-title">{cfg.title}</div>
            <span className="cs-badge" style={{ background: `${cfg.color}20`, color: cfg.color }}>
              {cfg.badge}
            </span>
          </div>
        </div>
        <div className="cs-phase-tag">
          Phase {phase} — {phase === 1 ? 'ML Baseline' : phase === 2 ? 'DL Standard' : 'Safe AI ✦'}
        </div>
      </div>

      {/* Summary row */}
      <div className="cs-body">
        {/* Left: What the AI found */}
        <div className="cs-left">
          <div className="cs-section-label">What the AI found</div>
          <p className="cs-summary">{cfg.summary}</p>

          {/* Differential diagnosis */}
          <DifferentialList
            conformalSet={conformal_set}
            topPrediction={top_prediction}
          />

          {/* Uncertainty bar */}
          {!is_ood && top_prediction && (
            <UncertaintyBar
              probability={top_prediction.probability}
              entropy={entropy}
            />
          )}
        </div>

        {/* Right: Action card */}
        <div className={`cs-action action-${cfg.actionLevel}`}>
          <div className="cs-action-icon">{cfg.actionIcon}</div>
          <div className="cs-action-label">Recommended Action</div>
          <p className="cs-action-text">{cfg.action}</p>

          <div className="cs-disclaimer">
            <strong>⚠️ Important:</strong> This AI tool is for research and educational purposes only.
            It does not replace professional medical diagnosis. Always consult a qualified dermatologist.
          </div>
        </div>
      </div>
    </div>
  )
}
