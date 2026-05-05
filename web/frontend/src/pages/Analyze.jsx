import { useState, useCallback, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { analyzeImage } from '../api/client'
import ResultsPanel from '../components/ResultsPanel'
import './Analyze.css'

const PHASES = [
  {
    id: 1,
    label: 'Phase 1',
    title: 'ML Baseline',
    subtitle: 'GP + IsoForest',
    desc: 'GLCM+LBP+HSV → PCA(50) → Gaussian Process',
    auroc: '0.862',
    color: 'var(--purple)',
  },
  {
    id: 2,
    label: 'Phase 2',
    title: 'DL Standard',
    subtitle: 'EfficientNet-B3',
    desc: 'MCDropout + Temperature Scaling + Mahalanobis OOD',
    auroc: '0.916',
    color: 'var(--teal)',
  },
  {
    id: 3,
    label: 'Phase 3',
    title: 'Safe AI ✦',
    subtitle: 'Hybrid Pipeline',
    desc: 'EfficientNet-B3 + GP OOD union + Conformal Prediction',
    auroc: '0.916 + 95% CP',
    color: 'var(--gold)',
  },
]

const LOCALIZATIONS = [
  'abdomen','acral','back','chest','ear','face','foot','genital',
  'hand','lower extremity','neck','scalp','trunk','upper extremity','unknown',
]

export default function Analyze() {
  const [file, setFile]               = useState(null)
  const [preview, setPreview]         = useState(null)
  const [selectedPhase, setPhase]     = useState(3)
  const [loading, setLoading]         = useState(false)
  const [result, setResult]           = useState(null)
  const [error, setError]             = useState(null)
  const [age, setAge]                 = useState('')
  const [sex, setSex]                 = useState('')
  const [localization, setLoc]        = useState('')
  const resultsRef                    = useRef(null)

  const onDrop = useCallback((accepted) => {
    if (!accepted.length) return
    const f = accepted[0]
    setFile(f)
    setResult(null)
    setError(null)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png'] },
    maxFiles: 1,
  })

  async function handleAnalyze() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await analyzeImage({
        file,
        phase: selectedPhase,
        age: age || undefined,
        sex: sex || undefined,
        localization: localization || undefined,
      })
      setResult(data)
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 100)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <main className="analyze-page">
      <div className="container">
        <div className="analyze-header fade-up">
          <div className="section-label">Analysis Studio</div>
          <h2>Lesion Analysis</h2>
          <p className="text-muted">Upload a dermoscopy image and run it through your selected pipeline phase.</p>
        </div>

        <div className="analyze-layout">
          {/* ── Left: Upload + Config ───────────────────── */}
          <div className="analyze-left">

            {/* Upload */}
            <div className="card upload-card fade-up">
              <h4 style={{ marginBottom: 16 }}>1. Upload Image</h4>
              <div
                {...getRootProps()}
                className={`dropzone ${isDragActive ? 'dropzone-active' : ''} ${file ? 'dropzone-filled' : ''}`}
              >
                <input {...getInputProps()} />
                {preview ? (
                  <div className="preview-wrap">
                    <img src={preview} alt="uploaded lesion" className="preview-img" />
                    <div className="preview-overlay">
                      <span>Click or drop to replace</span>
                    </div>
                  </div>
                ) : (
                  <div className="dropzone-placeholder">
                    <div className="drop-icon">🩻</div>
                    <p><strong>Drag & drop</strong> a dermoscopy image</p>
                    <p className="text-dim" style={{ fontSize: '0.8rem' }}>JPG, JPEG, PNG — HAM10000 format</p>
                  </div>
                )}
              </div>
              {file && (
                <div className="file-info">
                  <span className="text-muted" style={{ fontSize: '0.8rem' }}>📎 {file.name}</span>
                  <button className="btn btn-ghost" style={{ padding: '4px 10px', fontSize: '0.75rem' }} onClick={handleReset}>Clear</button>
                </div>
              )}
            </div>

            {/* Patient Metadata */}
            <div className="card meta-card fade-up fade-up-1">
              <h4 style={{ marginBottom: 16 }}>2. Patient Metadata <span className="text-dim" style={{ fontWeight: 400, fontSize: '0.8rem' }}>(optional)</span></h4>
              <div className="meta-grid">
                <div className="meta-field">
                  <label>Age</label>
                  <input
                    type="number"
                    placeholder="e.g. 45"
                    value={age}
                    onChange={e => setAge(e.target.value)}
                    className="meta-input"
                  />
                </div>
                <div className="meta-field">
                  <label>Sex</label>
                  <select value={sex} onChange={e => setSex(e.target.value)} className="meta-input">
                    <option value="">Select…</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="unknown">Unknown</option>
                  </select>
                </div>
                <div className="meta-field" style={{ gridColumn: '1 / -1' }}>
                  <label>Localization</label>
                  <select value={localization} onChange={e => setLoc(e.target.value)} className="meta-input">
                    <option value="">Select body site…</option>
                    {LOCALIZATIONS.map(l => (
                      <option key={l} value={l}>{l.charAt(0).toUpperCase() + l.slice(1)}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Phase Selector */}
            <div className="card phase-selector-card fade-up fade-up-2">
              <h4 style={{ marginBottom: 16 }}>3. Select Pipeline Phase</h4>
              <div className="phase-tabs">
                {PHASES.map(p => (
                  <button
                    key={p.id}
                    className={`phase-tab ${selectedPhase === p.id ? 'active' : ''}`}
                    style={{ '--tab-color': p.color }}
                    onClick={() => setPhase(p.id)}
                  >
                    <div className="phase-tab-label">{p.label}</div>
                    <div className="phase-tab-title">{p.title}</div>
                    <div className="phase-tab-sub">{p.subtitle}</div>
                    <div className="phase-tab-metric mono">{p.auroc}</div>
                  </button>
                ))}
              </div>
              <div className="phase-desc-box">
                <span className="text-muted" style={{ fontSize: '0.82rem' }}>
                  🔧 {PHASES.find(p => p.id === selectedPhase)?.desc}
                </span>
              </div>
            </div>

            {/* Analyze Button */}
            <button
              className={`btn btn-primary analyze-btn fade-up fade-up-3 ${loading ? 'loading' : ''}`}
              disabled={!file || loading}
              onClick={handleAnalyze}
            >
              {loading ? (
                <><div className="spinner" /> Analyzing…</>
              ) : (
                <>🔬 Run Analysis</>
              )}
            </button>

            {error && (
              <div className="ood-alert" style={{ marginTop: 12 }}>
                <span>⚠️</span>
                <div>
                  <strong style={{ color: 'var(--red)' }}>Error</strong>
                  <p className="text-muted" style={{ fontSize: '0.82rem', marginTop: 4 }}>{error}</p>
                  <p className="text-dim" style={{ fontSize: '0.75rem', marginTop: 4 }}>Make sure the backend is running: <code className="mono">python main.py</code></p>
                </div>
              </div>
            )}
          </div>

          {/* ── Right: Results ───────────────────────────── */}
          <div className="analyze-right" ref={resultsRef}>
            {!result && !loading && (
              <div className="results-empty card">
                <div className="empty-icon">📋</div>
                <h4>Results will appear here</h4>
                <p className="text-muted">Upload an image and click Run Analysis to begin.</p>
              </div>
            )}
            {loading && (
              <div className="card loading-card">
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: 32 }}>
                  <div className="skeleton" style={{ height: 24, width: '60%' }} />
                  <div className="skeleton" style={{ height: 120 }} />
                  <div className="skeleton" style={{ height: 16, width: '80%' }} />
                  <div className="skeleton" style={{ height: 16, width: '70%' }} />
                  <div className="skeleton" style={{ height: 16, width: '55%' }} />
                  <div className="skeleton" style={{ height: 200, marginTop: 8 }} />
                </div>
              </div>
            )}
            {result && !loading && <ResultsPanel result={result} />}
          </div>
        </div>
      </div>
    </main>
  )
}
