const BASE_URL = 'http://localhost:8000'

export async function analyzeImage({ file, phase, age, sex, localization }) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('phase', phase)
  if (age)          formData.append('age', age)
  if (sex)          formData.append('sex', sex)
  if (localization) formData.append('localization', localization)

  const res = await fetch(`${BASE_URL}/api/analyze`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function getSummary() {
  const res = await fetch(`${BASE_URL}/api/results/summary`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function getHealth() {
  const res = await fetch(`${BASE_URL}/api/health`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}
