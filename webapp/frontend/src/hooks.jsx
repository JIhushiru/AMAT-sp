import { useState, useEffect, useRef } from 'react'

// In production, VITE_API_URL points to the HuggingFace Space (e.g. https://xxx.hf.space)
// In development, it falls back to '/api' which Vite proxies to localhost:8000
const API = (import.meta.env.VITE_API_URL || '') + '/api'
export const API_BASE = API

export function useFetch(path) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [retrying, setRetrying] = useState(false)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    let cancelled = false
    let timer = null
    const start = Date.now()
    setLoading(true)
    setError(null)
    setRetrying(false)
    setElapsed(0)

    // Track elapsed time for long loads
    timer = setInterval(() => {
      if (!cancelled) setElapsed(Math.floor((Date.now() - start) / 1000))
    }, 1000)

    async function doFetch(attempt = 0) {
      try {
        const r = await fetch(`${API}${path}`)
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const d = await r.json()
        if (!cancelled) {
          setData(d)
          setLoading(false)
          setRetrying(false)
        }
      } catch (e) {
        if (cancelled) return
        // Retry up to 3 times for network/503 errors (HF cold start)
        if (attempt < 3 && (e.message.includes('Failed to fetch') || e.message.includes('503') || e.message.includes('502'))) {
          setRetrying(true)
          await new Promise((r) => setTimeout(r, 5000))
          if (!cancelled) doFetch(attempt + 1)
        } else {
          setError(e.message)
          setLoading(false)
        }
      }
    }

    doFetch()
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [path])

  return { data, loading, error, retrying, elapsed }
}

export function StatCard({ label, value, sub }) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <p className="text-sm text-gray-500">{label}</p>
      <p className="text-2xl font-bold text-emerald-700">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  )
}

export function Loader({ retrying, elapsed }) {
  return (
    <div className="flex flex-col justify-center items-center py-20 gap-3">
      <div className="animate-spin h-8 w-8 border-4 border-emerald-600 border-t-transparent rounded-full" />
      {elapsed > 5 && (
        <div className="text-center">
          <p className="text-sm text-gray-500">
            {retrying
              ? 'Server is waking up, retrying...'
              : 'Loading data...'}
          </p>
          <p className="text-xs text-gray-400 mt-1">
            {elapsed}s elapsed
            {elapsed > 15 && ' — HuggingFace Spaces can take up to a minute on cold start'}
          </p>
        </div>
      )}
    </div>
  )
}

export function downloadCSV(rows, filename = 'export.csv') {
  if (!rows || rows.length === 0) return
  const headers = Object.keys(rows[0])
  const csv = [
    headers.join(','),
    ...rows.map((r) =>
      headers.map((h) => {
        const val = r[h] ?? ''
        return typeof val === 'string' && val.includes(',') ? `"${val}"` : val
      }).join(',')
    ),
  ].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function ExportButton({ rows, filename, label = 'Export CSV' }) {
  if (!rows || rows.length === 0) return null
  return (
    <button
      onClick={() => downloadCSV(rows, filename)}
      className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm rounded border border-gray-300 transition"
    >
      {label}
    </button>
  )
}

export function ErrorBox({ message }) {
  const isServerDown = message.includes('Failed to fetch') || message.includes('503') || message.includes('502')
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
      <p className="font-medium">Error: {message}</p>
      {isServerDown && (
        <p className="text-sm mt-2">
          The API server may be sleeping. HuggingFace Spaces free tier spins down after inactivity.
          Try refreshing the page in about a minute.
        </p>
      )}
    </div>
  )
}
