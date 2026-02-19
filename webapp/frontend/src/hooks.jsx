import { useState, useEffect } from 'react'

// In production, VITE_API_URL points to the HuggingFace Space (e.g. https://xxx.hf.space)
// In development, it falls back to '/api' which Vite proxies to localhost:8000
const API = (import.meta.env.VITE_API_URL || '') + '/api'
export const API_BASE = API

export function useFetch(path) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetch(`${API}${path}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((d) => {
        if (!cancelled) {
          setData(d)
          setLoading(false)
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e.message)
          setLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [path])

  return { data, loading, error }
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

export function Loader() {
  return (
    <div className="flex justify-center items-center py-20">
      <div className="animate-spin h-8 w-8 border-4 border-emerald-600 border-t-transparent rounded-full" />
    </div>
  )
}

export function ErrorBox({ message }) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
      Error: {message}
    </div>
  )
}
