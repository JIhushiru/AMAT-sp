import { useState, useEffect, useRef, useMemo } from 'react'

// In production, VITE_API_URL points to the HuggingFace Space (e.g. https://xxx.hf.space)
// In development, it falls back to '/api' which Vite proxies to localhost:8000
const API = (import.meta.env.VITE_API_URL || '') + '/api'
export const API_BASE = API

export function useDarkMode() {
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') return false
    const stored = localStorage.getItem('theme')
    if (stored) return stored === 'dark'
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  useEffect(() => {
    const root = document.documentElement
    if (dark) {
      root.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      root.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }, [dark])

  // Listen for system theme changes when user hasn't explicitly toggled
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const handler = (e) => {
      const stored = localStorage.getItem('theme')
      if (!stored) setDark(e.matches)
    }
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [])

  return [dark, setDark]
}

export function useChartTheme() {
  const [dark, setDark] = useState(() =>
    typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
  )

  useEffect(() => {
    const el = document.documentElement
    const observer = new MutationObserver(() => {
      setDark(el.classList.contains('dark'))
    })
    observer.observe(el, { attributes: true, attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [])

  return useMemo(() => ({
    grid: dark ? '#374151' : '#e5e7eb',
    tooltip: {
      backgroundColor: dark ? '#1f2937' : '#fff',
      borderColor: dark ? '#374151' : '#e5e7eb',
      color: dark ? '#f3f4f6' : '#1f2937',
      fontSize: 12,
    },
    tick: { fontSize: 12, fill: dark ? '#9ca3af' : '#6b7280' },
    label: { fill: dark ? '#9ca3af' : '#6b7280', fontSize: 12 },
  }), [dark])
}

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
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">{label}</p>
      <p className="text-xl font-bold text-gray-800 dark:text-gray-100 mt-1">{value}</p>
      {sub && <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">{sub}</p>}
    </div>
  )
}

export function Loader({ retrying, elapsed }) {
  return (
    <div className="flex flex-col justify-center items-center py-24 gap-4">
      <div className="animate-spin h-8 w-8 border-[3px] border-gray-300 dark:border-gray-600 border-t-emerald-600 dark:border-t-emerald-400 rounded-full" />
      {elapsed > 5 && (
        <div className="text-center">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {retrying
              ? 'Server is waking up, retrying...'
              : 'Loading data...'}
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
            {elapsed}s elapsed
            {elapsed > 15 && ' \u2014 HuggingFace Spaces can take up to a minute on cold start'}
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
      className="px-3 py-1.5 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 text-gray-600 dark:text-gray-300 text-xs font-medium rounded-md border border-gray-300 dark:border-gray-600 transition"
    >
      {label}
    </button>
  )
}

export function SearchableSelect({ options = [], value, onChange, placeholder = 'Select...', className = '' }) {
  const [search, setSearch] = useState('')
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    function handleClick(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const filtered = options.filter((o) =>
    o.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div ref={ref} className={`relative ${className}`}>
      <input
        type="text"
        value={open ? search : value || ''}
        placeholder={placeholder}
        onFocus={() => { setOpen(true); setSearch('') }}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
      />
      {open && (
        <div className="absolute z-50 mt-1 w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-md shadow-lg max-h-48 overflow-y-auto">
          {value && (
            <div
              className="px-3 py-1.5 text-sm text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
              onClick={() => { onChange(''); setOpen(false); setSearch('') }}
            >
              Clear selection
            </div>
          )}
          {filtered.length === 0 && (
            <div className="px-3 py-1.5 text-sm text-gray-400">No matches</div>
          )}
          {filtered.map((o) => (
            <div
              key={o}
              className={`px-3 py-1.5 text-sm cursor-pointer hover:bg-emerald-50 dark:hover:bg-emerald-900/30 dark:text-gray-200 ${
                o === value ? 'bg-emerald-50 dark:bg-emerald-900/40 font-medium text-emerald-700 dark:text-emerald-400' : ''
              }`}
              onClick={() => { onChange(o); setOpen(false); setSearch('') }}
            >
              {o}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function CollapsibleSection({ title, defaultOpen = true, children, badge, actions }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-4 text-left"
      >
        <div className="flex items-center gap-2">
          <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200">{title}</h3>
          {badge && (
            <span className="text-[10px] font-medium bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 px-2 py-0.5 rounded-full uppercase tracking-wide">
              {badge}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {actions && <div onClick={(e) => e.stopPropagation()}>{actions}</div>}
          <svg
            className={`h-4 w-4 text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>
      {open && <div className="px-4 pb-4 border-t border-gray-100 dark:border-gray-700 pt-4">{children}</div>}
    </div>
  )
}

export function Accordion({ title, defaultOpen = true, children, badge }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border-b border-gray-100 dark:border-gray-700 last:border-b-0">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between py-2.5 text-left"
      >
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">{title}</h4>
          {badge && (
            <span className="text-[10px] font-medium bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 px-2 py-0.5 rounded-full">
              {badge}
            </span>
          )}
        </div>
        <svg
          className={`h-4 w-4 text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && <div className="pb-3">{children}</div>}
    </div>
  )
}

export function ErrorBox({ message }) {
  const isServerDown = message.includes('Failed to fetch') || message.includes('503') || message.includes('502')
  return (
    <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-400">
      <p className="font-medium text-sm">Error: {message}</p>
      {isServerDown && (
        <p className="text-xs mt-2">
          The API server may be sleeping. HuggingFace Spaces free tier spins down after inactivity.
          Try refreshing the page in about a minute.
        </p>
      )}
    </div>
  )
}
