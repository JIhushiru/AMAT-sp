import { lazy, Suspense } from 'react'
import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { useDarkMode, Loader } from './hooks'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const MapView = lazy(() => import('./pages/MapView'))
const HistoricalData = lazy(() => import('./pages/HistoricalData'))
const ModelResults = lazy(() => import('./pages/ModelResults'))
const SSPScenarios = lazy(() => import('./pages/SSPScenarios'))
const ProvinceDetail = lazy(() => import('./pages/ProvinceDetail'))
const Predict = lazy(() => import('./pages/Predict'))
const ProvincialAnalysis = lazy(() => import('./pages/ProvincialAnalysis'))

const NAV_ICONS = {
  '/': 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4',
  '/map': 'M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7',
  '/historical': 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z',
  '/provincial': 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
  '/models': 'M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z',
  '/ssp': 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6',
  '/predict': 'M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z',
}

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard' },
  { to: '/map', label: 'Map' },
  { to: '/historical', label: 'Historical' },
  { to: '/provincial', label: 'Provincial' },
  { to: '/models', label: 'Models' },
  { to: '/ssp', label: 'SSP Scenarios' },
  { to: '/predict', label: 'Predict' },
]

function Page({ children }) {
  const location = useLocation()
  return <div key={location.pathname} className="page-enter">{children}</div>
}

export default function App() {
  const [dark, setDark] = useDarkMode()

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex flex-col">
      <nav className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-6 md:gap-8">
          <div className="shrink-0">
            <h1 className="text-sm md:text-base font-bold tracking-tight text-gray-800 dark:text-gray-100 leading-tight">
              Banana Yield Prediction
            </h1>
            <p className="text-[10px] text-gray-400 dark:text-gray-500 font-medium tracking-wide uppercase hidden sm:block">
              UPLB &middot; Institute of Mathematical Sciences
            </p>
          </div>
          <div className="relative flex-1 min-w-0">
            <div className="flex gap-0.5 overflow-x-auto scrollbar-hide -mx-1 px-1">
              {NAV_ITEMS.map(({ to, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === '/'}
                  className={({ isActive }) =>
                    `flex items-center gap-1.5 px-2.5 md:px-3 py-1.5 rounded-md text-xs md:text-sm font-medium transition-colors whitespace-nowrap ${
                      isActive
                        ? 'bg-emerald-50 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-400'
                        : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700/50'
                    }`
                  }
                >
                  <svg className="h-3.5 w-3.5 shrink-0 hidden sm:block" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d={NAV_ICONS[to]} />
                  </svg>
                  {label}
                </NavLink>
              ))}
            </div>
            {/* Scroll fade hints */}
            <div className="absolute inset-y-0 right-0 w-6 bg-linear-to-l from-white dark:from-gray-800 pointer-events-none md:hidden" />
          </div>
          <button
            onClick={() => setDark(!dark)}
            className="text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 p-1.5 rounded transition shrink-0"
            aria-label="Toggle dark mode"
          >
            {dark ? (
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
            ) : (
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
            )}
          </button>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-3 md:px-6 py-6 md:py-8 flex-1 w-full">
        <Suspense fallback={<Loader />}>
          <Page>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/map" element={<MapView />} />
              <Route path="/historical" element={<HistoricalData />} />
              <Route path="/provincial" element={<ProvincialAnalysis />} />
              <Route path="/models" element={<ModelResults />} />
              <Route path="/ssp" element={<SSPScenarios />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/province/:name" element={<ProvinceDetail />} />
            </Routes>
          </Page>
        </Suspense>
      </main>

      <footer className="mt-auto">
        <div className="h-1 bg-linear-to-r from-emerald-500 via-teal-500 to-emerald-600" />
        <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
          <div className="max-w-7xl mx-auto px-4 py-6">
            <div className="grid sm:grid-cols-3 gap-4 text-[11px]">
              <div>
                <p className="font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide mb-1.5">Research</p>
                <p className="text-gray-500 dark:text-gray-400 leading-relaxed">
                  Geospatial Machine Learning for Predicting Banana Yield in the Philippines Under Climate Uncertainty
                </p>
              </div>
              <div>
                <p className="font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide mb-1.5">Institution</p>
                <p className="text-gray-500 dark:text-gray-400 leading-relaxed">
                  Institute of Mathematical Sciences<br />
                  College of Arts and Sciences<br />
                  University of the Philippines Los Ba&ntilde;os
                </p>
              </div>
              <div>
                <p className="font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide mb-1.5">Authors</p>
                <p className="text-gray-500 dark:text-gray-400 leading-relaxed">
                  <a href="https://jhraportfolio.vercel.app/" target="_blank" rel="noopener noreferrer" className="hover:text-emerald-600 dark:hover:text-emerald-400 transition font-medium">Arsolon, J.H.R.</a> (2025)<br />
                  Adviser: Dr. Destiny SM. Lutero<br />
                  Co-adviser: Dr. Mark Lexter De Lara
                </p>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
