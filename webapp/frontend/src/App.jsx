import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { useDarkMode } from './hooks'
import Dashboard from './pages/Dashboard'
import MapView from './pages/MapView'
import HistoricalData from './pages/HistoricalData'
import ModelResults from './pages/ModelResults'
import SSPScenarios from './pages/SSPScenarios'
import ProvinceDetail from './pages/ProvinceDetail'
import Predict from './pages/Predict'
import ProvincialAnalysis from './pages/ProvincialAnalysis'

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
          <div className="flex gap-0.5 overflow-x-auto scrollbar-hide -mx-1 px-1 flex-1">
            {NAV_ITEMS.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `px-2.5 md:px-3 py-1.5 rounded-md text-xs md:text-sm font-medium transition-colors whitespace-nowrap ${
                    isActive
                      ? 'bg-emerald-50 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-400'
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700/50'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
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
      </main>

      <footer className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 mt-auto">
        <div className="max-w-7xl mx-auto px-4 py-5 text-center space-y-1">
          <p className="text-[11px] font-semibold text-gray-500 dark:text-gray-400 tracking-wide uppercase">
            Geospatial Machine Learning for Predicting Banana Yield in the Philippines Under Climate Uncertainty
          </p>
          <p className="text-[11px] text-gray-400 dark:text-gray-500">
            Institute of Mathematical Sciences &middot; College of Arts and Sciences &middot; University of the Philippines Los Ba&ntilde;os
          </p>
          <p className="text-[11px] text-gray-400 dark:text-gray-500">
            <a href="https://jhraportfolio.vercel.app/" target="_blank" rel="noopener noreferrer" className="hover:text-emerald-600 dark:hover:text-emerald-400 transition">Arsolon, J.H.R.</a> (2025) &middot; Adviser: Dr. Destiny SM. Lutero &middot; Co-adviser: Dr. Mark Lexter De Lara
          </p>
        </div>
      </footer>
    </div>
  )
}
