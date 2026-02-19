import { Routes, Route, NavLink } from 'react-router-dom'
import { useDarkMode } from './hooks'
import Dashboard from './pages/Dashboard'
import MapView from './pages/MapView'
import HistoricalData from './pages/HistoricalData'
import ModelResults from './pages/ModelResults'
import SSPScenarios from './pages/SSPScenarios'
import ProvinceDetail from './pages/ProvinceDetail'
import Predict from './pages/Predict'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard' },
  { to: '/map', label: 'Map' },
  { to: '/historical', label: 'Historical' },
  { to: '/models', label: 'Models' },
  { to: '/ssp', label: 'SSP' },
  { to: '/predict', label: 'Predict' },
]

export default function App() {
  const [dark, setDark] = useDarkMode()

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex flex-col">
      <nav className="bg-emerald-800 dark:bg-gray-800 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-4 md:gap-8">
          <h1 className="text-base md:text-lg font-bold tracking-tight whitespace-nowrap">
            PH Banana Yield
          </h1>
          <div className="flex gap-1 overflow-x-auto scrollbar-hide -mx-1 px-1 flex-1">
            {NAV_ITEMS.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `px-2.5 md:px-3 py-1.5 rounded text-xs md:text-sm font-medium transition-colors whitespace-nowrap ${
                    isActive
                      ? 'bg-emerald-600 dark:bg-emerald-700 text-white'
                      : 'text-emerald-100 dark:text-gray-300 hover:bg-emerald-700 dark:hover:bg-gray-700'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </div>
          <button
            onClick={() => setDark(!dark)}
            className="text-emerald-100 dark:text-gray-300 hover:text-white p-1.5 rounded transition shrink-0"
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

      <main className="max-w-7xl mx-auto px-3 md:px-4 py-4 md:py-6 flex-1 w-full">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/map" element={<MapView />} />
          <Route path="/historical" element={<HistoricalData />} />
          <Route path="/models" element={<ModelResults />} />
          <Route path="/ssp" element={<SSPScenarios />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/province/:name" element={<ProvinceDetail />} />
        </Routes>
      </main>

      <footer className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 mt-auto">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
          <span>Philippine Banana Yield Prediction</span>
          <a
            href="https://jhraportfolio.vercel.app/"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-emerald-600 dark:hover:text-emerald-400 transition"
          >
            Portfolio &rarr;
          </a>
        </div>
      </footer>
    </div>
  )
}
