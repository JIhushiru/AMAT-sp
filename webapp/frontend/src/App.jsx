import { Routes, Route, NavLink } from 'react-router-dom'
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
  { to: '/historical', label: 'Historical Data' },
  { to: '/models', label: 'Model Results' },
  { to: '/ssp', label: 'SSP Scenarios' },
  { to: '/predict', label: 'Predict' },
]

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-emerald-800 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-8">
          <h1 className="text-lg font-bold tracking-tight whitespace-nowrap">
            PH Banana Yield
          </h1>
          <div className="flex gap-1 overflow-x-auto">
            {NAV_ITEMS.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded text-sm font-medium transition-colors whitespace-nowrap ${
                    isActive
                      ? 'bg-emerald-600 text-white'
                      : 'text-emerald-100 hover:bg-emerald-700'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-4 py-6">
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
    </div>
  )
}
