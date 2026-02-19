import { useFetch, StatCard, Loader, ErrorBox } from '../hooks'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Legend, Cell,
} from 'recharts'

export default function Dashboard() {
  const { data, loading, error, retrying, elapsed } = useFetch('/dashboard')

  if (loading) return <Loader retrying={retrying} elapsed={elapsed} />
  if (error) return <ErrorBox message={error} />

  const h = data.historical
  const trendData = Object.entries(h.national_trend).map(([year, val]) => ({
    year: +year, yield: val,
  }))

  if (data.ssp245_national_trend) {
    Object.entries(data.ssp245_national_trend).forEach(([year, val]) => {
      const existing = trendData.find((d) => d.year === +year)
      if (existing) existing.ssp245 = val
      else trendData.push({ year: +year, ssp245: val })
    })
  }
  if (data.ssp585_national_trend) {
    Object.entries(data.ssp585_national_trend).forEach(([year, val]) => {
      const existing = trendData.find((d) => d.year === +year)
      if (existing) existing.ssp585 = val
      else trendData.push({ year: +year, ssp585: val })
    })
  }
  trendData.sort((a, b) => a.year - b.year)

  const topProvinces = Object.entries(h.top_provinces).map(([name, val]) => ({
    name, yield: val,
  }))
  const bottomProvinces = Object.entries(h.bottom_provinces).map(([name, val]) => ({
    name, yield: val,
  }))

  const COLORS_TOP = ['#059669', '#10b981', '#34d399', '#6ee7b7', '#a7f3d0']
  const COLORS_BOT = ['#dc2626', '#ef4444', '#f87171', '#fca5a5', '#fecaca']

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Dashboard</h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="National Avg Yield" value={`${h.national_avg} t/ha`} />
        <StatCard label="Provinces" value={h.total_provinces} />
        <StatCard
          label="Year Range"
          value={`${h.year_range[0]}-${h.year_range[1]}`}
        />
        <StatCard
          label="SSP Data"
          value={data.ssp245_available || data.ssp585_available ? 'Available' : 'Not Yet'}
          sub={
            data.ssp245_available
              ? 'SSP2-4.5 & SSP5-8.5'
              : 'Run SSP pipeline to generate'
          }
        />
      </div>

      {/* National Yield Trend */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold mb-3 text-gray-700 dark:text-gray-200">
          National Yield Trend (2010-2024
          {data.ssp245_available ? ' + SSP Projections 2025-2034' : ''})
        </h3>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis domain={['auto', 'auto']} label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone" dataKey="yield" stroke="#059669"
              strokeWidth={2} name="Historical" dot={{ r: 3 }}
            />
            {data.ssp245_available && (
              <Line
                type="monotone" dataKey="ssp245" stroke="#2563eb"
                strokeWidth={2} strokeDasharray="5 5" name="SSP2-4.5"
                dot={{ r: 3 }}
              />
            )}
            {data.ssp585_available && (
              <Line
                type="monotone" dataKey="ssp585" stroke="#dc2626"
                strokeWidth={2} strokeDasharray="5 5" name="SSP5-8.5"
                dot={{ r: 3 }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Top & Bottom Provinces */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-3 text-gray-700 dark:text-gray-200">
            Top 5 Provinces by Yield
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={topProvinces} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" label={{ value: 't/ha', position: 'insideBottom', offset: -5 }} />
              <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="yield" name="Avg Yield">
                {topProvinces.map((_, i) => (
                  <Cell key={i} fill={COLORS_TOP[i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-3 text-gray-700 dark:text-gray-200">
            Bottom 5 Provinces by Yield
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={bottomProvinces} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" label={{ value: 't/ha', position: 'insideBottom', offset: -5 }} />
              <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="yield" name="Avg Yield">
                {bottomProvinces.map((_, i) => (
                  <Cell key={i} fill={COLORS_BOT[i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  )
}
