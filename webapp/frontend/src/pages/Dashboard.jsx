import { useFetch, StatCard, Loader, ErrorBox, useChartTheme } from '../hooks'
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

  const chart = useChartTheme()

  // Compute yield trend: compare last 3 years avg vs first 3 years avg
  const histOnly = trendData.filter((d) => d.yield != null)
  const yieldTrend = histOnly.length >= 6
    ? (() => {
        const early = histOnly.slice(0, 3).reduce((s, d) => s + d.yield, 0) / 3
        const late = histOnly.slice(-3).reduce((s, d) => s + d.yield, 0) / 3
        return ((late - early) / early) * 100
      })()
    : null

  const COLORS_TOP = ['#059669', '#10b981', '#34d399', '#6ee7b7', '#a7f3d0']
  const COLORS_BOT = ['#dc2626', '#ef4444', '#f87171', '#fca5a5', '#fecaca']

  return (
    <div className="space-y-6">
      <div className="bg-linear-to-r from-emerald-600 to-teal-600 dark:from-emerald-800 dark:to-teal-800 rounded-xl p-6 md:p-8 text-white">
        <p className="text-emerald-100 text-xs font-medium uppercase tracking-wider mb-2">
          UPLB &middot; Institute of Mathematical Sciences
        </p>
        <h2 className="text-xl md:text-2xl font-bold leading-snug">
          Geospatial Machine Learning for Predicting Banana Yield in the Philippines Under Climate Uncertainty
        </h2>
        <p className="text-emerald-100 text-sm mt-3 max-w-3xl leading-relaxed">
          Analyzing yield across {h.total_provinces} provinces using historical production
          data (2010&ndash;2024) from PSA and 17 climate variables from CRU-TS,
          TerraClimate, and NASA NEX-GDDP-CMIP6. Future projections
          (2025&ndash;2034) generated under SSP2-4.5 and SSP5-8.5 scenarios
          using a five-model CMIP6 ensemble.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="National Avg Yield" value={`${h.national_avg} t/ha`} trend={yieldTrend} sub="vs. early period" />
        <StatCard label="Provinces" value={h.total_provinces} />
        <StatCard
          label="Year Range"
          value={`${h.year_range[0]}\u2013${h.year_range[1]}`}
        />
        <StatCard
          label="SSP Projections"
          value={data.ssp245_available || data.ssp585_available ? 'Available' : 'Pending'}
          sub={
            data.ssp245_available
              ? 'SSP2-4.5 & SSP5-8.5'
              : 'Run pipeline to generate'
          }
        />
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
        <h3 className="text-base font-semibold mb-1 text-gray-800 dark:text-gray-200">
          National Yield Trend
        </h3>
        <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
          Historical (2010&ndash;2024){data.ssp245_available ? ' and SSP projections (2025\u20132034)' : ''}
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
            <XAxis dataKey="year" tick={{ fontSize: 12 }} />
            <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12 }} label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} />
            <Tooltip contentStyle={chart.tooltip} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
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

      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
          <h3 className="text-base font-semibold mb-1 text-gray-800 dark:text-gray-200">
            Top 5 Provinces by Yield
          </h3>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">Average yield (2010&ndash;2024)</p>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={topProvinces} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
              <XAxis type="number" tick={{ fontSize: 11 }} label={{ value: 't/ha', position: 'insideBottom', offset: -5, style: { fontSize: 11 } }} />
              <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={chart.tooltip} />
              <Bar dataKey="yield" name="Avg Yield" radius={[0, 4, 4, 0]}>
                {topProvinces.map((_, i) => (
                  <Cell key={i} fill={COLORS_TOP[i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
          <h3 className="text-base font-semibold mb-1 text-gray-800 dark:text-gray-200">
            Bottom 5 Provinces by Yield
          </h3>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">Average yield (2010&ndash;2024)</p>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={bottomProvinces} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
              <XAxis type="number" tick={{ fontSize: 11 }} label={{ value: 't/ha', position: 'insideBottom', offset: -5, style: { fontSize: 11 } }} />
              <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={chart.tooltip} />
              <Bar dataKey="yield" name="Avg Yield" radius={[0, 4, 4, 0]}>
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
