import { useState } from 'react'
import { useFetch, Loader, ErrorBox, CollapsibleSection } from '../hooks'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'

// Maximally distinct colors
const TOP_COLORS = ['#1e3a5f', '#059669', '#d97706', '#7c3aed', '#e11d48']
const BOT_COLORS = ['#dc2626', '#2563eb', '#16a34a', '#a855f7', '#ea580c']

function buildChartData(provinces, historical, ssp245, ssp585, showSsp245, showSsp585) {
  const yearMap = {}

  provinces.forEach((prov) => {
    const hist = historical?.[prov] || []
    hist.forEach(({ year, yield: y }) => {
      if (!yearMap[year]) yearMap[year] = { year }
      yearMap[year][prov] = y
    })

    if (showSsp245) {
      const s245 = ssp245?.[prov] || []
      s245.forEach(({ year, yield: y }) => {
        if (!yearMap[year]) yearMap[year] = { year }
        yearMap[year][`${prov}_ssp245`] = y
      })
    }

    if (showSsp585) {
      const s585 = ssp585?.[prov] || []
      s585.forEach(({ year, yield: y }) => {
        if (!yearMap[year]) yearMap[year] = { year }
        yearMap[year][`${prov}_ssp585`] = y
      })
    }
  })

  return Object.values(yearMap).sort((a, b) => a.year - b.year)
}

function ProvinceChart({ title, subtitle, provinces, colors, chartData, showSsp245, showSsp585 }) {
  const hasProjections = showSsp245 || showSsp585
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
      <h3 className="text-base font-semibold mb-1 text-gray-800 dark:text-gray-200">{title}</h3>
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">{subtitle}</p>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="year" tick={{ fontSize: 12 }} />
          <YAxis
            domain={['auto', 'auto']}
            tick={{ fontSize: 12 }}
            label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
          />
          <Tooltip
            contentStyle={{ fontSize: 12 }}
            formatter={(value, name) => [
              value != null ? `${value.toFixed(2)} t/ha` : '\u2013',
              name.replace('_ssp245', ' (SSP2-4.5)').replace('_ssp585', ' (SSP5-8.5)'),
            ]}
          />
          <Legend
            wrapperStyle={{ fontSize: 11 }}
            formatter={(value) =>
              value.replace('_ssp245', ' (SSP2-4.5)').replace('_ssp585', ' (SSP5-8.5)')
            }
          />
          {hasProjections && (
            <ReferenceLine x={2024} stroke="#9ca3af" strokeDasharray="4 4" strokeWidth={1} />
          )}
          {/* Historical lines */}
          {provinces.map((prov, i) => (
            <Line
              key={prov}
              type="monotone"
              dataKey={prov}
              stroke={colors[i]}
              strokeWidth={2.5}
              dot={{ r: 3 }}
              name={prov}
              connectNulls={false}
            />
          ))}
          {/* SSP2-4.5 lines */}
          {showSsp245 && provinces.map((prov, i) => (
            <Line
              key={`${prov}_ssp245`}
              type="monotone"
              dataKey={`${prov}_ssp245`}
              stroke={colors[i]}
              strokeWidth={2}
              strokeDasharray="8 4"
              dot={{ r: 2.5 }}
              name={`${prov}_ssp245`}
              connectNulls={false}
            />
          ))}
          {/* SSP5-8.5 lines */}
          {showSsp585 && provinces.map((prov, i) => (
            <Line
              key={`${prov}_ssp585`}
              type="monotone"
              dataKey={`${prov}_ssp585`}
              stroke={colors[i]}
              strokeWidth={2}
              strokeDasharray="3 3"
              dot={{ r: 2.5 }}
              name={`${prov}_ssp585`}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {hasProjections && (
        <div className="mt-3 flex flex-wrap gap-4 text-xs text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-6 h-0 border-t-[2.5px] border-gray-400" /> Historical
          </span>
          {showSsp245 && (
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-6 h-0 border-t-2 border-dashed border-gray-400" /> SSP2-4.5
            </span>
          )}
          {showSsp585 && (
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-6 h-0 border-t-2 border-dotted border-gray-400" /> SSP5-8.5
            </span>
          )}
        </div>
      )}
    </div>
  )
}

function Toggle({ checked, onChange, label, color }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all border ${
        checked
          ? `${color} border-current`
          : 'text-gray-400 dark:text-gray-500 border-gray-300 dark:border-gray-600 opacity-60'
      }`}
    >
      <span className={`w-3 h-3 rounded-full border-2 transition-all ${
        checked ? 'bg-current border-current' : 'border-gray-400 dark:border-gray-500'
      }`} />
      {label}
    </button>
  )
}

export default function ProvincialAnalysis() {
  const { data, loading, error, retrying, elapsed } = useFetch('/historical/provincial-analysis')
  const [showSsp245, setShowSsp245] = useState(false)
  const [showSsp585, setShowSsp585] = useState(false)

  if (loading) return <Loader retrying={retrying} elapsed={elapsed} />
  if (error) return <ErrorBox message={error} />

  const { top5, bottom5, province_avg, historical, ssp245, ssp585 } = data
  const hasSsp = !!(ssp245 || ssp585)

  const topChartData = buildChartData(top5, historical, ssp245, ssp585, showSsp245, showSsp585)
  const botChartData = buildChartData(bottom5, historical, ssp245, ssp585, showSsp245, showSsp585)

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Provincial Analysis</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 max-w-3xl leading-relaxed">
          Year-by-year yield trends for the top 5 and bottom 5 banana-producing provinces,
          ranked by historical average yield (2010&ndash;2024).
        </p>
      </div>

      {hasSsp && (
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500 dark:text-gray-400">Projections:</span>
          <Toggle
            checked={showSsp245}
            onChange={setShowSsp245}
            label="SSP2-4.5"
            color="text-blue-600 dark:text-blue-400"
          />
          <Toggle
            checked={showSsp585}
            onChange={setShowSsp585}
            label="SSP5-8.5"
            color="text-orange-600 dark:text-orange-400"
          />
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {top5.map((prov, i) => (
          <div key={prov} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-3">
            <div className="flex items-center gap-2 mb-1">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: TOP_COLORS[i] }} />
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">#{i + 1}</span>
            </div>
            <p className="text-sm font-semibold text-gray-800 dark:text-gray-100 truncate">{prov}</p>
            <p className="text-lg font-bold text-emerald-700 dark:text-emerald-400">{province_avg[prov]} t/ha</p>
          </div>
        ))}
      </div>

      <ProvinceChart
        title="Top 5 Provinces by Yield"
        subtitle={`Highest-yielding provinces: ${top5.join(', ')}`}
        provinces={top5}
        colors={TOP_COLORS}
        chartData={topChartData}
        showSsp245={showSsp245}
        showSsp585={showSsp585}
      />

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {bottom5.map((prov, i) => (
          <div key={prov} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-3">
            <div className="flex items-center gap-2 mb-1">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: BOT_COLORS[i] }} />
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">#{82 - 4 + i}</span>
            </div>
            <p className="text-sm font-semibold text-gray-800 dark:text-gray-100 truncate">{prov}</p>
            <p className="text-lg font-bold text-red-600 dark:text-red-400">{province_avg[prov]} t/ha</p>
          </div>
        ))}
      </div>

      <ProvinceChart
        title="Bottom 5 Provinces by Yield"
        subtitle={`Lowest-yielding provinces: ${bottom5.join(', ')}`}
        provinces={bottom5}
        colors={BOT_COLORS}
        chartData={botChartData}
        showSsp245={showSsp245}
        showSsp585={showSsp585}
      />

      <CollapsibleSection title="Average Yield Summary" defaultOpen={false} badge="10 provinces">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="text-left p-2 text-gray-500 dark:text-gray-400">Rank</th>
                <th className="text-left p-2 text-gray-500 dark:text-gray-400">Province</th>
                <th className="text-right p-2 text-gray-500 dark:text-gray-400">Avg Yield (t/ha)</th>
                <th className="text-left p-2 text-gray-500 dark:text-gray-400">Category</th>
              </tr>
            </thead>
            <tbody>
              {top5.map((prov, i) => (
                <tr key={prov} className="border-t border-gray-100 dark:border-gray-700">
                  <td className="p-2 text-gray-400 text-xs">{i + 1}</td>
                  <td className="p-2 font-medium text-gray-700 dark:text-gray-200">
                    <span className="inline-block w-2.5 h-2.5 rounded-full mr-2" style={{ backgroundColor: TOP_COLORS[i] }} />
                    {prov}
                  </td>
                  <td className="p-2 text-right font-mono text-emerald-700 dark:text-emerald-400">{province_avg[prov]}</td>
                  <td className="p-2">
                    <span className="text-[10px] font-medium bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 px-2 py-0.5 rounded-full">
                      Top 5
                    </span>
                  </td>
                </tr>
              ))}
              {bottom5.map((prov, i) => (
                <tr key={prov} className="border-t border-gray-100 dark:border-gray-700">
                  <td className="p-2 text-gray-400 text-xs">{78 + i}</td>
                  <td className="p-2 font-medium text-gray-700 dark:text-gray-200">
                    <span className="inline-block w-2.5 h-2.5 rounded-full mr-2" style={{ backgroundColor: BOT_COLORS[i] }} />
                    {prov}
                  </td>
                  <td className="p-2 text-right font-mono text-red-600 dark:text-red-400">{province_avg[prov]}</td>
                  <td className="p-2">
                    <span className="text-[10px] font-medium bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400 px-2 py-0.5 rounded-full">
                      Bottom 5
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CollapsibleSection>
    </div>
  )
}
