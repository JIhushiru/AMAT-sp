import { useParams, Link } from 'react-router-dom'
import { useFetch, Loader, ErrorBox, StatCard } from '../hooks'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar,
} from 'recharts'

const FEATURE_LABELS = {
  tmp: 'Mean Temp', tmx: 'Max Temp', tmn: 'Min Temp',
  dtr: 'DTR', pre: 'Precip', pet: 'PET', aet: 'AET', def: 'Deficit',
  cld: 'Cloud', wet: 'Wet Days', vap: 'Vapor P.', vpd: 'VPD',
  PDSI: 'PDSI', q: 'Humidity', soil: 'Soil Moist.', srad: 'Solar Rad.', ws: 'Wind',
}

export default function ProvinceDetail() {
  const { name } = useParams()
  const { data, loading, error, retrying, elapsed } = useFetch(`/historical/province/${encodeURIComponent(name)}`)
  const { data: climate } = useFetch('/historical/climate-features')

  if (loading) return <Loader retrying={retrying} elapsed={elapsed} />
  if (error) return <ErrorBox message={`Province "${name}" not found. ${error}`} />

  const trendData = data.trend
    ? Object.entries(data.trend).map(([y, v]) => ({ year: +y, yield: v }))
    : []

  // Radar chart: normalized feature values for this province
  const features = climate?.features || []
  const radarData = features.map((f) => {
    const vals = data.data.map((r) => r[f]).filter((v) => v != null)
    const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
    const globalMin = climate?.stats?.[f]?.min || 0
    const globalMax = climate?.stats?.[f]?.max || 1
    const range = globalMax - globalMin || 1
    return {
      feature: FEATURE_LABELS[f] || f,
      value: ((avg - globalMin) / range) * 100,
      raw: avg.toFixed(2),
    }
  })

  const yieldMin = Math.min(...trendData.map((d) => d.yield))
  const yieldMax = Math.max(...trendData.map((d) => d.yield))

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link to="/historical" className="text-emerald-600 dark:text-emerald-400 hover:underline text-sm">
          &larr; Back
        </Link>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">{name}</h2>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Avg Yield" value={`${data.avg_yield} t/ha`} />
        <StatCard label="Data Points" value={data.data.length} sub="year records" />
        <StatCard
          label="Min Yield"
          value={`${yieldMin.toFixed(2)} t/ha`}
        />
        <StatCard
          label="Max Yield"
          value={`${yieldMax.toFixed(2)} t/ha`}
        />
      </div>

      {/* Yield Trend */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3">Yield Trend</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis domain={['auto', 'auto']} label={{ value: 't/ha', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Line type="monotone" dataKey="yield" stroke="#059669" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Climate Profile Radar */}
      {radarData.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3">
            Climate Profile (Normalized)
          </h3>
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="feature" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} />
              <Radar
                name={name}
                dataKey="value"
                stroke="#059669"
                fill="#059669"
                fillOpacity={0.3}
              />
              <Tooltip
                content={({ payload }) => {
                  if (!payload?.[0]) return null
                  const d = payload[0].payload
                  return (
                    <div className="bg-white dark:bg-gray-800 border dark:border-gray-700 rounded p-2 text-xs shadow">
                      <p className="font-semibold text-gray-800 dark:text-gray-100">{d.feature}</p>
                      <p className="text-gray-600 dark:text-gray-400">Raw value: {d.raw}</p>
                      <p className="text-gray-600 dark:text-gray-400">Normalized: {d.value.toFixed(0)}%</p>
                    </div>
                  )
                }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Raw Data Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3">Raw Data</h3>
        <div className="overflow-x-auto">
          <table className="text-xs w-full">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="p-1.5 text-left text-gray-600 dark:text-gray-300">Year</th>
                <th className="p-1.5 text-right text-gray-600 dark:text-gray-300">Yield</th>
                {features.map((f) => (
                  <th key={f} className="p-1.5 text-right text-gray-600 dark:text-gray-300">{f}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.data
                .sort((a, b) => a.year - b.year)
                .map((row) => (
                  <tr key={row.year} className="border-t dark:border-gray-700">
                    <td className="p-1.5 font-medium text-gray-800 dark:text-gray-200">{row.year}</td>
                    <td className="p-1.5 text-right font-mono text-emerald-700 dark:text-emerald-400">
                      {row.yield?.toFixed(2)}
                    </td>
                    {features.map((f) => (
                      <td key={f} className="p-1.5 text-right font-mono text-gray-700 dark:text-gray-300">
                        {row[f]?.toFixed(2) ?? '-'}
                      </td>
                    ))}
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
