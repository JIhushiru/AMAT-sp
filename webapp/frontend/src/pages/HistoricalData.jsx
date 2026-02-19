import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useFetch, Loader, ErrorBox } from '../hooks'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ScatterChart, Scatter, ZAxis,
} from 'recharts'

const FEATURE_LABELS = {
  tmp: 'Mean Temp (°C)', tmx: 'Max Temp (°C)', tmn: 'Min Temp (°C)',
  dtr: 'Diurnal Temp Range (°C)', pre: 'Precipitation (mm)',
  pet: 'Potential ET (mm)', aet: 'Actual ET (mm)', def: 'Water Deficit (mm)',
  cld: 'Cloud Cover (%)', wet: 'Wet Days', vap: 'Vapor Pressure (kPa)',
  vpd: 'Vapor Pressure Deficit (kPa)', PDSI: 'Palmer Drought Index',
  q: 'Specific Humidity (kg/kg)', soil: 'Soil Moisture',
  srad: 'Solar Radiation (W/m²)', ws: 'Wind Speed (m/s)',
}

export default function HistoricalData() {
  const { data: summary, loading: sLoad, error: sErr, retrying, elapsed } = useFetch('/historical/summary')
  const { data: climate, loading: cLoad } = useFetch('/historical/climate-features')
  const { data: correlation, loading: corrLoad } = useFetch('/historical/correlation')
  const { data: rawData, loading: rLoad } = useFetch('/historical/data')
  const [selectedFeature, setSelectedFeature] = useState('tmp')
  const [searchTerm, setSearchTerm] = useState('')

  if (sLoad) return <Loader retrying={retrying} elapsed={elapsed} />
  if (sErr) return <ErrorBox message={sErr} />

  const trendData = summary
    ? Object.entries(summary.national_trend).map(([y, v]) => ({ year: +y, yield: v }))
    : []

  const provinces = summary
    ? Object.entries(summary.province_avg)
        .filter(([name]) => name.toLowerCase().includes(searchTerm.toLowerCase()))
    : []

  // Scatter: selected feature vs yield
  const scatterData = rawData
    ? rawData.map((r) => ({
        x: r[selectedFeature],
        y: r.yield,
        province: r.province,
      }))
    : []

  // Correlation heatmap data for top correlations with yield
  const yieldCorrelations = correlation
    ? correlation.columns
        .map((col, i) => ({
          feature: col,
          corr: correlation.data[correlation.columns.indexOf('yield')]?.[i] || 0,
        }))
        .filter((d) => d.feature !== 'yield')
        .sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr))
    : []

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Historical Data Explorer</h2>

      {/* National Trend */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold text-gray-700 mb-3">
          National Average Yield Trend
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis domain={['auto', 'auto']} label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Line type="monotone" dataKey="yield" stroke="#059669" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Feature vs Yield Scatter */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h3 className="text-lg font-semibold text-gray-700">
            Climate Feature vs Yield
          </h3>
          <select
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(e.target.value)}
            className="border rounded px-3 py-1.5 text-sm"
          >
            {climate?.features?.map((f) => (
              <option key={f} value={f}>
                {FEATURE_LABELS[f] || f}
              </option>
            ))}
          </select>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="x" type="number"
              label={{ value: FEATURE_LABELS[selectedFeature] || selectedFeature, position: 'insideBottom', offset: -5 }}
            />
            <YAxis dataKey="y" type="number" label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft' }} />
            <ZAxis range={[20, 20]} />
            <Tooltip
              content={({ payload }) => {
                if (!payload?.[0]) return null
                const d = payload[0].payload
                return (
                  <div className="bg-white border rounded p-2 text-xs shadow">
                    <p className="font-semibold">{d.province}</p>
                    <p>{selectedFeature}: {d.x?.toFixed(2)}</p>
                    <p>Yield: {d.y?.toFixed(2)} t/ha</p>
                  </div>
                )
              }}
            />
            <Scatter data={scatterData} fill="#059669" opacity={0.5} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Correlation with Yield */}
      {yieldCorrelations.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-700 mb-3">
            Feature Correlation with Yield
          </h3>
          <div className="space-y-2">
            {yieldCorrelations.map(({ feature, corr }) => (
              <div key={feature} className="flex items-center gap-3">
                <span className="text-sm w-28 text-right font-mono">{feature}</span>
                <div className="flex-1 bg-gray-100 rounded-full h-5 relative">
                  <div
                    className={`h-5 rounded-full ${corr >= 0 ? 'bg-emerald-500' : 'bg-red-500'}`}
                    style={{ width: `${Math.abs(corr) * 100}%` }}
                  />
                </div>
                <span className={`text-sm font-mono w-16 ${corr >= 0 ? 'text-emerald-700' : 'text-red-700'}`}>
                  {corr.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Province Table */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h3 className="text-lg font-semibold text-gray-700">
            Province Avg Yield ({summary?.total_provinces} provinces)
          </h3>
          <input
            type="text"
            placeholder="Search province..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="border rounded px-3 py-1.5 text-sm w-60"
          />
        </div>
        <div className="max-h-96 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-gray-50">
              <tr>
                <th className="text-left p-2">#</th>
                <th className="text-left p-2">Province</th>
                <th className="text-right p-2">Avg Yield (t/ha)</th>
                <th className="p-2"></th>
              </tr>
            </thead>
            <tbody>
              {provinces.map(([name, val], i) => (
                <tr key={name} className="border-t hover:bg-gray-50">
                  <td className="p-2 text-gray-400">{i + 1}</td>
                  <td className="p-2 font-medium">{name}</td>
                  <td className="p-2 text-right font-mono text-emerald-700">
                    {val.toFixed(2)}
                  </td>
                  <td className="p-2">
                    <Link
                      to={`/province/${encodeURIComponent(name)}`}
                      className="text-blue-600 hover:underline text-xs"
                    >
                      View
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Climate Feature Stats */}
      {climate && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-700 mb-3">
            Climate Feature Statistics
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="text-left p-2">Feature</th>
                  <th className="text-left p-2">Description</th>
                  <th className="text-right p-2">Mean</th>
                  <th className="text-right p-2">Std</th>
                  <th className="text-right p-2">Min</th>
                  <th className="text-right p-2">Max</th>
                </tr>
              </thead>
              <tbody>
                {climate.features.map((f) => (
                  <tr key={f} className="border-t">
                    <td className="p-2 font-mono">{f}</td>
                    <td className="p-2 text-gray-600">{FEATURE_LABELS[f] || f}</td>
                    <td className="p-2 text-right font-mono">{climate.stats[f].mean}</td>
                    <td className="p-2 text-right font-mono">{climate.stats[f].std}</td>
                    <td className="p-2 text-right font-mono">{climate.stats[f].min}</td>
                    <td className="p-2 text-right font-mono">{climate.stats[f].max}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
