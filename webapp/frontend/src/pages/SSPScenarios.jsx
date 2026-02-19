import { useState } from 'react'
import { useFetch, Loader, ErrorBox, StatCard, API_BASE, ExportButton } from '../hooks'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar, Cell,
} from 'recharts'

export default function SSPScenarios() {
  const [scenario, setScenario] = useState('ssp245')
  const { data, loading, error, retrying, elapsed } = useFetch(`/ssp/${scenario}`)

  const scenarioInfo = {
    ssp245: {
      name: 'SSP2-4.5',
      subtitle: 'Middle of the Road',
      desc: 'Moderate emissions scenario. CO₂ around current levels until mid-century, then declining.',
      color: '#2563eb',
    },
    ssp585: {
      name: 'SSP5-8.5',
      subtitle: 'Fossil-fueled Development',
      desc: 'High emissions scenario. CO₂ continues rising throughout 21st century.',
      color: '#dc2626',
    },
  }
  const info = scenarioInfo[scenario]

  if (loading) return <Loader retrying={retrying} elapsed={elapsed} />
  if (error) return <ErrorBox message={error} />

  if (!data?.available) {
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-gray-800">SSP Scenarios</h2>
        <div className="flex gap-2 mb-4">
          {Object.entries(scenarioInfo).map(([key, s]) => (
            <button
              key={key}
              onClick={() => setScenario(key)}
              className={`px-4 py-2 rounded font-medium text-sm ${
                scenario === key
                  ? 'text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              style={scenario === key ? { backgroundColor: s.color } : {}}
            >
              {s.name}
            </button>
          ))}
        </div>
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-yellow-800">
          <h3 className="font-semibold text-lg mb-2">
            {info.name} Data Not Available
          </h3>
          <p className="mb-4">{info.desc}</p>
          <p className="text-sm">
            Run the SSP pipeline to generate predictions:
          </p>
          <ol className="list-decimal ml-5 mt-2 text-sm space-y-1">
            <li><code>python gee_to_gdrive.py</code> — Export climate data from Google Earth Engine</li>
            <li><code>python tif_to_excel.py</code> — Extract province-level features from rasters</li>
            <li><code>python merge_excels.py</code> — Merge historical + SSP climate data</li>
            <li><code>python future_predictions.py</code> — Train Cubist model and predict 2025-2034</li>
            <li><code>python projections.py</code> — Generate analysis and visualizations</li>
            <li><code>python shap_analysis.py</code> — SHAP feature importance analysis</li>
          </ol>
        </div>
      </div>
    )
  }

  // Combined historical + future trend
  const trendData = []
  if (data.historical_national_trend) {
    Object.entries(data.historical_national_trend).forEach(([y, v]) => {
      trendData.push({ year: +y, historical: v })
    })
  }
  if (data.future_national_trend) {
    Object.entries(data.future_national_trend).forEach(([y, v]) => {
      const existing = trendData.find((d) => d.year === +y)
      if (existing) existing.future = v
      else trendData.push({ year: +y, future: v })
    })
  }
  trendData.sort((a, b) => a.year - b.year)

  // Province summary sorted by % change
  const provinceSummary = data.province_summary
    ? Object.entries(data.province_summary)
        .map(([name, d]) => ({ name, ...d }))
        .sort((a, b) => (b['% Change'] || 0) - (a['% Change'] || 0))
    : []

  // 2024 vs 2034 comparison
  const compareData = data.compare_2024_2034
    ? Object.entries(data.compare_2024_2034)
        .map(([name, d]) => ({ name, ...d }))
        .sort((a, b) => (b['% Change'] || 0) - (a['% Change'] || 0))
    : []

  const increasing = provinceSummary.filter((p) => (p['% Change'] || 0) > 0).length
  const decreasing = provinceSummary.filter((p) => (p['% Change'] || 0) < 0).length

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h2 className="text-2xl font-bold text-gray-800">SSP Scenarios</h2>
        <div className="flex gap-2">
          {Object.entries(scenarioInfo).map(([key, s]) => (
            <button
              key={key}
              onClick={() => setScenario(key)}
              className={`px-4 py-2 rounded font-medium text-sm ${
                scenario === key
                  ? 'text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              style={scenario === key ? { backgroundColor: s.color } : {}}
            >
              {s.name}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-4 border-l-4" style={{ borderLeftColor: info.color }}>
        <h3 className="font-semibold text-lg" style={{ color: info.color }}>
          {info.name}: {info.subtitle}
        </h3>
        <p className="text-sm text-gray-600 mt-1">{info.desc}</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Provinces Increasing" value={increasing} sub="yield projected to rise" />
        <StatCard label="Provinces Decreasing" value={decreasing} sub="yield projected to drop" />
        <StatCard
          label="Future Avg Yield"
          value={
            data.future_national_trend
              ? `${(Object.values(data.future_national_trend).reduce((a, b) => a + b, 0) / Object.values(data.future_national_trend).length).toFixed(2)} t/ha`
              : 'N/A'
          }
        />
        <StatCard label="Projection Period" value="2025-2034" sub="5-GCM Ensemble / CMIP6" />
      </div>

      {/* National Trend */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold text-gray-700 mb-3">
          National Yield Trend: Historical vs {info.name}
        </h3>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis domain={['auto', 'auto']} label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone" dataKey="historical" stroke="#059669"
              strokeWidth={2} name="Historical (2010-2024)" dot={{ r: 3 }}
            />
            <Line
              type="monotone" dataKey="future" stroke={info.color}
              strokeWidth={2} strokeDasharray="5 5"
              name={`${info.name} (2025-2034)`} dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Province % Change Bar Chart */}
      {provinceSummary.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-700 mb-3">
            Province Yield % Change (Historical vs {info.name})
          </h3>
          <div className="overflow-x-auto">
            <ResponsiveContainer width={Math.max(800, provinceSummary.length * 20)} height={400}>
              <BarChart data={provinceSummary}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-90} textAnchor="end" height={120} tick={{ fontSize: 10 }} />
                <YAxis label={{ value: '% Change', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Bar dataKey="% Change" name="% Change">
                  {provinceSummary.map((entry, i) => (
                    <Cell key={i} fill={(entry['% Change'] || 0) >= 0 ? '#059669' : '#dc2626'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* SSP Plots */}
      {data.plots && data.plots.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-700 mb-3">Generated Plots</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {data.plots.map((plot) => (
              <div key={plot}>
                <p className="text-sm font-medium text-gray-600 mb-1">
                  {plot.replace('.png', '').replace(/_/g, ' ')}
                </p>
                <img
                  src={`${API_BASE}/ssp/${scenario}/plot/${plot}`}
                  alt={plot}
                  className="w-full rounded border"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* SHAP Images */}
      {data.shap_images && data.shap_images.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-700 mb-3">
            SHAP Feature Importance
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            {data.shap_images.map((img) => (
              <div key={img}>
                <p className="text-sm font-medium text-gray-600 mb-1">
                  {img.replace('.png', '').replace(/_/g, ' ')}
                </p>
                <img
                  src={`${API_BASE}/ssp/${scenario}/plot/${img}`}
                  alt={img}
                  className="w-full rounded border"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Province Detail Table */}
      {provinceSummary.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-gray-700">
              Province-Level Summary
            </h3>
            <ExportButton
              rows={provinceSummary.map((p) => ({
                Province: p.name,
                'Historical Avg': p['Historical Avg (2010\u20132024)']?.toFixed(2) ?? '',
                'Future Avg': p['Future Avg (2025\u20132034)']?.toFixed(2) ?? '',
                '% Change': p['% Change']?.toFixed(2) ?? '',
              }))}
              filename={`${info.name}_province_summary.csv`}
            />
          </div>
          <div className="max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-gray-50">
                <tr>
                  <th className="text-left p-2">Province</th>
                  <th className="text-right p-2">Historical Avg</th>
                  <th className="text-right p-2">Future Avg</th>
                  <th className="text-right p-2">% Change</th>
                </tr>
              </thead>
              <tbody>
                {provinceSummary.map((p) => (
                  <tr key={p.name} className="border-t hover:bg-gray-50">
                    <td className="p-2 font-medium">{p.name}</td>
                    <td className="p-2 text-right font-mono">
                      {p['Historical Avg (2010\u20132024)']?.toFixed(2) || 'N/A'}
                    </td>
                    <td className="p-2 text-right font-mono">
                      {p['Future Avg (2025\u20132034)']?.toFixed(2) || 'N/A'}
                    </td>
                    <td className={`p-2 text-right font-mono font-semibold ${
                      (p['% Change'] || 0) >= 0 ? 'text-emerald-700' : 'text-red-700'
                    }`}>
                      {p['% Change'] != null ? `${p['% Change'] > 0 ? '+' : ''}${p['% Change'].toFixed(2)}%` : 'N/A'}
                    </td>
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
