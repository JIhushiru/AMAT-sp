import { useState } from 'react'
import { useFetch, Loader, ErrorBox, StatCard, API_BASE, ExportButton } from '../hooks'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar, Cell,
} from 'recharts'

const scenarioInfo = {
  ssp245: {
    name: 'SSP2-4.5',
    subtitle: 'Middle of the Road',
    desc: 'Moderate emissions scenario. CO\u2082 around current levels until mid-century, then declining.',
    color: '#2563eb',
  },
  ssp585: {
    name: 'SSP5-8.5',
    subtitle: 'Fossil-fueled Development',
    desc: 'High emissions scenario. CO\u2082 continues rising throughout 21st century.',
    color: '#dc2626',
  },
}

function CompareView() {
  const { data: d245, loading: l245, error: e245, retrying, elapsed } = useFetch('/ssp/ssp245')
  const { data: d585, loading: l585, error: e585 } = useFetch('/ssp/ssp585')

  if (l245 || l585) return <Loader retrying={retrying} elapsed={elapsed} />
  if (e245) return <ErrorBox message={e245} />
  if (e585) return <ErrorBox message={e585} />

  if (!d245?.available || !d585?.available) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-yellow-800">
        <h3 className="font-semibold text-lg mb-2">Both SSP datasets required for comparison</h3>
        <p className="text-sm">
          {!d245?.available && 'SSP2-4.5 data not available. '}
          {!d585?.available && 'SSP5-8.5 data not available. '}
          Run the SSP pipeline for both scenarios first.
        </p>
      </div>
    )
  }

  // Build combined national trend
  const trendData = []
  if (d245.historical_national_trend) {
    Object.entries(d245.historical_national_trend).forEach(([y, v]) => {
      trendData.push({ year: +y, historical: v })
    })
  }
  if (d245.future_national_trend) {
    Object.entries(d245.future_national_trend).forEach(([y, v]) => {
      const existing = trendData.find((d) => d.year === +y)
      if (existing) existing.ssp245 = v
      else trendData.push({ year: +y, ssp245: v })
    })
  }
  if (d585.future_national_trend) {
    Object.entries(d585.future_national_trend).forEach(([y, v]) => {
      const existing = trendData.find((d) => d.year === +y)
      if (existing) existing.ssp585 = v
      else trendData.push({ year: +y, ssp585: v })
    })
  }
  trendData.sort((a, b) => a.year - b.year)

  // Province comparison
  const provinceCompare = []
  const ps245 = d245.province_summary || {}
  const ps585 = d585.province_summary || {}
  const allProvinces = [...new Set([...Object.keys(ps245), ...Object.keys(ps585)])]
  allProvinces.forEach((name) => {
    provinceCompare.push({
      name,
      ssp245: ps245[name]?.['% Change'] || 0,
      ssp585: ps585[name]?.['% Change'] || 0,
    })
  })
  provinceCompare.sort((a, b) => b.ssp245 - a.ssp245)

  const avg245 = d245.future_national_trend
    ? (Object.values(d245.future_national_trend).reduce((a, b) => a + b, 0) / Object.values(d245.future_national_trend).length).toFixed(2)
    : 'N/A'
  const avg585 = d585.future_national_trend
    ? (Object.values(d585.future_national_trend).reduce((a, b) => a + b, 0) / Object.values(d585.future_national_trend).length).toFixed(2)
    : 'N/A'

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="SSP2-4.5 Avg" value={`${avg245} t/ha`} sub="Moderate scenario" />
        <StatCard label="SSP5-8.5 Avg" value={`${avg585} t/ha`} sub="High emissions" />
        <StatCard
          label="Difference"
          value={avg245 !== 'N/A' && avg585 !== 'N/A'
            ? `${(avg585 - avg245).toFixed(2)} t/ha`
            : 'N/A'}
          sub="SSP5-8.5 minus SSP2-4.5"
        />
        <StatCard label="Provinces" value={allProvinces.length} sub="Compared across scenarios" />
      </div>

      {/* Combined trend chart */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold text-gray-700 mb-3">
          National Yield: Historical + Both SSP Scenarios
        </h3>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis domain={['auto', 'auto']} label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="historical" stroke="#059669" strokeWidth={2} name="Historical" dot={{ r: 3 }} />
            <Line type="monotone" dataKey="ssp245" stroke="#2563eb" strokeWidth={2} strokeDasharray="5 5" name="SSP2-4.5" dot={{ r: 3 }} />
            <Line type="monotone" dataKey="ssp585" stroke="#dc2626" strokeWidth={2} strokeDasharray="5 5" name="SSP5-8.5" dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Province % Change comparison */}
      {provinceCompare.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-gray-700">Province % Change: SSP2-4.5 vs SSP5-8.5</h3>
            <ExportButton
              rows={provinceCompare.map((p) => ({
                Province: p.name,
                'SSP2-4.5 % Change': p.ssp245.toFixed(2),
                'SSP5-8.5 % Change': p.ssp585.toFixed(2),
              }))}
              filename="ssp_comparison_by_province.csv"
            />
          </div>
          <div className="max-h-125 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-gray-50">
                <tr>
                  <th className="text-left p-2">Province</th>
                  <th className="text-right p-2 text-blue-700">SSP2-4.5</th>
                  <th className="text-right p-2 text-red-700">SSP5-8.5</th>
                  <th className="text-right p-2">Gap</th>
                </tr>
              </thead>
              <tbody>
                {provinceCompare.map((p) => (
                  <tr key={p.name} className="border-t hover:bg-gray-50">
                    <td className="p-2 font-medium">{p.name}</td>
                    <td className={`p-2 text-right font-mono ${p.ssp245 >= 0 ? 'text-emerald-700' : 'text-red-700'}`}>
                      {p.ssp245 > 0 ? '+' : ''}{p.ssp245.toFixed(2)}%
                    </td>
                    <td className={`p-2 text-right font-mono ${p.ssp585 >= 0 ? 'text-emerald-700' : 'text-red-700'}`}>
                      {p.ssp585 > 0 ? '+' : ''}{p.ssp585.toFixed(2)}%
                    </td>
                    <td className="p-2 text-right font-mono text-gray-600">
                      {(p.ssp585 - p.ssp245).toFixed(2)}pp
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

export default function SSPScenarios() {
  const [view, setView] = useState('single') // 'single' or 'compare'
  const [scenario, setScenario] = useState('ssp245')
  const { data, loading, error, retrying, elapsed } = useFetch(`/ssp/${scenario}`)

  const info = scenarioInfo[scenario]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h2 className="text-2xl font-bold text-gray-800">SSP Scenarios</h2>
        <div className="flex gap-2">
          {view === 'single' && Object.entries(scenarioInfo).map(([key, s]) => (
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
          <button
            onClick={() => setView(view === 'single' ? 'compare' : 'single')}
            className={`px-4 py-2 rounded font-medium text-sm ${
              view === 'compare'
                ? 'bg-purple-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {view === 'compare' ? 'Single View' : 'Compare Both'}
          </button>
        </div>
      </div>

      {view === 'compare' ? (
        <CompareView />
      ) : (
        <SingleScenarioView
          scenario={scenario}
          data={data}
          loading={loading}
          error={error}
          retrying={retrying}
          elapsed={elapsed}
          info={info}
        />
      )}
    </div>
  )
}

function SingleScenarioView({ scenario, data, loading, error, retrying, elapsed, info }) {
  if (loading) return <Loader retrying={retrying} elapsed={elapsed} />
  if (error) return <ErrorBox message={error} />

  if (!data?.available) {
    return (
      <div className="space-y-4">
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

  const provinceSummary = data.province_summary
    ? Object.entries(data.province_summary)
        .map(([name, d]) => ({ name, ...d }))
        .sort((a, b) => (b['% Change'] || 0) - (a['% Change'] || 0))
    : []

  const increasing = provinceSummary.filter((p) => (p['% Change'] || 0) > 0).length
  const decreasing = provinceSummary.filter((p) => (p['% Change'] || 0) < 0).length

  return (
    <>
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
    </>
  )
}
