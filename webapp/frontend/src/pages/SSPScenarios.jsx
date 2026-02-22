import { useState } from 'react'
import { useFetch, Loader, ErrorBox, EmptyState, StatCard, API_BASE, ExportButton, CollapsibleSection, useChartTheme } from '../hooks'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar, Cell,
} from 'recharts'

const scenarioInfo = {
  ssp245: {
    name: 'SSP2-4.5',
    subtitle: 'Middle of the Road',
    desc: 'Moderate emissions scenario. CO\u2082 concentrations stabilize near current levels by mid-century, then gradually decline. Represents intermediate challenges to mitigation and adaptation.',
    color: '#2563eb',
  },
  ssp585: {
    name: 'SSP5-8.5',
    subtitle: 'Fossil-fueled Development',
    desc: 'High emissions scenario. CO\u2082 concentrations continue rising throughout the 21st century driven by fossil-fuel-intensive economic growth.',
    color: '#dc2626',
  },
}

function CompareView() {
  const { data: d245, loading: l245, error: e245, retrying, elapsed } = useFetch('/ssp/ssp245')
  const { data: d585, loading: l585, error: e585 } = useFetch('/ssp/ssp585')
  const chart = useChartTheme()

  if (l245 || l585) return <Loader retrying={retrying} elapsed={elapsed} />
  if (e245) return <ErrorBox message={e245} />
  if (e585) return <ErrorBox message={e585} />

  if (!d245?.available || !d585?.available) {
    return (
      <EmptyState
        title="Both SSP datasets required for comparison"
        message={`${!d245?.available ? 'SSP2-4.5 data not available. ' : ''}${!d585?.available ? 'SSP5-8.5 data not available. ' : ''}Run the SSP pipeline for both scenarios first.`}
      />
    )
  }

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

      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
        <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200 mb-1">
          National Yield: Historical + Both SSP Scenarios
        </h3>
        <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
          Five-GCM ensemble mean predictions under each scenario
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
            <XAxis dataKey="year" tick={{ fontSize: 12 }} />
            <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12 }} label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} />
            <Tooltip contentStyle={chart.tooltip} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="historical" stroke="#059669" strokeWidth={2} name="Historical" dot={{ r: 3 }} />
            <Line type="monotone" dataKey="ssp245" stroke="#2563eb" strokeWidth={2} strokeDasharray="5 5" name="SSP2-4.5" dot={{ r: 3 }} />
            <Line type="monotone" dataKey="ssp585" stroke="#dc2626" strokeWidth={2} strokeDasharray="5 5" name="SSP5-8.5" dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {provinceCompare.length > 0 && (
        <CollapsibleSection
          title="Province % Change: SSP2-4.5 vs SSP5-8.5"
          defaultOpen={false}
          badge={`${provinceCompare.length} provinces`}
          actions={
            <ExportButton
              rows={provinceCompare.map((p) => ({
                Province: p.name,
                'SSP2-4.5 % Change': p.ssp245.toFixed(2),
                'SSP5-8.5 % Change': p.ssp585.toFixed(2),
              }))}
              filename="ssp_comparison_by_province.csv"
            />
          }
        >
          <div className="max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="text-left p-2 text-gray-500 dark:text-gray-400">Province</th>
                  <th className="text-right p-2 text-blue-600 dark:text-blue-400">SSP2-4.5</th>
                  <th className="text-right p-2 text-red-600 dark:text-red-400">SSP5-8.5</th>
                  <th className="text-right p-2 text-gray-500 dark:text-gray-400">Gap</th>
                </tr>
              </thead>
              <tbody>
                {provinceCompare.map((p) => (
                  <tr key={p.name} className="border-t border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="p-2 font-medium text-gray-700 dark:text-gray-200">{p.name}</td>
                    <td className={`p-2 text-right font-mono text-xs ${p.ssp245 >= 0 ? 'text-emerald-700 dark:text-emerald-400' : 'text-red-700 dark:text-red-400'}`}>
                      {p.ssp245 > 0 ? '+' : ''}{p.ssp245.toFixed(2)}%
                    </td>
                    <td className={`p-2 text-right font-mono text-xs ${p.ssp585 >= 0 ? 'text-emerald-700 dark:text-emerald-400' : 'text-red-700 dark:text-red-400'}`}>
                      {p.ssp585 > 0 ? '+' : ''}{p.ssp585.toFixed(2)}%
                    </td>
                    <td className="p-2 text-right font-mono text-xs text-gray-500 dark:text-gray-400">
                      {(p.ssp585 - p.ssp245).toFixed(2)}pp
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CollapsibleSection>
      )}
    </div>
  )
}

export default function SSPScenarios() {
  const [view, setView] = useState('single')
  const [scenario, setScenario] = useState('ssp245')
  const { data, loading, error, retrying, elapsed } = useFetch(`/ssp/${scenario}`)

  const info = scenarioInfo[scenario]

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center justify-between flex-wrap gap-3">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">SSP Scenarios</h2>
          <div className="flex gap-2">
            {view === 'single' && Object.entries(scenarioInfo).map(([key, s]) => (
              <button
                key={key}
                onClick={() => setScenario(key)}
                className={`px-4 py-1.5 rounded-md font-medium text-sm transition ${
                  scenario === key
                    ? 'text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
                style={scenario === key ? { backgroundColor: s.color } : {}}
              >
                {s.name}
              </button>
            ))}
            <button
              onClick={() => setView(view === 'single' ? 'compare' : 'single')}
              className={`px-4 py-1.5 rounded-md font-medium text-sm transition ${
                view === 'compare'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {view === 'compare' ? 'Single View' : 'Compare Both'}
            </button>
          </div>
        </div>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 max-w-3xl leading-relaxed">
          Future banana yield projections (2025&ndash;2034) derived from a five-model CMIP6
          ensemble with delta bias correction. Climate features extracted via Google Earth
          Engine from NASA NEX-GDDP-CMIP6.
        </p>
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
  const [showAllProvinces, setShowAllProvinces] = useState(false)
  const chart = useChartTheme()

  if (loading) return <Loader retrying={retrying} elapsed={elapsed} />
  if (error) return <ErrorBox message={error} />

  if (!data?.available) {
    return (
      <EmptyState
        title={`${info.name} Data Not Available`}
        message={`${info.desc} Run the SSP pipeline (gee_to_gdrive.py, tif_to_excel.py, merge_excels.py, future_predictions.py, projections.py, shap_analysis.py) to generate predictions.`}
      />
    )
  }

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

  const displayedProvinces = showAllProvinces || provinceSummary.length <= 20
    ? provinceSummary
    : [...provinceSummary.slice(0, 10), ...provinceSummary.slice(-10)]

  return (
    <>
      <div className="bg-white dark:bg-gray-800 rounded-lg border-l-4 border border-gray-200 dark:border-gray-700 p-4" style={{ borderLeftColor: info.color }}>
        <h3 className="font-semibold text-base" style={{ color: info.color }}>
          {info.name}: {info.subtitle}
        </h3>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 leading-relaxed">{info.desc}</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Provinces Increasing" value={increasing} sub="Yield projected to rise" />
        <StatCard label="Provinces Decreasing" value={decreasing} sub="Yield projected to drop" />
        <StatCard
          label="Future Avg Yield"
          value={
            data.future_national_trend
              ? `${(Object.values(data.future_national_trend).reduce((a, b) => a + b, 0) / Object.values(data.future_national_trend).length).toFixed(2)} t/ha`
              : 'N/A'
          }
        />
        <StatCard label="Projection Period" value={"2025\u20132034"} sub="5-GCM ensemble / CMIP6" />
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
        <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200 mb-1">
          National Yield Trend: Historical vs {info.name}
        </h3>
        <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
          Observed yield (2010&ndash;2024) and projected yield (2025&ndash;2034) under {info.name}
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
            <XAxis dataKey="year" tick={{ fontSize: 12 }} />
            <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12 }} label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} />
            <Tooltip contentStyle={chart.tooltip} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="historical" stroke="#059669" strokeWidth={2} name={"Historical (2010\u20132024)"} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="future" stroke={info.color} strokeWidth={2} strokeDasharray="5 5" name={`${info.name} (2025\u20132034)`} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {provinceSummary.length > 0 && (
        <CollapsibleSection title={`Province Yield % Change (Historical vs ${info.name})`}>
          <div className="flex items-center gap-3 mb-3">
            {provinceSummary.length > 20 && (
              <button
                onClick={() => setShowAllProvinces(!showAllProvinces)}
                className="text-xs text-emerald-600 dark:text-emerald-400 hover:underline font-medium"
              >
                {showAllProvinces ? 'Show Top & Bottom 10' : `Show All ${provinceSummary.length} Provinces`}
              </button>
            )}
          </div>
          <ResponsiveContainer width="100%" height={showAllProvinces ? 600 : 400}>
            <BarChart data={displayedProvinces}>
              <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} label={{ value: '% Change', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
              <Tooltip contentStyle={chart.tooltip} />
              <Bar dataKey="% Change" name="% Change" radius={[4, 4, 0, 0]}>
                {displayedProvinces.map((entry, i) => (
                  <Cell key={i} fill={(entry['% Change'] || 0) >= 0 ? '#059669' : '#dc2626'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CollapsibleSection>
      )}

      {data.plots && data.plots.length > 0 && (
        <CollapsibleSection title="Generated Plots" defaultOpen={false} badge={`${data.plots.length} plots`}>
          <div className="grid md:grid-cols-2 gap-4">
            {data.plots.map((plot) => (
              <div key={plot}>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                  {plot.replace('.png', '').replace(/_/g, ' ')}
                </p>
                <img
                  src={`${API_BASE}/ssp/${scenario}/plot/${plot}`}
                  alt={plot}
                  className="w-full rounded border border-gray-200 dark:border-gray-600"
                  loading="lazy"
                />
              </div>
            ))}
          </div>
        </CollapsibleSection>
      )}

      {data.shap_images && data.shap_images.length > 0 && (
        <CollapsibleSection title="SHAP Feature Importance" defaultOpen={false} badge={`${data.shap_images.length} plots`}>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
            SHapley Additive exPlanations (SHAP) values indicating the contribution of each climate feature to the model predictions.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            {data.shap_images.map((img) => (
              <div key={img}>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                  {img.replace('.png', '').replace(/_/g, ' ')}
                </p>
                <img
                  src={`${API_BASE}/ssp/${scenario}/plot/${img}`}
                  alt={img}
                  className="w-full rounded border border-gray-200 dark:border-gray-600"
                  loading="lazy"
                />
              </div>
            ))}
          </div>
        </CollapsibleSection>
      )}

      {provinceSummary.length > 0 && (
        <CollapsibleSection
          title="Province-Level Summary"
          defaultOpen={false}
          badge={`${provinceSummary.length} provinces`}
          actions={
            <ExportButton
              rows={provinceSummary.map((p) => ({
                Province: p.name,
                'Historical Avg': p['Historical Avg (2010\u20132024)']?.toFixed(2) ?? '',
                'Future Avg': p['Future Avg (2025\u20132034)']?.toFixed(2) ?? '',
                '% Change': p['% Change']?.toFixed(2) ?? '',
              }))}
              filename={`${info.name}_province_summary.csv`}
            />
          }
        >
          <div className="max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="text-left p-2 text-gray-500 dark:text-gray-400">Province</th>
                  <th className="text-right p-2 text-gray-500 dark:text-gray-400">Historical Avg</th>
                  <th className="text-right p-2 text-gray-500 dark:text-gray-400">Future Avg</th>
                  <th className="text-right p-2 text-gray-500 dark:text-gray-400">% Change</th>
                </tr>
              </thead>
              <tbody>
                {provinceSummary.map((p) => (
                  <tr key={p.name} className="border-t border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="p-2 font-medium text-gray-700 dark:text-gray-200">{p.name}</td>
                    <td className="p-2 text-right font-mono text-xs text-gray-600 dark:text-gray-300">
                      {p['Historical Avg (2010\u20132024)']?.toFixed(2) || 'N/A'}
                    </td>
                    <td className="p-2 text-right font-mono text-xs text-gray-600 dark:text-gray-300">
                      {p['Future Avg (2025\u20132034)']?.toFixed(2) || 'N/A'}
                    </td>
                    <td className={`p-2 text-right font-mono text-xs font-semibold ${
                      (p['% Change'] || 0) >= 0 ? 'text-emerald-700 dark:text-emerald-400' : 'text-red-700 dark:text-red-400'
                    }`}>
                      {p['% Change'] != null ? `${p['% Change'] > 0 ? '+' : ''}${p['% Change'].toFixed(2)}%` : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CollapsibleSection>
      )}
    </>
  )
}
