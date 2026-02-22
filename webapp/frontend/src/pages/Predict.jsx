import { useState, useEffect, useMemo } from 'react'
import { useFetch, Loader, ErrorBox, StatCard, ExportButton, SearchableSelect, Accordion, useChartTheme } from '../hooks'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
  LineChart, Line, Legend,
} from 'recharts'

const API = (import.meta.env.VITE_API_URL || '') + '/api'

const FEATURE_LABELS = {
  tmp: 'Mean Temp (\u00b0C)', tmx: 'Max Temp (\u00b0C)', tmn: 'Min Temp (\u00b0C)',
  dtr: 'Diurnal Temp Range (\u00b0C)', pre: 'Precipitation (mm)',
  pet: 'Potential ET (mm)', aet: 'Actual ET (mm)', def: 'Water Deficit (mm)',
  cld: 'Cloud Cover (%)', wet: 'Wet Days', vap: 'Vapor Pressure (kPa)',
  vpd: 'Vapor Pressure Deficit (kPa)', PDSI: 'Palmer Drought Index',
  q: 'Specific Humidity (kg/kg)', soil: 'Soil Moisture',
  srad: 'Solar Radiation (W/m\u00b2)', ws: 'Wind Speed (m/s)',
}

const FEATURE_GROUPS = {
  Temperature: ['tmp', 'tmx', 'tmn', 'dtr'],
  Precipitation: ['pre', 'pet', 'aet', 'def'],
  Atmospheric: ['cld', 'wet', 'vap', 'vpd'],
  Other: ['PDSI', 'q', 'soil', 'srad', 'ws'],
}

export default function Predict() {
  const { data: modelData, loading: mLoad, error: mErr, retrying, elapsed } = useFetch('/predict/models')
  const { data: featureData, loading: fLoad } = useFetch('/predict/features')
  const { data: provinces } = useFetch('/historical/provinces')

  const [selectedModel, setSelectedModel] = useState('')
  const [featureValues, setFeatureValues] = useState({})
  const [selectedProvince, setSelectedProvince] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [predError, setPredError] = useState(null)

  const [batchMode, setBatchMode] = useState(false)
  const [batchScenario, setBatchScenario] = useState('ssp245')
  const [batchYear, setBatchYear] = useState(2025)
  const [batchProvince, setBatchProvince] = useState('')
  const [batchResults, setBatchResults] = useState(null)
  const [batchPredicting, setBatchPredicting] = useState(false)
  const [batchError, setBatchError] = useState(null)
  const chart = useChartTheme()

  useEffect(() => {
    if (featureData?.stats && Object.keys(featureValues).length === 0) {
      const defaults = {}
      for (const f of featureData.features) {
        defaults[f] = featureData.stats[f]?.mean ?? 0
      }
      setFeatureValues(defaults)
    }
  }, [featureData])

  useEffect(() => {
    if (modelData?.models?.length && !selectedModel) {
      setSelectedModel(modelData.models[0].key)
    }
  }, [modelData])

  const handleFeatureChange = (feature, value) => {
    setFeatureValues((prev) => ({ ...prev, [feature]: parseFloat(value) || 0 }))
  }

  const loadProvinceDefaults = (province) => {
    setSelectedProvince(province)
    if (!province || !featureData?.province_defaults?.[province]) return
    const defaults = featureData.province_defaults[province]
    setFeatureValues((prev) => ({ ...prev, ...defaults }))
  }

  const resetToMeans = () => {
    if (!featureData?.stats) return
    const defaults = {}
    for (const f of featureData.features) {
      defaults[f] = featureData.stats[f]?.mean ?? 0
    }
    setFeatureValues(defaults)
    setSelectedProvince('')
  }

  const handlePredict = async () => {
    setPredicting(true)
    setPredError(null)
    setPrediction(null)
    try {
      const resp = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: featureValues,
          model_key: selectedModel || null,
        }),
      })
      if (!resp.ok) {
        let msg = `HTTP ${resp.status}`
        try { const err = await resp.json(); msg = err.detail || msg } catch {}
        throw new Error(msg)
      }
      const data = await resp.json()
      setPrediction(data)
    } catch (e) {
      setPredError(e.message)
    } finally {
      setPredicting(false)
    }
  }

  const handleBatchPredict = async () => {
    setBatchPredicting(true)
    setBatchError(null)
    setBatchResults(null)
    try {
      const resp = await fetch(`${API}/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario: batchScenario,
          year: batchYear || null,
          province: batchProvince || null,
          model_key: selectedModel || null,
        }),
      })
      if (!resp.ok) {
        let msg = `HTTP ${resp.status}`
        try { const err = await resp.json(); msg = err.detail || msg } catch {}
        throw new Error(msg)
      }
      const data = await resp.json()
      setBatchResults(data)
    } catch (e) {
      setBatchError(e.message)
    } finally {
      setBatchPredicting(false)
    }
  }

  const handleReloadModels = async () => {
    try {
      const resp = await fetch(`${API}/predict/reload-models`, { method: 'POST' })
      const data = await resp.json()
      if (data.status === 'ok') {
        window.location.reload()
      } else {
        alert(data.message || 'No models found.')
      }
    } catch {
      alert('Failed to reload models.')
    }
  }

  const groupedFeatures = useMemo(() => {
    if (!featureData?.features) return {}
    const groups = {}
    const assigned = new Set()
    for (const [group, members] of Object.entries(FEATURE_GROUPS)) {
      const active = members.filter((f) => featureData.features.includes(f))
      if (active.length) {
        groups[group] = active
        active.forEach((f) => assigned.add(f))
      }
    }
    const remaining = featureData.features.filter((f) => !assigned.has(f))
    if (remaining.length) groups['Other'] = [...(groups['Other'] || []), ...remaining]
    return groups
  }, [featureData])

  if (mLoad || fLoad) return <Loader retrying={retrying} elapsed={elapsed} />
  if (mErr) return <ErrorBox message={mErr} />

  if (!modelData?.available) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Yield Prediction</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-700 rounded-lg p-6 text-yellow-800 dark:text-yellow-200">
          <h3 className="font-semibold text-lg mb-2">Models Not Available</h3>
          <p className="mb-3 text-sm">
            No trained models were found. The training pipeline needs to complete first.
          </p>
          <p className="text-xs mb-4 text-yellow-700 dark:text-yellow-300">
            After training finishes, models are saved to <code className="bg-yellow-100 dark:bg-yellow-800 px-1 rounded">Training/Models/top3/</code>.
          </p>
          <button
            onClick={handleReloadModels}
            className="px-4 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700 transition text-sm font-medium"
          >
            Check for Models
          </button>
        </div>
      </div>
    )
  }

  const batchByProvince = batchResults?.predictions
    ? [...batchResults.predictions]
        .sort((a, b) => b.predicted_yield - a.predicted_yield)
    : []

  const batchByYear = batchResults?.predictions && batchProvince
    ? [...batchResults.predictions].sort((a, b) => a.year - b.year)
    : []

  const groupEntries = Object.entries(groupedFeatures)

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center justify-between flex-wrap gap-3">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Yield Prediction</h2>
          <div className="flex gap-2">
            <button
              onClick={() => setBatchMode(false)}
              className={`px-4 py-1.5 rounded-md font-medium text-sm transition ${
                !batchMode
                  ? 'bg-emerald-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              Single Prediction
            </button>
            <button
              onClick={() => setBatchMode(true)}
              className={`px-4 py-1.5 rounded-md font-medium text-sm transition ${
                batchMode
                  ? 'bg-emerald-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              Batch / SSP Prediction
            </button>
          </div>
        </div>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 max-w-3xl leading-relaxed">
          Generate yield predictions using trained ML models. Single mode accepts custom
          climate inputs; batch mode uses projected climate data from CMIP6 SSP scenarios.
        </p>
      </div>

      {/* Model Selection */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200">Trained Models</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">Select a model for prediction</p>
          </div>
          <button
            onClick={handleReloadModels}
            className="text-xs text-emerald-600 dark:text-emerald-400 hover:text-emerald-800 dark:hover:text-emerald-300 underline font-medium"
          >
            Reload models
          </button>
        </div>
        <div className="grid sm:grid-cols-3 gap-3 mt-4">
          {modelData.models.map((m) => (
            <button
              key={m.key}
              onClick={() => setSelectedModel(m.key)}
              className={`p-3 rounded-lg border-2 text-left transition ${
                selectedModel === m.key
                  ? 'border-emerald-600 bg-emerald-50 dark:bg-emerald-900/30'
                  : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold text-sm text-gray-800 dark:text-gray-100">{m.name}</span>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                  m.rank === 1 ? 'bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300' :
                  m.rank === 2 ? 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300' :
                  'bg-orange-50 dark:bg-orange-900/30 text-orange-600 dark:text-orange-300'
                }`}>
                  #{m.rank}
                </span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                CV R&sup2; = {m.cv_r2}
              </p>
              <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-0.5">
                {m.features.length} features
              </p>
            </button>
          ))}
        </div>
      </div>

      {!batchMode ? (
        <>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-3">
              <div>
                <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200">Climate Feature Inputs</h3>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                  Enter values manually or load province-level averages from the historical dataset
                </p>
              </div>
              <div className="flex gap-2 items-center flex-wrap sm:flex-nowrap">
                <SearchableSelect
                  options={provinces || []}
                  value={selectedProvince}
                  onChange={loadProvinceDefaults}
                  placeholder="Search province..."
                  className="w-full sm:w-56"
                />
                <button
                  onClick={resetToMeans}
                  className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 underline whitespace-nowrap"
                >
                  Reset to national means
                </button>
              </div>
            </div>

            {groupEntries.map(([group, features], idx) => (
              <Accordion
                key={group}
                title={group}
                badge={`${features.length} features`}
                defaultOpen={idx === 0}
              >
                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {features.map((f) => {
                    const stat = featureData?.stats?.[f]
                    return (
                      <div key={f}>
                        <label className="block text-[11px] font-medium text-gray-600 dark:text-gray-300 mb-1">
                          {FEATURE_LABELS[f] || f}
                          <span className="text-gray-400 dark:text-gray-500 ml-1">({f})</span>
                        </label>
                        <input
                          type="number"
                          step="any"
                          value={featureValues[f] ?? ''}
                          onChange={(e) => handleFeatureChange(f, e.target.value)}
                          className="w-full border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1.5 text-sm font-mono bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                        />
                        {stat && (
                          <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-0.5">
                            Range: {stat.min} &ndash; {stat.max} | Mean: {stat.mean}
                          </p>
                        )}
                      </div>
                    )
                  })}
                </div>
              </Accordion>
            ))}

            <div className="flex items-center gap-4 mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
              <button
                onClick={handlePredict}
                disabled={predicting}
                className="px-6 py-2 bg-emerald-600 text-white rounded-md hover:bg-emerald-700 transition font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {predicting ? 'Predicting...' : 'Predict Yield'}
              </button>
              {selectedProvince && (
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Using average climate data from <strong className="text-gray-700 dark:text-gray-200">{selectedProvince}</strong>
                </p>
              )}
            </div>
          </div>

          {predError && <ErrorBox message={predError} />}

          {prediction && (
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
              <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200 mb-4">Prediction Result</h3>

              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
                <div className="bg-emerald-50 dark:bg-emerald-900/30 rounded-lg p-4 border border-emerald-200 dark:border-emerald-700">
                  <p className="text-xs text-emerald-600 dark:text-emerald-400 font-medium uppercase tracking-wide">Predicted Yield</p>
                  <p className="text-2xl font-bold text-emerald-700 dark:text-emerald-300 mt-1">
                    {prediction.predicted_yield.toFixed(2)}
                  </p>
                  <p className="text-[10px] text-emerald-500 dark:text-emerald-400 mt-1">metric tons / hectare</p>
                </div>
                {prediction.confidence_interval && (
                  <div className="bg-blue-50 dark:bg-blue-900/30 rounded-lg p-4 border border-blue-200 dark:border-blue-700">
                    <p className="text-xs text-blue-600 dark:text-blue-400 font-medium uppercase tracking-wide">{prediction.confidence_interval.level} CI</p>
                    <p className="text-lg font-bold text-blue-700 dark:text-blue-300 mt-1">
                      {prediction.confidence_interval.lower.toFixed(2)} &ndash; {prediction.confidence_interval.upper.toFixed(2)}
                    </p>
                    <p className="text-[10px] text-blue-500 dark:text-blue-400 mt-1">RMSE: {prediction.rmse?.toFixed(2)} t/ha</p>
                  </div>
                )}
                <StatCard
                  label="National Average"
                  value={`${prediction.context.national_avg} t/ha`}
                  sub={"Historical 2010\u20132024"}
                />
                <StatCard
                  label="vs. National Avg"
                  value={`${prediction.predicted_yield > prediction.context.national_avg ? '+' : ''}${(prediction.predicted_yield - prediction.context.national_avg).toFixed(2)} t/ha`}
                  sub={prediction.predicted_yield >= prediction.context.national_avg ? 'Above average' : 'Below average'}
                />
                <StatCard
                  label="Model CV R\u00b2"
                  value={prediction.cv_r2}
                  sub={`Model: ${prediction.model_name}`}
                />
              </div>

              <div className="mt-4">
                <h4 className="text-xs font-semibold text-gray-600 dark:text-gray-300 mb-2 uppercase tracking-wide">Yield Comparison</h4>
                <ResponsiveContainer width="100%" height={120}>
                  <BarChart
                    data={[
                      { name: 'Predicted', value: prediction.predicted_yield },
                      { name: 'National Avg', value: prediction.context.national_avg },
                      { name: 'Historical Min', value: prediction.context.national_min },
                      { name: 'Historical Max', value: prediction.context.national_max },
                    ]}
                    layout="vertical"
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                    <XAxis type="number" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11 }} />
                    <Tooltip contentStyle={chart.tooltip} formatter={(v) => `${v.toFixed(2)} t/ha`} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      <Cell fill="#059669" />
                      <Cell fill="#6b7280" />
                      <Cell fill="#dc2626" />
                      <Cell fill="#2563eb" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
            <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200 mb-1">
              Batch Prediction with SSP Scenarios
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
              Use projected climate data from the five-GCM CMIP6 ensemble to predict yields for multiple provinces and years.
            </p>

            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              <div>
                <label className="block text-[11px] font-medium text-gray-600 dark:text-gray-300 mb-1 uppercase tracking-wide">SSP Scenario</label>
                <select
                  value={batchScenario}
                  onChange={(e) => setBatchScenario(e.target.value)}
                  className="w-full border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1.5 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-emerald-500"
                >
                  <option value="ssp245">SSP2-4.5 (Moderate)</option>
                  <option value="ssp585">SSP5-8.5 (High Emissions)</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] font-medium text-gray-600 dark:text-gray-300 mb-1 uppercase tracking-wide">
                  Year (optional)
                </label>
                <select
                  value={batchYear}
                  onChange={(e) => setBatchYear(e.target.value ? parseInt(e.target.value) : null)}
                  className="w-full border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1.5 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-emerald-500"
                >
                  <option value="">All Years (2025&ndash;2034)</option>
                  {Array.from({ length: 10 }, (_, i) => 2025 + i).map((y) => (
                    <option key={y} value={y}>{y}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-[11px] font-medium text-gray-600 dark:text-gray-300 mb-1 uppercase tracking-wide">
                  Province (optional)
                </label>
                <SearchableSelect
                  options={provinces || []}
                  value={batchProvince}
                  onChange={setBatchProvince}
                  placeholder="All provinces"
                />
              </div>
              <div className="flex items-end sm:col-span-2 lg:col-span-1">
                <button
                  onClick={handleBatchPredict}
                  disabled={batchPredicting}
                  className="w-full px-4 py-1.5 bg-emerald-600 text-white rounded-md hover:bg-emerald-700 transition font-medium text-sm disabled:opacity-50"
                >
                  {batchPredicting ? 'Predicting...' : 'Run Batch Prediction'}
                </button>
              </div>
            </div>
          </div>

          {batchError && <ErrorBox message={batchError} />}

          {batchResults && (
            <>
              <div className="grid sm:grid-cols-3 gap-4">
                <StatCard
                  label="Predictions Generated"
                  value={batchResults.count}
                />
                <StatCard
                  label="Scenario"
                  value={batchResults.scenario === 'ssp245' ? 'SSP2-4.5' : 'SSP5-8.5'}
                />
                <StatCard
                  label="Avg Predicted Yield"
                  value={`${(batchResults.predictions.reduce((s, p) => s + p.predicted_yield, 0) / batchResults.predictions.length).toFixed(2)} t/ha`}
                />
              </div>

              {batchProvince && batchByYear.length > 1 && (
                <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
                  <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200 mb-1">
                    Predicted Yield Trend: {batchProvince}
                  </h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
                    Year-by-year projections under {batchResults.scenario === 'ssp245' ? 'SSP2-4.5' : 'SSP5-8.5'}
                  </p>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={batchByYear}>
                      <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                      <XAxis dataKey="year" tick={{ fontSize: 12 }} />
                      <YAxis
                        domain={['auto', 'auto']} tick={{ fontSize: 12 }}
                        label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
                      />
                      <Tooltip contentStyle={chart.tooltip} formatter={(v) => `${v.toFixed(2)} t/ha`} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      <Line
                        type="monotone"
                        dataKey="predicted_yield"
                        stroke="#059669"
                        strokeWidth={2}
                        name="Predicted Yield"
                        dot={{ r: 3 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {batchYear && batchByProvince.length > 1 && (
                <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
                  <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200 mb-1">
                    Province Predictions ({batchYear})
                  </h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
                    Predicted yield across all provinces for the selected year
                  </p>
                  <div className="overflow-x-auto">
                    <ResponsiveContainer
                      width={Math.max(800, batchByProvince.length * 25)}
                      height={400}
                    >
                      <BarChart data={batchByProvince}>
                        <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                        <XAxis
                          dataKey="province"
                          angle={-45}
                          textAnchor="end"
                          height={120}
                          tick={{ fontSize: 10 }}
                        />
                        <YAxis tick={{ fontSize: 11 }} label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
                        <Tooltip contentStyle={chart.tooltip} formatter={(v) => `${v.toFixed(2)} t/ha`} />
                        <Bar dataKey="predicted_yield" name="Predicted Yield" fill="#059669" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200">
                    Detailed Results
                  </h3>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-400 dark:text-gray-500">{batchResults.count} predictions</span>
                    <ExportButton
                      rows={batchResults.predictions.map((p) => ({
                        Province: p.province,
                        Year: p.year,
                        'Predicted Yield (t/ha)': p.predicted_yield.toFixed(2),
                        ...(p.actual_yield != null ? { 'Actual Yield (t/ha)': p.actual_yield.toFixed(2) } : {}),
                      }))}
                      filename={`batch_predictions_${batchResults.scenario}.csv`}
                    />
                  </div>
                </div>
                <div className="max-h-96 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-gray-50 dark:bg-gray-700">
                      <tr>
                        <th className="text-left p-2 text-gray-500 dark:text-gray-400">#</th>
                        <th className="text-left p-2 text-gray-500 dark:text-gray-400">Province</th>
                        <th className="text-right p-2 text-gray-500 dark:text-gray-400">Year</th>
                        <th className="text-right p-2 text-gray-500 dark:text-gray-400">Predicted (t/ha)</th>
                        {'actual_yield' in (batchResults.predictions[0] || {}) && (
                          <th className="text-right p-2 text-gray-500 dark:text-gray-400">Actual (t/ha)</th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.predictions.map((p, i) => (
                        <tr key={i} className="border-t border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                          <td className="p-2 text-gray-400 dark:text-gray-500 text-xs">{i + 1}</td>
                          <td className="p-2 font-medium text-gray-700 dark:text-gray-200">{p.province}</td>
                          <td className="p-2 text-right font-mono text-xs text-gray-600 dark:text-gray-300">{p.year}</td>
                          <td className="p-2 text-right font-mono text-xs text-emerald-700 dark:text-emerald-400">
                            {p.predicted_yield.toFixed(2)}
                          </td>
                          {p.actual_yield != null && (
                            <td className="p-2 text-right font-mono text-xs text-blue-700 dark:text-blue-400">
                              {p.actual_yield.toFixed(2)}
                            </td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}
