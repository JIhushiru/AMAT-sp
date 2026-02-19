import { useState, useEffect, useMemo } from 'react'
import { useFetch, Loader, ErrorBox, StatCard, ExportButton, SearchableSelect } from '../hooks'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
  LineChart, Line, Legend,
} from 'recharts'

const API = (import.meta.env.VITE_API_URL || '') + '/api'

const FEATURE_LABELS = {
  tmp: 'Mean Temp (°C)', tmx: 'Max Temp (°C)', tmn: 'Min Temp (°C)',
  dtr: 'Diurnal Temp Range (°C)', pre: 'Precipitation (mm)',
  pet: 'Potential ET (mm)', aet: 'Actual ET (mm)', def: 'Water Deficit (mm)',
  cld: 'Cloud Cover (%)', wet: 'Wet Days', vap: 'Vapor Pressure (kPa)',
  vpd: 'Vapor Pressure Deficit (kPa)', PDSI: 'Palmer Drought Index',
  q: 'Specific Humidity (kg/kg)', soil: 'Soil Moisture',
  srad: 'Solar Radiation (W/m²)', ws: 'Wind Speed (m/s)',
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

  // Batch prediction state
  const [batchMode, setBatchMode] = useState(false)
  const [batchScenario, setBatchScenario] = useState('ssp245')
  const [batchYear, setBatchYear] = useState(2025)
  const [batchProvince, setBatchProvince] = useState('')
  const [batchResults, setBatchResults] = useState(null)
  const [batchPredicting, setBatchPredicting] = useState(false)
  const [batchError, setBatchError] = useState(null)

  // Initialize feature values with means when data loads
  useEffect(() => {
    if (featureData?.stats && Object.keys(featureValues).length === 0) {
      const defaults = {}
      for (const f of featureData.features) {
        defaults[f] = featureData.stats[f]?.mean ?? 0
      }
      setFeatureValues(defaults)
    }
  }, [featureData])

  // Set default model
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
        const err = await resp.json()
        throw new Error(err.detail || `HTTP ${resp.status}`)
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
        const err = await resp.json()
        throw new Error(err.detail || `HTTP ${resp.status}`)
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

  // Grouped features for the form
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

  // No models available yet
  if (!modelData?.available) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-800">Yield Prediction</h2>
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-yellow-800">
          <h3 className="font-semibold text-lg mb-2">Models Not Available Yet</h3>
          <p className="mb-3">
            No trained models were found. The training pipeline needs to complete first.
          </p>
          <p className="text-sm mb-4">
            After training finishes, models are saved to <code>Training/Models/top3/</code>.
          </p>
          <button
            onClick={handleReloadModels}
            className="px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700 transition text-sm font-medium"
          >
            Check for Models
          </button>
        </div>
      </div>
    )
  }

  // Batch results charts
  const batchByProvince = batchResults?.predictions
    ? [...batchResults.predictions]
        .sort((a, b) => b.predicted_yield - a.predicted_yield)
    : []

  const batchByYear = batchResults?.predictions && batchProvince
    ? [...batchResults.predictions].sort((a, b) => a.year - b.year)
    : []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h2 className="text-2xl font-bold text-gray-800">Yield Prediction</h2>
        <div className="flex gap-2">
          <button
            onClick={() => setBatchMode(false)}
            className={`px-4 py-2 rounded font-medium text-sm transition ${
              !batchMode ? 'bg-emerald-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Single Prediction
          </button>
          <button
            onClick={() => setBatchMode(true)}
            className={`px-4 py-2 rounded font-medium text-sm transition ${
              batchMode ? 'bg-emerald-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Batch / SSP Prediction
          </button>
        </div>
      </div>

      {/* Model Selection */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h3 className="text-lg font-semibold text-gray-700">Trained Models</h3>
            <p className="text-sm text-gray-500">Select the model to use for prediction</p>
          </div>
          <button
            onClick={handleReloadModels}
            className="text-sm text-emerald-600 hover:text-emerald-800 underline"
          >
            Reload models
          </button>
        </div>
        <div className="grid sm:grid-cols-3 gap-3 mt-3">
          {modelData.models.map((m) => (
            <button
              key={m.key}
              onClick={() => setSelectedModel(m.key)}
              className={`p-3 rounded-lg border-2 text-left transition ${
                selectedModel === m.key
                  ? 'border-emerald-600 bg-emerald-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold text-gray-800">{m.name}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  m.rank === 1 ? 'bg-yellow-100 text-yellow-700' :
                  m.rank === 2 ? 'bg-gray-100 text-gray-600' :
                  'bg-orange-50 text-orange-600'
                }`}>
                  #{m.rank}
                </span>
              </div>
              <p className="text-sm text-gray-500 mt-1">
                CV R² = {m.cv_r2}
              </p>
              <p className="text-xs text-gray-400 mt-0.5">
                {m.features.length} features
              </p>
            </button>
          ))}
        </div>
      </div>

      {!batchMode ? (
        /* ====== SINGLE PREDICTION MODE ====== */
        <>
          {/* Province Preset */}
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between flex-wrap gap-3 mb-3">
              <div>
                <h3 className="text-lg font-semibold text-gray-700">Climate Feature Inputs</h3>
                <p className="text-sm text-gray-500">
                  Enter climate values manually or load defaults from a province
                </p>
              </div>
              <div className="flex gap-2 items-center">
                <SearchableSelect
                  options={provinces || []}
                  value={selectedProvince}
                  onChange={loadProvinceDefaults}
                  placeholder="Search province..."
                  className="w-56"
                />
                <button
                  onClick={resetToMeans}
                  className="text-sm text-gray-500 hover:text-gray-700 underline whitespace-nowrap"
                >
                  Reset to national means
                </button>
              </div>
            </div>

            {/* Feature input groups */}
            {Object.entries(groupedFeatures).map(([group, features]) => (
              <div key={group} className="mb-4">
                <h4 className="text-sm font-semibold text-gray-600 mb-2 border-b pb-1">{group}</h4>
                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {features.map((f) => {
                    const stat = featureData?.stats?.[f]
                    return (
                      <div key={f} className="relative">
                        <label className="block text-xs font-medium text-gray-600 mb-1">
                          {FEATURE_LABELS[f] || f}
                          <span className="text-gray-400 ml-1">({f})</span>
                        </label>
                        <input
                          type="number"
                          step="any"
                          value={featureValues[f] ?? ''}
                          onChange={(e) => handleFeatureChange(f, e.target.value)}
                          className="w-full border rounded px-3 py-1.5 text-sm font-mono focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                        />
                        {stat && (
                          <p className="text-[10px] text-gray-400 mt-0.5">
                            Range: {stat.min} – {stat.max} | Mean: {stat.mean}
                          </p>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}

            {/* Predict Button */}
            <div className="flex items-center gap-4 mt-4 pt-4 border-t">
              <button
                onClick={handlePredict}
                disabled={predicting}
                className="px-6 py-2.5 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {predicting ? 'Predicting...' : 'Predict Yield'}
              </button>
              {selectedProvince && (
                <p className="text-sm text-gray-500">
                  Using average climate data from <strong>{selectedProvince}</strong>
                </p>
              )}
            </div>
          </div>

          {/* Prediction Error */}
          {predError && <ErrorBox message={predError} />}

          {/* Prediction Result */}
          {prediction && (
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-700 mb-4">Prediction Result</h3>

              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="bg-emerald-50 rounded-lg p-4 border border-emerald-200">
                  <p className="text-sm text-emerald-600 font-medium">Predicted Yield</p>
                  <p className="text-3xl font-bold text-emerald-700 mt-1">
                    {prediction.predicted_yield.toFixed(2)}
                  </p>
                  <p className="text-xs text-emerald-500 mt-1">metric tons / hectare</p>
                </div>
                <StatCard
                  label="National Average"
                  value={`${prediction.context.national_avg} t/ha`}
                  sub="Historical 2010-2024"
                />
                <StatCard
                  label="vs. National Avg"
                  value={`${prediction.predicted_yield > prediction.context.national_avg ? '+' : ''}${(prediction.predicted_yield - prediction.context.national_avg).toFixed(2)} t/ha`}
                  sub={prediction.predicted_yield >= prediction.context.national_avg ? 'Above average' : 'Below average'}
                />
                <StatCard
                  label="Model CV R²"
                  value={prediction.cv_r2}
                  sub={`Model: ${prediction.model_name}`}
                />
              </div>

              {/* Yield comparison bar */}
              <div className="mt-4">
                <h4 className="text-sm font-semibold text-gray-600 mb-2">Yield Comparison</h4>
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
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(v) => `${v.toFixed(2)} t/ha`} />
                    <Bar dataKey="value">
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
        /* ====== BATCH PREDICTION MODE ====== */
        <>
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold text-gray-700 mb-3">
              Batch Prediction with SSP Scenarios
            </h3>
            <p className="text-sm text-gray-500 mb-4">
              Use projected climate data from SSP scenarios to predict yields for multiple provinces or years.
            </p>

            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">SSP Scenario</label>
                <select
                  value={batchScenario}
                  onChange={(e) => setBatchScenario(e.target.value)}
                  className="w-full border rounded px-3 py-1.5 text-sm"
                >
                  <option value="ssp245">SSP2-4.5 (Moderate)</option>
                  <option value="ssp585">SSP5-8.5 (High Emissions)</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Year (optional)
                </label>
                <select
                  value={batchYear}
                  onChange={(e) => setBatchYear(e.target.value ? parseInt(e.target.value) : null)}
                  className="w-full border rounded px-3 py-1.5 text-sm"
                >
                  <option value="">All Years (2025-2034)</option>
                  {Array.from({ length: 10 }, (_, i) => 2025 + i).map((y) => (
                    <option key={y} value={y}>{y}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Province (optional)
                </label>
                <SearchableSelect
                  options={provinces || []}
                  value={batchProvince}
                  onChange={setBatchProvince}
                  placeholder="All Provinces"
                />
              </div>
              <div className="flex items-end">
                <button
                  onClick={handleBatchPredict}
                  disabled={batchPredicting}
                  className="w-full px-4 py-1.5 bg-emerald-600 text-white rounded hover:bg-emerald-700 transition font-medium text-sm disabled:opacity-50"
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

              {/* If filtered to a province: show year trend */}
              {batchProvince && batchByYear.length > 1 && (
                <div className="bg-white rounded-lg shadow p-4">
                  <h3 className="text-lg font-semibold text-gray-700 mb-3">
                    Predicted Yield Trend: {batchProvince}
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={batchByYear}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="year" />
                      <YAxis
                        domain={['auto', 'auto']}
                        label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip formatter={(v) => `${v.toFixed(2)} t/ha`} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="predicted_yield"
                        stroke="#059669"
                        strokeWidth={2}
                        name="Predicted Yield"
                        dot={{ r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* If filtered to a year: show province comparison */}
              {batchYear && batchByProvince.length > 1 && (
                <div className="bg-white rounded-lg shadow p-4">
                  <h3 className="text-lg font-semibold text-gray-700 mb-3">
                    Province Predictions ({batchYear})
                  </h3>
                  <div className="overflow-x-auto">
                    <ResponsiveContainer
                      width={Math.max(800, batchByProvince.length * 25)}
                      height={400}
                    >
                      <BarChart data={batchByProvince}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="province"
                          angle={-90}
                          textAnchor="end"
                          height={120}
                          tick={{ fontSize: 10 }}
                        />
                        <YAxis label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip formatter={(v) => `${v.toFixed(2)} t/ha`} />
                        <Bar dataKey="predicted_yield" name="Predicted Yield" fill="#059669" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Results Table */}
              <div className="bg-white rounded-lg shadow p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-semibold text-gray-700">
                    Detailed Results ({batchResults.count} predictions)
                  </h3>
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
                <div className="max-h-96 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-gray-50">
                      <tr>
                        <th className="text-left p-2">#</th>
                        <th className="text-left p-2">Province</th>
                        <th className="text-right p-2">Year</th>
                        <th className="text-right p-2">Predicted Yield (t/ha)</th>
                        {'actual_yield' in (batchResults.predictions[0] || {}) && (
                          <th className="text-right p-2">Actual Yield (t/ha)</th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.predictions.map((p, i) => (
                        <tr key={i} className="border-t hover:bg-gray-50">
                          <td className="p-2 text-gray-400">{i + 1}</td>
                          <td className="p-2 font-medium">{p.province}</td>
                          <td className="p-2 text-right font-mono">{p.year}</td>
                          <td className="p-2 text-right font-mono text-emerald-700">
                            {p.predicted_yield.toFixed(2)}
                          </td>
                          {p.actual_yield != null && (
                            <td className="p-2 text-right font-mono text-blue-700">
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
