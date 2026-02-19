import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useFetch, Loader, ErrorBox, ExportButton, CollapsibleSection } from '../hooks'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis,
} from 'recharts'

const FEATURE_LABELS = {
  tmp: 'Mean Temp (\u00b0C)', tmx: 'Max Temp (\u00b0C)', tmn: 'Min Temp (\u00b0C)',
  dtr: 'Diurnal Temp Range (\u00b0C)', pre: 'Precipitation (mm)',
  pet: 'Potential ET (mm)', aet: 'Actual ET (mm)', def: 'Water Deficit (mm)',
  cld: 'Cloud Cover (%)', wet: 'Wet Days', vap: 'Vapor Pressure (kPa)',
  vpd: 'Vapor Pressure Deficit (kPa)', PDSI: 'Palmer Drought Index',
  q: 'Specific Humidity (kg/kg)', soil: 'Soil Moisture',
  srad: 'Solar Radiation (W/m\u00b2)', ws: 'Wind Speed (m/s)',
}

export default function HistoricalData() {
  const { data: summary, loading: sLoad, error: sErr, retrying, elapsed } = useFetch('/historical/summary')
  const { data: climate } = useFetch('/historical/climate-features')
  const { data: correlation } = useFetch('/historical/correlation')
  const { data: rawData } = useFetch('/historical/data')
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

  const scatterData = rawData
    ? rawData.map((r) => ({
        x: r[selectedFeature],
        y: r.yield,
        province: r.province,
      }))
    : []

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
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Historical Data Explorer</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 max-w-3xl leading-relaxed">
          Banana crop yield collected from the Philippine Statistics Authority (PSA),
          covering {summary?.total_provinces || 82} provinces from 2010 to 2024. Climate
          and environmental variables extracted from CRU-TS and TerraClimate gridded
          datasets, aggregated at the province level.
        </p>
      </div>

      <CollapsibleSection title="National Average Yield Trend">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="year" tick={{ fontSize: 12 }} />
            <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12 }} label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} />
            <Tooltip contentStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="yield" stroke="#059669" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </CollapsibleSection>

      <CollapsibleSection
        title="Climate Feature vs Yield"
        actions={
          <select
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(e.target.value)}
            className="border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1 text-sm bg-white dark:bg-gray-700 dark:text-gray-200 focus:ring-2 focus:ring-emerald-500"
          >
            {climate?.features?.map((f) => (
              <option key={f} value={f}>
                {FEATURE_LABELS[f] || f}
              </option>
            ))}
          </select>
        }
      >
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="x" type="number" tick={{ fontSize: 11 }}
              label={{ value: FEATURE_LABELS[selectedFeature] || selectedFeature, position: 'insideBottom', offset: -5, style: { fontSize: 11 } }}
            />
            <YAxis dataKey="y" type="number" tick={{ fontSize: 11 }} label={{ value: 'Yield (t/ha)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
            <ZAxis range={[20, 20]} />
            <Tooltip
              content={({ payload }) => {
                if (!payload?.[0]) return null
                const d = payload[0].payload
                return (
                  <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-md p-2 text-xs shadow-lg">
                    <p className="font-semibold text-gray-800 dark:text-gray-200">{d.province}</p>
                    <p className="text-gray-600 dark:text-gray-300">{selectedFeature}: {d.x?.toFixed(2)}</p>
                    <p className="text-gray-600 dark:text-gray-300">Yield: {d.y?.toFixed(2)} t/ha</p>
                  </div>
                )
              }}
            />
            <Scatter data={scatterData} fill="#059669" opacity={0.5} />
          </ScatterChart>
        </ResponsiveContainer>
      </CollapsibleSection>

      {yieldCorrelations.length > 0 && (
        <CollapsibleSection title="Feature Correlation with Yield" defaultOpen={false} badge={`${yieldCorrelations.length} features`}>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
            Pearson correlation coefficients between each climate variable and banana yield.
          </p>
          <div className="space-y-1.5">
            {yieldCorrelations.map(({ feature, corr }) => (
              <div key={feature} className="flex items-center gap-3">
                <span className="text-xs w-28 text-right font-mono text-gray-600 dark:text-gray-300">{feature}</span>
                <div className="flex-1 bg-gray-100 dark:bg-gray-700 rounded-full h-4 relative">
                  <div
                    className={`h-4 rounded-full ${corr >= 0 ? 'bg-emerald-500' : 'bg-red-500'}`}
                    style={{ width: `${Math.abs(corr) * 100}%` }}
                  />
                </div>
                <span className={`text-xs font-mono w-16 ${corr >= 0 ? 'text-emerald-700 dark:text-emerald-400' : 'text-red-700 dark:text-red-400'}`}>
                  {corr.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </CollapsibleSection>
      )}

      <CollapsibleSection
        title={`Province Average Yield`}
        badge={`${summary?.total_provinces} provinces`}
        actions={
          <div className="flex items-center gap-2">
            <ExportButton
              rows={provinces.map(([name, val]) => ({ Province: name, 'Avg Yield (t/ha)': val.toFixed(2) }))}
              filename="province_avg_yield.csv"
            />
            <input
              type="text"
              placeholder="Search province..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1 text-sm w-48 bg-white dark:bg-gray-700 dark:text-gray-200 focus:ring-2 focus:ring-emerald-500"
            />
          </div>
        }
      >
        <div className="max-h-96 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="text-left p-2 text-gray-500 dark:text-gray-400">#</th>
                <th className="text-left p-2 text-gray-500 dark:text-gray-400">Province</th>
                <th className="text-right p-2 text-gray-500 dark:text-gray-400">Avg Yield (t/ha)</th>
                <th className="p-2"></th>
              </tr>
            </thead>
            <tbody>
              {provinces.map(([name, val], i) => (
                <tr key={name} className="border-t border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                  <td className="p-2 text-gray-400 text-xs">{i + 1}</td>
                  <td className="p-2 font-medium text-gray-700 dark:text-gray-200">{name}</td>
                  <td className="p-2 text-right font-mono text-emerald-700 dark:text-emerald-400">
                    {val.toFixed(2)}
                  </td>
                  <td className="p-2">
                    <Link
                      to={`/province/${encodeURIComponent(name)}`}
                      className="text-emerald-600 dark:text-emerald-400 hover:underline text-xs font-medium"
                    >
                      Details
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CollapsibleSection>

      {climate && (
        <CollapsibleSection title="Climate Feature Statistics" defaultOpen={false} badge="17 features">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
            Descriptive statistics of the 17 climate and environmental variables used as predictors.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="text-left p-2 text-gray-500 dark:text-gray-400">Feature</th>
                  <th className="text-left p-2 text-gray-500 dark:text-gray-400">Description</th>
                  <th className="text-right p-2 text-gray-500 dark:text-gray-400">Mean</th>
                  <th className="text-right p-2 text-gray-500 dark:text-gray-400">Std</th>
                  <th className="text-right p-2 text-gray-500 dark:text-gray-400">Min</th>
                  <th className="text-right p-2 text-gray-500 dark:text-gray-400">Max</th>
                </tr>
              </thead>
              <tbody>
                {climate.features.map((f) => (
                  <tr key={f} className="border-t border-gray-100 dark:border-gray-700">
                    <td className="p-2 font-mono text-gray-700 dark:text-gray-200">{f}</td>
                    <td className="p-2 text-gray-500 dark:text-gray-400">{FEATURE_LABELS[f] || f}</td>
                    <td className="p-2 text-right font-mono text-gray-700 dark:text-gray-300">{climate.stats[f].mean}</td>
                    <td className="p-2 text-right font-mono text-gray-700 dark:text-gray-300">{climate.stats[f].std}</td>
                    <td className="p-2 text-right font-mono text-gray-700 dark:text-gray-300">{climate.stats[f].min}</td>
                    <td className="p-2 text-right font-mono text-gray-700 dark:text-gray-300">{climate.stats[f].max}</td>
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
