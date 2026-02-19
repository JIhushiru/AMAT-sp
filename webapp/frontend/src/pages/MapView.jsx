import { useState } from 'react'
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet'
import { useNavigate } from 'react-router-dom'
import { useFetch, Loader, ErrorBox, API_BASE } from '../hooks'

const YIELD_BINS = [
  { min: 0, max: 15, color: '#fde725', label: '0-15' },
  { min: 15, max: 30, color: '#5ec962', label: '15-30' },
  { min: 30, max: 46, color: '#21918c', label: '30-46' },
  { min: 46, max: 61, color: '#3b528b', label: '46-61' },
  { min: 61, max: 100, color: '#440154', label: '61-100' },
]

function getColor(val) {
  if (!val || val === 0) return '#D3D3D3'
  for (const bin of YIELD_BINS) {
    if (val > bin.min && val <= bin.max) return bin.color
  }
  return '#440154'
}

export default function MapView() {
  const { data: geojson, loading: gLoad, error: gErr, retrying, elapsed } = useFetch('/map/geojson')
  const { data: yieldData, loading: yLoad, error: yErr } = useFetch('/map/yield-by-province')
  const { data: mapImages } = useFetch('/map/images')
  const { data: ssp245Data } = useFetch('/ssp/ssp245')
  const { data: ssp585Data } = useFetch('/ssp/ssp585')

  const [selectedYear, setSelectedYear] = useState('average')
  const [dataLayer, setDataLayer] = useState('historical') // 'historical', 'ssp245', 'ssp585'
  const [hoveredProvince, setHoveredProvince] = useState(null)
  const [showStatic, setShowStatic] = useState(false)
  const navigate = useNavigate()

  if (gLoad || yLoad) return <Loader retrying={retrying} elapsed={elapsed} />
  if (gErr) return <ErrorBox message={gErr} />
  if (yErr) return <ErrorBox message={yErr} />

  const years = ['average']
  if (yieldData) {
    const allYears = new Set()
    Object.values(yieldData).forEach((prov) => {
      if (prov.yearly) Object.keys(prov.yearly).forEach((y) => allYears.add(y))
    })
    years.push(...[...allYears].sort())
  }

  // Build SSP province averages from province_summary data
  function getSSPYield(provinceName, sspData) {
    if (!sspData?.province_summary) return 0
    const entry = sspData.province_summary[provinceName]
    if (entry) return entry['Future Avg (2025\u20132034)'] || 0
    return 0
  }

  function getYield(provinceName) {
    if (dataLayer === 'ssp245') return getSSPYield(provinceName, ssp245Data)
    if (dataLayer === 'ssp585') return getSSPYield(provinceName, ssp585Data)
    if (!yieldData) return 0
    const names = [provinceName, provinceName?.replace(/\s+/g, ' ').trim()]
    for (const n of names) {
      if (yieldData[n]) {
        if (selectedYear === 'average') return yieldData[n].average
        return yieldData[n].yearly?.[selectedYear] || 0
      }
    }
    return 0
  }

  function style(feature) {
    const name = feature.properties.name
    const val = getYield(name)
    return {
      fillColor: getColor(val),
      weight: 1,
      opacity: 1,
      color: '#333',
      fillOpacity: 0.8,
    }
  }

  function onEachFeature(feature, layer) {
    const name = feature.properties.name
    const val = getYield(name)
    const layerLabel = dataLayer === 'historical'
      ? (selectedYear === 'average' ? 'Avg (2010-2024)' : selectedYear)
      : dataLayer === 'ssp245' ? 'SSP2-4.5 Projected' : 'SSP5-8.5 Projected'
    layer.bindTooltip(
      `<strong>${name}</strong><br/>${layerLabel}: ${val ? val.toFixed(2) + ' t/ha' : 'No data'}`,
      { sticky: true }
    )
    layer.on({
      mouseover: () => setHoveredProvince({ name, yield: val }),
      mouseout: () => setHoveredProvince(null),
      click: () => navigate(`/province/${encodeURIComponent(name)}`),
    })
  }

  const mapKey = `${dataLayer}-${selectedYear}`
  const ssp245Available = ssp245Data?.available
  const ssp585Available = ssp585Data?.available

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h2 className="text-2xl font-bold text-gray-800">Yield Map</h2>
        <div className="flex items-center gap-3 flex-wrap">
          {/* Data layer selector */}
          <div className="flex rounded overflow-hidden border text-sm">
            <button
              onClick={() => setDataLayer('historical')}
              className={`px-3 py-1.5 ${dataLayer === 'historical' ? 'bg-emerald-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-100'}`}
            >
              Historical
            </button>
            {ssp245Available && (
              <button
                onClick={() => setDataLayer('ssp245')}
                className={`px-3 py-1.5 border-l ${dataLayer === 'ssp245' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-100'}`}
              >
                SSP2-4.5
              </button>
            )}
            {ssp585Available && (
              <button
                onClick={() => setDataLayer('ssp585')}
                className={`px-3 py-1.5 border-l ${dataLayer === 'ssp585' ? 'bg-red-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-100'}`}
              >
                SSP5-8.5
              </button>
            )}
          </div>

          {dataLayer === 'historical' && (
            <select
              value={selectedYear}
              onChange={(e) => setSelectedYear(e.target.value)}
              className="border rounded px-3 py-1.5 text-sm"
            >
              {years.map((y) => (
                <option key={y} value={y}>
                  {y === 'average' ? 'Average (2010-2024)' : y}
                </option>
              ))}
            </select>
          )}
          <button
            onClick={() => setShowStatic(!showStatic)}
            className="bg-emerald-600 text-white px-3 py-1.5 rounded text-sm hover:bg-emerald-700"
          >
            {showStatic ? 'Interactive Map' : 'Static Maps'}
          </button>
        </div>
      </div>

      {showStatic ? (
        <div className="grid md:grid-cols-2 gap-4">
          {mapImages?.map((img) => (
            <div key={img.name} className="bg-white rounded-lg shadow p-3">
              <h3 className="font-semibold text-gray-700 mb-2">{img.label}</h3>
              <img
                src={`${API_BASE}/map/image/${img.name}`}
                alt={img.label}
                className="w-full rounded"
              />
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden relative h-100 md:h-150">
          {geojson && (
            <MapContainer
              center={[12.5, 122]}
              zoom={6}
              scrollWheelZoom={true}
              style={{ height: '100%', width: '100%' }}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org">OSM</a>'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <GeoJSON
                key={mapKey}
                data={geojson}
                style={style}
                onEachFeature={onEachFeature}
              />
            </MapContainer>
          )}

          {/* Legend */}
          <div className="absolute bottom-6 left-6 bg-white rounded-lg shadow-lg p-3 z-[1000]">
            <p className="text-xs font-semibold mb-1.5">
              Yield (t/ha)
              {dataLayer !== 'historical' && (
                <span className="ml-1 font-normal text-gray-400">
                  — {dataLayer === 'ssp245' ? 'SSP2-4.5' : 'SSP5-8.5'}
                </span>
              )}
            </p>
            {YIELD_BINS.map((bin) => (
              <div key={bin.label} className="flex items-center gap-2 text-xs">
                <span
                  className="w-4 h-3 inline-block rounded-sm"
                  style={{ backgroundColor: bin.color }}
                />
                {bin.label}
              </div>
            ))}
            <div className="flex items-center gap-2 text-xs">
              <span className="w-4 h-3 inline-block rounded-sm bg-gray-300" />
              No data
            </div>
          </div>
        </div>
      )}

      {hoveredProvince && (
        <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg p-3 z-50 border">
          <p className="font-semibold">{hoveredProvince.name}</p>
          <p className="text-emerald-700 font-bold">
            {hoveredProvince.yield
              ? `${hoveredProvince.yield.toFixed(2)} t/ha`
              : 'No data'}
          </p>
        </div>
      )}
    </div>
  )
}
