import { useState, useMemo } from 'react'
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet'
import { useNavigate } from 'react-router-dom'
import { useFetch, Loader, ErrorBox, API_BASE } from '../hooks'

const YIELD_BINS = [
  { min: 0, max: 15, color: '#fde725', label: '0\u201315' },
  { min: 15, max: 30, color: '#5ec962', label: '15\u201330' },
  { min: 30, max: 46, color: '#21918c', label: '30\u201346' },
  { min: 46, max: 61, color: '#3b528b', label: '46\u201361' },
  { min: 61, max: 100, color: '#440154', label: '61\u2013100' },
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
  const [dataLayer, setDataLayer] = useState('historical')
  const [hoveredProvince, setHoveredProvince] = useState(null)
  const [showStatic, setShowStatic] = useState(false)
  const navigate = useNavigate()

  // Filter out Point features (admin centres) — keep only Polygon/MultiPolygon provinces
  const filteredGeojson = useMemo(() => {
    if (!geojson) return null
    return {
      ...geojson,
      features: geojson.features.filter(
        (f) => f.geometry?.type === 'Polygon' || f.geometry?.type === 'MultiPolygon'
      ),
    }
  }, [geojson])

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
      color: '#666',
      fillOpacity: 0.8,
    }
  }

  function onEachFeature(feature, layer) {
    const name = feature.properties.name
    const val = getYield(name)
    const layerLabel = dataLayer === 'historical'
      ? (selectedYear === 'average' ? 'Avg (2010\u20132024)' : selectedYear)
      : dataLayer === 'ssp245' ? 'SSP2-4.5 Projected' : 'SSP5-8.5 Projected'
    layer.bindTooltip(
      `<strong>${name}</strong><br/>${layerLabel}: ${val ? val.toFixed(2) + ' t/ha' : 'No data'}`,
      { sticky: true }
    )
    layer.on({
      mouseover: (e) => {
        setHoveredProvince({ name, yield: val })
        e.target.setStyle({ weight: 3, color: '#fff', fillOpacity: 0.95 })
        e.target.bringToFront()
      },
      mouseout: (e) => {
        setHoveredProvince(null)
        e.target.setStyle({ weight: 1, color: '#666', fillOpacity: 0.8 })
      },
      click: () => navigate(`/province/${encodeURIComponent(name)}`),
    })
  }

  const mapKey = `${dataLayer}-${selectedYear}`
  const ssp245Available = ssp245Data?.available
  const ssp585Available = ssp585Data?.available

  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center justify-between flex-wrap gap-3">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Yield Map</h2>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex rounded-md overflow-hidden border border-gray-300 dark:border-gray-600 text-sm">
              <button
                onClick={() => setDataLayer('historical')}
                className={`px-3 py-1.5 font-medium text-xs transition ${dataLayer === 'historical' ? 'bg-emerald-600 text-white' : 'bg-white dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-600'}`}
              >
                Historical
              </button>
              {ssp245Available && (
                <button
                  onClick={() => setDataLayer('ssp245')}
                  className={`px-3 py-1.5 font-medium text-xs border-l border-gray-300 dark:border-gray-600 transition ${dataLayer === 'ssp245' ? 'bg-blue-600 text-white' : 'bg-white dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-600'}`}
                >
                  SSP2-4.5
                </button>
              )}
              {ssp585Available && (
                <button
                  onClick={() => setDataLayer('ssp585')}
                  className={`px-3 py-1.5 font-medium text-xs border-l border-gray-300 dark:border-gray-600 transition ${dataLayer === 'ssp585' ? 'bg-red-600 text-white' : 'bg-white dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-600'}`}
                >
                  SSP5-8.5
                </button>
              )}
            </div>

            {dataLayer === 'historical' && (
              <select
                value={selectedYear}
                onChange={(e) => setSelectedYear(e.target.value)}
                className="border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1.5 text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-emerald-500"
              >
                {years.map((y) => (
                  <option key={y} value={y}>
                    {y === 'average' ? 'Average (2010\u20132024)' : y}
                  </option>
                ))}
              </select>
            )}
            <button
              onClick={() => setShowStatic(!showStatic)}
              className="bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 px-3 py-1.5 rounded-md text-xs font-medium hover:bg-gray-200 dark:hover:bg-gray-600 transition border border-gray-300 dark:border-gray-600"
            >
              {showStatic ? 'Interactive Map' : 'Static Maps'}
            </button>
          </div>
        </div>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 max-w-3xl leading-relaxed">
          Choropleth map of banana yield (t/ha) across Philippine provinces.
          Province boundaries from GADM shapefiles. Click any province for detailed data.
        </p>
      </div>

      {showStatic ? (
        <div className="space-y-6">
          {(() => {
            const layerConfig = {
              historical: { title: 'Historical (2010\u20132024)' },
              ssp245: { title: 'SSP2-4.5 Projected (2025\u20132034)' },
              ssp585: { title: 'SSP5-8.5 Projected (2025\u20132034)' },
            }
            const cfg = layerConfig[dataLayer]
            const imgs = mapImages?.filter((img) => (img.category || 'historical') === dataLayer)
            if (!imgs?.length) return (
              <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 text-yellow-800 dark:text-yellow-300">
                <p className="text-sm">No static maps available for {cfg.title}. Generate choropleth maps first.</p>
              </div>
            )
            return (
              <div>
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100 mb-3">{cfg.title}</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  {imgs.map((img) => (
                    <div key={img.name} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
                      <h4 className="font-medium text-sm text-gray-700 dark:text-gray-300 mb-2">{img.label}</h4>
                      <img
                        src={`${API_BASE}/map/image/${img.name}`}
                        alt={`${cfg.title} - ${img.label}`}
                        className="w-full rounded"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )
          })()}
        </div>
      ) : (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden relative h-100 md:h-150">
          {filteredGeojson && (
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
                data={filteredGeojson}
                style={style}
                onEachFeature={onEachFeature}
              />
            </MapContainer>
          )}

          {/* Legend */}
          <div className="absolute bottom-6 left-6 bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm rounded-lg shadow-lg p-3 z-1000 border border-gray-200 dark:border-gray-700">
            <p className="text-[10px] font-semibold mb-1.5 text-gray-700 dark:text-gray-200 uppercase tracking-wide">
              Yield (t/ha)
              {dataLayer !== 'historical' && (
                <span className="ml-1 font-normal normal-case text-gray-400 dark:text-gray-500">
                  &mdash; {dataLayer === 'ssp245' ? 'SSP2-4.5' : 'SSP5-8.5'}
                </span>
              )}
            </p>
            {YIELD_BINS.map((bin) => (
              <div key={bin.label} className="flex items-center gap-2 text-[11px] text-gray-600 dark:text-gray-300">
                <span
                  className="w-4 h-3 inline-block rounded-sm"
                  style={{ backgroundColor: bin.color }}
                />
                {bin.label}
              </div>
            ))}
            <div className="flex items-center gap-2 text-[11px] text-gray-600 dark:text-gray-300">
              <span className="w-4 h-3 inline-block rounded-sm bg-gray-300 dark:bg-gray-600" />
              No data
            </div>
          </div>

          {/* Hover info */}
          <div className={`absolute top-4 right-4 bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm rounded-lg shadow-lg p-3 z-1000 border border-gray-200 dark:border-gray-700 transition-opacity ${hoveredProvince ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
            {hoveredProvince ? (
              <>
                <p className="font-semibold text-sm text-gray-800 dark:text-gray-100">{hoveredProvince.name}</p>
                <p className="text-emerald-700 dark:text-emerald-400 font-bold text-lg">
                  {hoveredProvince.yield ? `${hoveredProvince.yield.toFixed(2)} t/ha` : 'No data'}
                </p>
                <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-0.5">Click for details</p>
              </>
            ) : (
              <p className="text-xs text-gray-400 dark:text-gray-500">Hover over a province</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
