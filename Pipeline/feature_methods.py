"""
Feature Computation Methods for SSP Climate Projections
========================================================

This module documents how each of the 17 climate features is computed
for future SSP scenario predictions (2025-2034). Features are derived
from CMIP6 SSP projections (which provide only precipitation and mean
temperature) combined with historical TerraClimate data (2010-2024).

The delta change method is used to perturb historical baselines with
projected climate shifts, preserving internal feature consistency.

References
----------
[1] Hay, L. E., Wilby, R. L., & Leavesley, G. H. (2000). A comparison
    of delta change and downscaled GCM scenarios for three mountainous
    basins in the United States. Journal of the American Water Resources
    Association, 36(2), 387-397.
    https://doi.org/10.1111/j.1752-1688.2000.tb04276.x

[2] Anandhi, A., Frei, A., Pierson, D. C., Schneiderman, E. M.,
    Zion, M. S., Lounsbury, D., & Matonse, A. H. (2011). Examination
    of change factor methodologies for climate change impact assessment.
    Water Resources Research, 47(3).
    https://doi.org/10.1029/2010WR009104

[3] Tetens, O. (1930). Uber einige meteorologische Begriffe.
    Zeitschrift fur Geophysik, 6, 297-309.

[4] Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998).
    Crop evapotranspiration: Guidelines for computing crop water
    requirements. FAO Irrigation and Drainage Paper No. 56. Rome: FAO.

[5] Grossiord, C., Buckley, T. N., Cernusak, L. A., Novick, K. A.,
    Poulter, B., Siegwolf, R. T. W., Sperry, J. S., & McDowell, N. G.
    (2020). Plant responses to rising vapor pressure deficit. New
    Phytologist, 226(6), 1550-1566.
    https://doi.org/10.1111/nph.16485

[6] Yuan, W., Zheng, Y., Piao, S., et al. (2019). Increased atmospheric
    vapor pressure deficit reduces global vegetation growth. Science
    Advances, 5(8), eaax1396.
    https://doi.org/10.1126/sciadv.aax1396

[7] Lobell, D. B., & Burke, M. B. (2010). On the use of statistical
    models to predict crop yield responses to climate change. Agricultural
    and Forest Meteorology, 150(11), 1443-1452.
    https://doi.org/10.1016/j.agrformet.2010.07.008

[8] Challinor, A. J., Watson, J., Lobell, D. B., Howden, S. M.,
    Smith, D. R., & Chhetri, N. (2014). A meta-analysis of crop yield
    under climate change and adaptation. Nature Climate Change, 4,
    287-291.
    https://doi.org/10.1038/nclimate2153
"""

import numpy as np


# ---------------------------------------------------------------------------
# Feature source categories
# ---------------------------------------------------------------------------

# Features directly from CMIP6 SSP projections
SSP_DIRECT = ['tmp', 'pre']

# Features derived from SSP temperature using physical equations
SSP_DERIVED = ['tmx', 'tmn', 'dtr', 'vpd', 'pet']

# Features held at historical province averages (no SSP projection available)
HISTORICAL_STATIC = ['cld', 'wet', 'vap', 'aet', 'def', 'PDSI', 'q', 'soil', 'srad', 'ws']


# ---------------------------------------------------------------------------
# tmp — Mean Temperature (°C)
# ---------------------------------------------------------------------------
def compute_tmp(tmp_ssp):
    """
    Source: CMIP6 SSP projections (direct).

    Mean temperature is taken directly from the downscaled CMIP6 SSP
    scenario output for each province and year.

    No transformation is applied; values are used as provided by the
    climate model ensemble.
    """
    return tmp_ssp


# ---------------------------------------------------------------------------
# pre — Precipitation (mm/month)
# ---------------------------------------------------------------------------
def compute_pre(pre_ssp):
    """
    Source: CMIP6 SSP projections (direct).

    Monthly mean precipitation is taken directly from the downscaled
    CMIP6 SSP scenario output for each province and year.

    No transformation is applied; values are used as provided by the
    climate model ensemble.
    """
    return pre_ssp


# ---------------------------------------------------------------------------
# tmx — Maximum Temperature (°C)
# ---------------------------------------------------------------------------
def compute_tmx(tmx_hist_avg, tmp_ssp, tmp_hist_avg):
    """
    Source: Delta change method applied to historical baseline [1, 2].

    The SSP temperature anomaly (delta) is computed as the difference
    between the SSP-projected mean temperature and the historical
    province average. This delta is then added to the historical
    average maximum temperature:

        delta_T = tmp_ssp - tmp_hist_avg
        tmx_new = tmx_hist_avg + delta_T

    This assumes that the diurnal temperature asymmetry remains
    approximately constant under near-term climate change, which is
    supported by observations for tropical regions over decadal
    timescales.

    References: [1] Hay et al. (2000), [2] Anandhi et al. (2011)
    """
    delta_t = tmp_ssp - tmp_hist_avg
    return tmx_hist_avg + delta_t


# ---------------------------------------------------------------------------
# tmn — Minimum Temperature (°C)
# ---------------------------------------------------------------------------
def compute_tmn(tmn_hist_avg, tmp_ssp, tmp_hist_avg):
    """
    Source: Delta change method applied to historical baseline [1, 2].

    Same approach as tmx. The SSP temperature anomaly is added to the
    historical average minimum temperature:

        delta_T = tmp_ssp - tmp_hist_avg
        tmn_new = tmn_hist_avg + delta_T

    References: [1] Hay et al. (2000), [2] Anandhi et al. (2011)
    """
    delta_t = tmp_ssp - tmp_hist_avg
    return tmn_hist_avg + delta_t


# ---------------------------------------------------------------------------
# dtr — Diurnal Temperature Range (°C)
# ---------------------------------------------------------------------------
def compute_dtr(tmx_new, tmn_new):
    """
    Source: Derived from adjusted tmx and tmn.

    Diurnal temperature range is computed as:

        dtr = tmx - tmn

    Since both tmx and tmn are shifted by the same delta, dtr remains
    approximately equal to its historical value. This is consistent
    with the assumption that near-term climate change primarily shifts
    the temperature distribution rather than altering the diurnal cycle.

    References: [1] Hay et al. (2000)
    """
    return tmx_new - tmn_new


# ---------------------------------------------------------------------------
# vpd — Vapor Pressure Deficit (kPa)
# ---------------------------------------------------------------------------
def compute_vpd(vpd_hist_avg, tmp_ssp, tmp_hist_avg):
    """
    Source: Scaled from historical using the Tetens equation [3, 4].

    VPD is the difference between saturation vapor pressure (es) and
    actual vapor pressure (ea). Since es increases exponentially with
    temperature (Clausius-Clapeyron relation), VPD is expected to
    increase under warming if relative humidity remains approximately
    constant.

    The Tetens formula for saturation vapor pressure (kPa):

        es(T) = 0.6108 * exp(17.27 * T / (T + 237.3))

    VPD is scaled proportionally:

        vpd_new = vpd_hist * (es(tmp_ssp) / es(tmp_hist_avg))

    This approach assumes that actual vapor pressure (ea) remains
    near its historical level, which is reasonable for near-term
    projections where moisture sources remain similar.

    Rising VPD increases atmospheric water demand and is a key driver
    of crop water stress, particularly in tropical systems [5, 6].

    References: [3] Tetens (1930), [4] Allen et al. (1998),
                [5] Grossiord et al. (2020), [6] Yuan et al. (2019)
    """
    es_new = 0.6108 * np.exp(17.27 * tmp_ssp / (tmp_ssp + 237.3))
    es_hist = 0.6108 * np.exp(17.27 * tmp_hist_avg / (tmp_hist_avg + 237.3))
    return vpd_hist_avg * (es_new / es_hist)


# ---------------------------------------------------------------------------
# pet — Potential Evapotranspiration (mm/day)
# ---------------------------------------------------------------------------
def compute_pet(pet_hist_avg, tmp_ssp, tmp_hist_avg):
    """
    Source: Scaled from historical using saturation vapor pressure ratio [4].

    Potential evapotranspiration is driven by atmospheric evaporative
    demand, which increases with temperature through the saturation
    vapor pressure. PET is scaled using the same Tetens-based ratio
    as VPD:

        pet_new = pet_hist * (es(tmp_ssp) / es(tmp_hist_avg))

    This is a simplified scaling consistent with temperature-based
    PET estimation methods (e.g., Hargreaves, Thornthwaite), where
    PET is primarily a function of temperature and available energy.

    Reference: [4] Allen et al. (1998)
    """
    es_new = 0.6108 * np.exp(17.27 * tmp_ssp / (tmp_ssp + 237.3))
    es_hist = 0.6108 * np.exp(17.27 * tmp_hist_avg / (tmp_hist_avg + 237.3))
    return pet_hist_avg * (es_new / es_hist)


# ---------------------------------------------------------------------------
# cld — Cloud Cover (%)
# ---------------------------------------------------------------------------
def compute_cld(cld_hist_avg):
    """
    Source: Historical province average (static).

    Cloud cover is held at its historical (2010-2024) province-level
    average. CMIP6 SSP projections for this variable were not available
    at the required spatial resolution. Cloud cover changes under
    near-term climate change (2025-2034) are expected to be small
    relative to natural variability in the Philippines.
    """
    return cld_hist_avg


# ---------------------------------------------------------------------------
# wet — Wet Days (days/month)
# ---------------------------------------------------------------------------
def compute_wet(wet_hist_avg):
    """
    Source: Historical province average (static).

    Number of wet days is held at its historical province-level average.
    While precipitation amount (pre) is adjusted via SSP projections,
    the frequency distribution of rainfall events is not available from
    the CMIP6 outputs used. This is a known limitation.
    """
    return wet_hist_avg


# ---------------------------------------------------------------------------
# vap — Actual Vapor Pressure (kPa)
# ---------------------------------------------------------------------------
def compute_vap(vap_hist_avg):
    """
    Source: Historical province average (static).

    Actual vapor pressure is held at its historical province-level
    average. This is consistent with the VPD computation, where the
    increase in VPD under warming is driven entirely by the rise in
    saturation vapor pressure (es) while actual vapor pressure (ea)
    remains approximately constant.

    Reference: [4] Allen et al. (1998)
    """
    return vap_hist_avg


# ---------------------------------------------------------------------------
# aet — Actual Evapotranspiration (mm/month)
# ---------------------------------------------------------------------------
def compute_aet(aet_hist_avg):
    """
    Source: Historical province average (static).

    Actual evapotranspiration depends on available water, vegetation,
    and atmospheric demand. Without coupled hydrological modeling under
    SSP scenarios, AET is held at its historical province-level average.
    This is a limitation, as AET may change with altered precipitation
    patterns and increased PET.
    """
    return aet_hist_avg


# ---------------------------------------------------------------------------
# def_ — Climatic Water Deficit (mm/month)
# ---------------------------------------------------------------------------
def compute_def(def_hist_avg):
    """
    Source: Historical province average (static).

    Water deficit (PET - AET) is held at its historical average. In
    principle, deficit should increase with rising PET under warming.
    However, without SSP-projected AET, a consistent computation is
    not possible. This is a known limitation.
    """
    return def_hist_avg


# ---------------------------------------------------------------------------
# PDSI — Palmer Drought Severity Index
# ---------------------------------------------------------------------------
def compute_pdsi(pdsi_hist_avg):
    """
    Source: Historical province average (static).

    PDSI is a standardized drought index that integrates temperature,
    precipitation, and soil moisture. Computing future PDSI requires
    a full water balance model under SSP scenarios. Without this, PDSI
    is held at its historical province-level average. This is a
    limitation for provinces where drought patterns may shift.
    """
    return pdsi_hist_avg


# ---------------------------------------------------------------------------
# q — Runoff (mm/month)
# ---------------------------------------------------------------------------
def compute_q(q_hist_avg):
    """
    Source: Historical province average (static).

    Runoff depends on precipitation intensity, soil properties, and
    land use. Without hydrological modeling under SSP scenarios, runoff
    is held at its historical province-level average.
    """
    return q_hist_avg


# ---------------------------------------------------------------------------
# soil — Soil Moisture (mm)
# ---------------------------------------------------------------------------
def compute_soil(soil_hist_avg):
    """
    Source: Historical province average (static).

    Soil moisture is held at its historical province-level average.
    Future soil moisture depends on the balance between precipitation,
    evapotranspiration, and runoff, requiring coupled modeling not
    available in this study.
    """
    return soil_hist_avg


# ---------------------------------------------------------------------------
# srad — Solar Radiation (W/m²)
# ---------------------------------------------------------------------------
def compute_srad(srad_hist_avg):
    """
    Source: Historical province average (static).

    Downward shortwave radiation is primarily determined by latitude,
    season, and cloud cover. Since cloud cover is held constant and
    the projection period is near-term (10 years), solar radiation
    is held at its historical province-level average.
    """
    return srad_hist_avg


# ---------------------------------------------------------------------------
# ws — Wind Speed (m/s)
# ---------------------------------------------------------------------------
def compute_ws(ws_hist_avg):
    """
    Source: Historical province average (static).

    Wind speed is held at its historical province-level average.
    Near-term changes in wind patterns under SSP scenarios are
    expected to be small relative to natural interannual variability.
    """
    return ws_hist_avg


# ---------------------------------------------------------------------------
# Prediction Clipping
# ---------------------------------------------------------------------------
def clip_predictions(predictions, province_labels, hist_province_bounds):
    """
    Clip predicted yields to each province's historical observed range.

    Statistical crop models trained on historical data can produce
    unreliable predictions when input features fall outside the
    training distribution [7, 8]. This is particularly relevant for
    climate change projections where feature combinations may be
    novel relative to the training period.

    For each province, predictions are clipped to:

        yield_clipped = clip(yield_pred, province_hist_min, province_hist_max)

    This conservative approach ensures that projected yields remain
    within the historically observed envelope for each province,
    preventing extrapolation artifacts from inflating or deflating
    national-level aggregates.

    References: [7] Lobell & Burke (2010), [8] Challinor et al. (2014)
    """
    clipped = predictions.copy()
    for province in np.unique(province_labels):
        if province in hist_province_bounds.index:
            pmin = hist_province_bounds.loc[province, 'min']
            pmax = hist_province_bounds.loc[province, 'max']
            mask = province_labels == province
            clipped[mask] = np.clip(predictions[mask], pmin, pmax)
    return clipped
