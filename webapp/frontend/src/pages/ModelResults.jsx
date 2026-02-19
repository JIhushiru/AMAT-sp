import { useFetch, Loader, ErrorBox, API_BASE } from '../hooks'

export default function ModelResults() {
  const { data: plots, loading, error } = useFetch('/training/plots')

  if (loading) return <Loader />
  if (error) return <ErrorBox message={error} />

  if (!plots || plots.length === 0) {
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-gray-800">Model Results</h2>
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-yellow-800">
          No training plots found. Run <code>python regression.py</code> in the
          Training directory to generate results.
        </div>
      </div>
    )
  }

  const DESCRIPTIONS = {
    'regression_r2_comparison_with_fs': 'R² Score: Measures how well each model explains the variance in yield data. Higher is better (max 1.0).',
    'regression_rmse_comparison_with_fs': 'RMSE: Root Mean Squared Error. Lower values indicate better prediction accuracy.',
    'regression_mse_comparison_with_fs': 'MSE: Mean Squared Error. The squared average of prediction errors.',
    'regression_mae_comparison_with_fs': 'MAE: Mean Absolute Error. The average magnitude of prediction errors.',
    'model_performance_heatmap': 'Normalized heatmap comparing all models across all metrics. Allows quick visual comparison.',
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Model Training Results</h2>
      <p className="text-gray-600">
        Six regression models were trained with Time Series 5-fold cross-validation
        and Boruta + VIF feature selection: <strong>Cubist, GBM, MARS, RF, SVM, XGBoost</strong>.
      </p>

      {/* Model descriptions */}
      <div className="grid md:grid-cols-3 gap-4">
        {[
          { name: 'Cubist', desc: 'Rule-based regression with committees. Used for future SSP predictions.' },
          { name: 'Random Forest', desc: 'Ensemble of decision trees with bagging.' },
          { name: 'XGBoost', desc: 'Gradient-boosted trees with regularization.' },
          { name: 'GBM', desc: 'Sequential tree boosting for residual correction.' },
          { name: 'SVM', desc: 'Support Vector Regression with kernel functions.' },
          { name: 'MARS', desc: 'Multivariate Adaptive Regression Splines (R earth).' },
        ].map((m) => (
          <div key={m.name} className="bg-white rounded-lg shadow p-3">
            <h4 className="font-semibold text-emerald-700">{m.name}</h4>
            <p className="text-xs text-gray-600 mt-1">{m.desc}</p>
          </div>
        ))}
      </div>

      {/* Plots */}
      {plots.map((plot) => (
        <div key={plot.name} className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-700 mb-1">{plot.label}</h3>
          <p className="text-sm text-gray-500 mb-3">
            {DESCRIPTIONS[plot.name.replace('.png', '')] || ''}
          </p>
          <img
            src={`${API_BASE}/training/plot/${plot.name}`}
            alt={plot.label}
            className="w-full max-w-4xl mx-auto rounded"
          />
        </div>
      ))}

      {/* Methodology */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold text-gray-700 mb-3">Methodology</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-600">
          <div>
            <h4 className="font-semibold text-gray-800 mb-1">Feature Selection</h4>
            <ul className="list-disc ml-5 space-y-1">
              <li>VIF filtering (threshold = 5.0, tmp protected)</li>
              <li>Boruta selection (1000 estimators, 50 iterations)</li>
              <li>Temperature rule: tmp forced, tmn/tmx/dtr excluded</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-gray-800 mb-1">Cross-Validation</h4>
            <ul className="list-disc ml-5 space-y-1">
              <li>Time Series 5-fold split (temporal order preserved)</li>
              <li>Grid search over all parameter combinations</li>
              <li>Metrics: R², RMSE, MSE, MAE, MAPE</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
