import { useFetch, Loader, ErrorBox, EmptyState, API_BASE } from '../hooks'

export default function ModelResults() {
  const { data: plots, loading, error, retrying, elapsed } = useFetch('/training/plots')

  if (loading) return <Loader retrying={retrying} elapsed={elapsed} />
  if (error) return <ErrorBox message={error} />

  if (!plots || plots.length === 0) {
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Model Training Results</h2>
        <EmptyState
          title="No training plots found"
          message="Run python regression.py in the Training directory to generate model evaluation results."
        />
      </div>
    )
  }

  const DESCRIPTIONS = {
    'regression_r2_comparison_with_fs': 'R\u00b2 Score: Proportion of variance in yield explained by each model. Higher values indicate better fit (max 1.0).',
    'regression_rmse_comparison_with_fs': 'RMSE: Root Mean Squared Error in the same units as the target variable (t/ha). Lower values indicate better accuracy.',
    'regression_mse_comparison_with_fs': 'MSE: Mean Squared Error. Penalizes large prediction errors due to squaring.',
    'regression_mae_comparison_with_fs': 'MAE: Mean Absolute Error. Average magnitude of prediction errors in t/ha.',
    'regression_mape_comparison_with_fs': 'MAPE: Mean Absolute Percentage Error. Scale-independent error metric expressed as a percentage.',
    'model_performance_heatmap': 'Normalized heatmap comparing all models across all metrics for visual comparison.',
    'normalized_error_metrics_with_fs': 'Normalized error metrics comparison across all models.',
  }

  const MODELS = [
    {
      name: 'Cubist',
      desc: 'Rule-based algorithm combining regression trees with linear models at each terminal node. Uses committees (ensemble averaging) for improved predictions. Selected as the best model for future projections.',
    },
    {
      name: 'GBM',
      desc: 'Gradient Boosting Machine. Builds models sequentially, where each new tree corrects the residual errors of the previous one through minimization of a loss function.',
    },
    {
      name: 'MARS',
      desc: 'Multivariate Adaptive Regression Splines. Non-parametric technique that models non-linearities using piecewise linear basis functions with automatically determined knots.',
    },
    {
      name: 'Random Forest',
      desc: 'Ensemble method that builds a forest of decision trees trained on random subsets of data (bagging) and features. Predictions are averaged across all trees.',
    },
    {
      name: 'SVM',
      desc: 'Support Vector Regression with RBF kernel. Maps features into higher-dimensional space and fits an epsilon-tube around the regression surface.',
    },
    {
      name: 'XGBoost',
      desc: 'Extreme Gradient Boosting. Sequential ensemble of decision trees with L1/L2 regularization to prevent overfitting. Known for speed and performance on structured data.',
    },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">Model Training Results</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 max-w-3xl leading-relaxed">
          Six regression models trained with Time Series 5-fold cross-validation
          (temporal order preserved) and feature selection via Boruta algorithm and
          VIF filtering (threshold = 5.0). Temperature rule applied: mean temperature
          (tmp) forced; tmn, tmx, and dtr excluded.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        {MODELS.map((m) => (
          <div key={m.name} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h4 className="font-semibold text-gray-800 dark:text-gray-100">{m.name}</h4>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1.5 leading-relaxed">{m.desc}</p>
          </div>
        ))}
      </div>

      {plots.map((plot) => (
        <div key={plot.name} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
          <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200 mb-1">{plot.label}</h3>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
            {DESCRIPTIONS[plot.name.replace('.png', '')] || ''}
          </p>
          <img
            src={`${API_BASE}/training/plot/${plot.name}`}
            alt={plot.label}
            className="w-full max-w-4xl mx-auto rounded"
            loading="lazy"
          />
        </div>
      ))}

      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5">
        <h3 className="text-base font-semibold text-gray-800 dark:text-gray-200 mb-4">Methodology</h3>
        <div className="grid md:grid-cols-2 gap-6 text-sm text-gray-600 dark:text-gray-400">
          <div>
            <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2 text-xs uppercase tracking-wide">Feature Selection</h4>
            <ul className="list-disc ml-5 space-y-1.5 text-xs leading-relaxed">
              <li>VIF filtering (threshold = 5.0) to address multicollinearity, with mean temperature (tmp) protected</li>
              <li>Boruta algorithm (1,000 estimators, 50 iterations) as a wrapper around Random Forest</li>
              <li>Temperature rule: tmp forced as predictor; tmn, tmx, and dtr excluded to avoid redundancy</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2 text-xs uppercase tracking-wide">Cross-Validation</h4>
            <ul className="list-disc ml-5 space-y-1.5 text-xs leading-relaxed">
              <li>TimeSeriesSplit with 5 folds (temporal ordering preserved across province-year pairs)</li>
              <li>Grid search over all hyperparameter combinations for each model</li>
              <li>Evaluation metrics: R&sup2;, RMSE, MSE, MAE, MAPE</li>
              <li>Baseline comparison using province-level historical mean yield</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
