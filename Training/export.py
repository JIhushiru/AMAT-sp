"""
Export training results to Word document.
"""
import os
from docx import Document


def save_results_to_word(results, feature_selection_info=None, filename="model_results_with_fs.docx"):
    """Save training results and feature selection info to a Word document."""
    try:
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, filename)
        doc = Document()
        doc.add_heading('Model Training and Performance Results', level=1)

        if feature_selection_info:
            doc.add_heading('Feature Selection Summary per Fold', level=2)

            for fold_idx, fs_info in enumerate(feature_selection_info):
                doc.add_heading(f"Fold {fold_idx + 1}", level=3)

                doc.add_paragraph(f"Original features: {fs_info.get('original_count', 'N/A')}")
                doc.add_paragraph(f"Final selected features: {fs_info.get('final_count', 'N/A')}")

                selected_features = fs_info.get('selected_features', [])
                if selected_features:
                    doc.add_paragraph("Selected Features:", style="List Bullet")
                    for feature in selected_features:
                        doc.add_paragraph(f"  {feature}", style="List Bullet 2")

                removed_by_vif = fs_info.get('removed_by_vif', [])
                if removed_by_vif:
                    doc.add_paragraph("Features Removed by VIF Filtering:", style="List Bullet")
                    for item in removed_by_vif:
                        doc.add_paragraph(f"  {item['feature']} (VIF: {item['vif']:.2f})", style="List Bullet 2")

                rejected_by_boruta = fs_info.get('rejected_by_boruta', [])
                if rejected_by_boruta:
                    doc.add_paragraph("Features Rejected by Boruta:", style="List Bullet")
                    for feature in rejected_by_boruta:
                        doc.add_paragraph(f"  {feature}", style="List Bullet 2")

                doc.add_paragraph()

        for model_type, model_data in results.items():
            doc.add_heading(f"{model_type.upper()} Models", level=2)

            for model_name, metrics_data in model_data.items():
                doc.add_heading(f"{model_name} Performance", level=3)

                doc.add_paragraph(f"Best Average R2: {metrics_data.get('best_avg_r2', 'N/A'):.4f}")

                best_params = metrics_data.get('best_params', {})
                doc.add_paragraph("Best Parameters:", style="List Bullet")
                for param, value in best_params.items():
                    doc.add_paragraph(f"{param}: {value}", style="List Bullet 2")

                if 'avg_metrics' in metrics_data:
                    doc.add_paragraph("Average Metrics:", style="List Bullet")
                    for metric, value in metrics_data['avg_metrics'].items():
                        doc.add_paragraph(f"{metric}: {value:.4f}", style="List Bullet 2")

                if 'fold_r2_scores' in metrics_data:
                    doc.add_paragraph("R2 per Fold:", style="List Bullet")
                    for i, r2 in enumerate(metrics_data['fold_r2_scores'], 1):
                        doc.add_paragraph(f"Fold {i}: {r2:.4f}", style="List Bullet 2")

                doc.add_paragraph()

        doc.save(file_path)
        print(f"\nResults successfully saved to {file_path}")
        return True
    except Exception as e:
        print(f"\nError saving results: {str(e)}")
        return False
