"""
Deployment helper script for Philippine Banana Yield Prediction webapp.

Prepares files for deployment to:
  - HuggingFace Spaces (backend API with models)
  - Vercel (frontend React app)

Usage:
  python deploy.py prepare-hf     # Prepare HuggingFace Spaces directory
  python deploy.py prepare-vercel # Prepare Vercel frontend build
  python deploy.py prepare-all    # Prepare both
"""

import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DIR = Path(__file__).resolve().parent
API_DIR = WEBAPP_DIR / "api"
FRONTEND_DIR = WEBAPP_DIR / "frontend"


def prepare_huggingface(output_dir=None):
    """Prepare a directory ready to push to HuggingFace Spaces."""
    output_dir = Path(output_dir or WEBAPP_DIR / "deploy" / "huggingface")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing HuggingFace Spaces deployment in: {output_dir}")

    # Copy API source files
    for f in ["main.py", "requirements.txt", "Dockerfile"]:
        src = API_DIR / f
        if src.exists():
            shutil.copy2(src, output_dir / f)
            print(f"  Copied {f}")

    # Copy HF README as the main README
    hf_readme = API_DIR / "HF_README.md"
    if hf_readme.exists():
        shutil.copy2(hf_readme, output_dir / "README.md")
        print("  Copied README.md (from HF_README.md)")

    # Copy data files needed by the API
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    training_data = PROJECT_ROOT / "Training" / "data" / "banana_yield_2010-2024.xlsx"
    if training_data.exists():
        shutil.copy2(training_data, data_dir / "banana_yield_2010-2024.xlsx")
        print("  Copied training data")

    # Copy trained models
    models_src = PROJECT_ROOT / "Training" / "Models" / "top3"
    if models_src.exists():
        models_dst = output_dir / "models"
        if models_dst.exists():
            shutil.rmtree(models_dst)
        shutil.copytree(models_src, models_dst)
        model_count = len(list(models_dst.glob("*.joblib")))
        print(f"  Copied {model_count} trained models")
    else:
        print("  WARNING: No trained models found at Training/Models/top3/")
        print("           Run training first, then re-run this script.")

    # Copy training plots
    plots_src = PROJECT_ROOT / "Training" / "plots"
    if plots_src.exists():
        plots_dst = output_dir / "plots"
        if plots_dst.exists():
            shutil.rmtree(plots_dst)
        shutil.copytree(plots_src, plots_dst)
        print("  Copied training plots")

    # Copy mapping data
    mapping_src = PROJECT_ROOT / "Mapping"
    mapping_dst = output_dir / "mapping"
    mapping_dst.mkdir(exist_ok=True)
    geojson = mapping_src / "philippines_provinces.geojson"
    if geojson.exists():
        shutil.copy2(geojson, mapping_dst / "philippines_provinces.geojson")
        print("  Copied GeoJSON")
    mapping_xlsx = mapping_src / "banana_yield_2010-2024.xlsx"
    if mapping_xlsx.exists():
        shutil.copy2(mapping_xlsx, mapping_dst / "banana_yield_2010-2024.xlsx")
        print("  Copied mapping data")
    for png in mapping_src.glob("*.png"):
        shutil.copy2(png, mapping_dst / png.name)
    print("  Copied map images")

    # Copy SSP data
    for ssp_name in ["SSP2-4.5", "SSP5-8.5"]:
        ssp_src = PROJECT_ROOT / "SSPs Data collection" / ssp_name
        if ssp_src.exists():
            ssp_dst = output_dir / "ssp" / ssp_name
            if ssp_dst.exists():
                shutil.rmtree(ssp_dst)
            ssp_dst.mkdir(parents=True, exist_ok=True)
            for pattern in ["*.xlsx", "*.csv", "*.png"]:
                for f in ssp_src.glob(pattern):
                    shutil.copy2(f, ssp_dst / f.name)
            # Copy SHAP results
            shap_src = ssp_src / "shap_results" / "cubist"
            if shap_src.exists():
                shap_dst = ssp_dst / "shap_results" / "cubist"
                shap_dst.mkdir(parents=True, exist_ok=True)
                for f in shap_src.glob("*.png"):
                    shutil.copy2(f, shap_dst / f.name)
            print(f"  Copied {ssp_name} data")

    # Create the HuggingFace-compatible main.py that uses relative paths
    _create_hf_main(output_dir)

    print(f"\nHuggingFace Space ready at: {output_dir}")
    print("Next steps:")
    print("  1. Create a new Space on huggingface.co/spaces (SDK: Docker)")
    print("  2. Clone the space repo")
    print("  3. Copy all files from the deploy directory into the space repo")
    print("  4. git add . && git commit -m 'Deploy API' && git push")


def _create_hf_main(output_dir):
    """Create a modified main.py for HuggingFace that uses relative paths."""
    main_path = output_dir / "main.py"
    content = main_path.read_text(encoding="utf-8")

    # Replace PROJECT_ROOT-based paths with relative paths for HF
    new_paths = '''# --- Paths (HuggingFace Spaces layout) ---
PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DATA = PROJECT_ROOT / "data" / "banana_yield_2010-2024.xlsx"
TRAINING_PLOTS = PROJECT_ROOT / "plots"
MODELS_DIR = PROJECT_ROOT / "models"
MAPPING_DIR = PROJECT_ROOT / "mapping"
MAPPING_CSV = MAPPING_DIR / "banana_yield_2010-2024.xlsx"
GEOJSON_PATH = MAPPING_DIR / "philippines_provinces.geojson"
SSP245_DIR = PROJECT_ROOT / "ssp" / "SSP2-4.5"
SSP585_DIR = PROJECT_ROOT / "ssp" / "SSP5-8.5"'''

    # Find and replace the paths block
    lines = content.split('\n')
    new_lines = []
    skip = False
    for line in lines:
        if '# --- Paths' in line:
            new_lines.append(new_paths)
            skip = True
            continue
        if skip:
            if line.strip() == '' or line.startswith('SSP585_DIR'):
                if line.startswith('SSP585_DIR'):
                    pass  # already included
                skip = False
                if line.strip() == '':
                    new_lines.append('')
                continue
            continue
        new_lines.append(line)

    main_path.write_text('\n'.join(new_lines), encoding="utf-8")
    print("  Updated main.py paths for HuggingFace layout")


def prepare_vercel():
    """Build frontend and prepare for Vercel deployment."""
    print("Preparing Vercel frontend deployment...")
    print(f"Frontend directory: {FRONTEND_DIR}")

    # Check if vercel.json exists
    vercel_json = FRONTEND_DIR / "vercel.json"
    if not vercel_json.exists():
        print("  WARNING: vercel.json not found (should already be created)")

    print("\nTo deploy to Vercel:")
    print("  1. Install Vercel CLI: npm i -g vercel")
    print(f"  2. cd {FRONTEND_DIR}")
    print("  3. Set environment variable VITE_API_URL to your HuggingFace Space URL")
    print("     e.g., https://your-username-banana-yield-api.hf.space")
    print("  4. vercel --prod")
    print("\nOr connect the GitHub repo to Vercel:")
    print("  - Root Directory: webapp/frontend")
    print("  - Build Command: npm run build")
    print("  - Output Directory: dist")
    print("  - Environment Variable: VITE_API_URL = https://your-hf-space-url.hf.space")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "prepare-hf":
        prepare_huggingface()
    elif cmd == "prepare-vercel":
        prepare_vercel()
    elif cmd == "prepare-all":
        prepare_huggingface()
        print()
        prepare_vercel()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
