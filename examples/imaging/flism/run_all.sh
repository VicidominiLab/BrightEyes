#!/usr/bin/env bash
set -euo pipefail

mkdir -p executed

# Enable safe glob expansion:
# - nullglob: unmatched patterns disappear instead of staying literal
# - globstar: allows recursive patterns like **/*.ipynb
shopt -s nullglob globstar

while IFS= read -r pattern || [ -n "$pattern" ]; do
    # Skip empty lines and comments
    [[ -z "$pattern" || "$pattern" =~ ^[[:space:]]*# ]] && continue

    # Expand possible glob patterns such as *.ipynb or notebooks/*.ipynb
    notebooks=( $pattern )

    for notebook in "${notebooks[@]}"; do
        # Skip directories, just in case
        [[ -f "$notebook" ]] || continue
        
        echo "================================"
        echo "Running notebook: $notebook"
        echo "================================"

        name="$(basename "$notebook" .ipynb)"
        executed="executed/${name}.ipynb"

        papermill "$notebook" "$executed" --log-output

        echo "================================"
        echo "Done: $notebook"
        echo "================================"
        echo ""
        echo ""
        
    done
done <<'EOF'
# List of notebooks to run, one per line. You can use glob patterns to match multiple notebooks.
APR_Phasors.ipynb
CLSM_Phasors.ipynb
S2ISM_Phasors.ipynb

APR_Fit.ipynb
CLSM_Fit.ipynb
S2ISM_Fit.ipynb

# Run all notebooks in the current directory
#*.ipynb

# Run all notebooks recursively
# **/*.ipynb

# Run only specific notebooks
# notebook1.ipynb
# notebook2.ipynb

# Run notebooks in a specific folder
# notebooks/*.ipynb
EOF