#!/bin/bash
set -e

echo "Post-merge setup: verifying Python dependencies..."
python -c "import streamlit, numpy, pandas, sklearn; print('All dependencies OK')"
echo "Post-merge setup complete."
