#!/bin/bash
set -e

if [ "$APP_MODE" = "train" ]; then
    echo "Running fine-tuning utility..."
    python utils/finetune_deepseekcode.py
else
    echo "Running Streamlit app..."
    streamlit run rag_fabric_app.py
fi 