#!/bin/bash
# Biomedical Knowledge Graph Q&A System Startup

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate knowledge_graph
export PATH=~/ollama-local/bin:$PATH

# Start Ollama if needed
if ! ollama list >/dev/null 2>&1; then
    nohup ollama serve > ~/ollama-local/server.log 2>&1 &
    sleep 5
fi

python -m src.llm_integration.qa_system