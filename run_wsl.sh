#!/bin/bash
# Script para executar no WSL onde funciona melhor

echo "🐧 Executando crawl4ai-llm-docs no WSL (recomendado)"
echo "Este script resolve problemas de encoding Unicode no Windows"
echo ""

# Verificar se o arquivo existe
if [ ! -f "$1" ]; then
    echo "❌ Arquivo não encontrado: $1"
    echo "Uso: ./run_wsl.sh arquivo_urls.txt"
    exit 1
fi

# Executar a aplicação
python3 -m crawl4ai_llm_docs "$1" "${@:2}"