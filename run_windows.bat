@echo off
REM Script para executar no Windows com encoding UTF-8
echo Running crawl4ai-llm-docs no Windows com UTF-8 encoding
echo Este script resolve problemas de encoding Unicode
echo.

REM Verificar se o arquivo existe
if not exist "%~1" (
    echo Arquivo nao encontrado: %~1
    echo Uso: run_windows.bat arquivo_urls.txt
    exit /b 1
)

REM Configurar encoding UTF-8
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

REM Executar a aplicacao
python -m src.crawl4ai_llm_docs "%~1" %2 %3 %4 %5