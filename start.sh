#!/bin/bash
echo "========================================================"
echo "  RVK Safety CV — Запуск системы видеоаналитики ТБ"
echo "  ООО «РВК.Цифровые решения» (ГК «Росводоканал»)"
echo "========================================================"

if [ ! -d ".venv" ]; then
    echo "[INFO] Создаем виртуальное окружение .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

export PORT=8088
echo "[INFO] Запуск веб-сервера на http://localhost:8088 ..."
python3 backend/app.py
