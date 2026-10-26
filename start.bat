@echo off
chcp 65001 >nul
echo ========================================================
echo   RVK Safety CV — Запуск системы видеоаналитики ТБ
echo   ООО «РВК.Цифровые решения» (ГК «Росводоканал»)
echo ========================================================
echo.

if not exist .venv (
    echo [INFO] Виртуальное окружение не найдено. Создаем .venv...
    python -m venv .venv
    call .venv\Scripts\activate
    echo [INFO] Установка зависимостей из requirements.txt...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)

set PORT=8088
echo [INFO] Запуск веб-сервера на http://localhost:8088 ...
python backend/app.py
pause
