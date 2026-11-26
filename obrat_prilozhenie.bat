@echo off
chcp 65001 >nul
echo === Сборка приложения "Моя ИИ" ===
echo.
python --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python не найден. Установи его с python.org
    echo     и при установке поставь галочку "Add Python to PATH".
    pause
    exit /b
)
echo Устанавливаю сборщик (PyInstaller)...
python -m pip install --upgrade pyinstaller
echo.
echo Собираю приложение...
python -m PyInstaller --onefile --noconsole --name "MoyaII" moya_ii.py
if not exist modeli mkdir modeli
if exist "dist\MoyaII.exe" (
    copy /y "dist\MoyaII.exe" "MoyaII.exe" >nul
    rmdir /s /q build
    rmdir /s /q dist
    del /q MoyaII.spec
    echo.
    echo === Готово! Запускай MoyaII.exe ===
) else (
    echo [!] Что-то пошло не так, пришли текст ошибки выше.
)
pause
