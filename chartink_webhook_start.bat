@echo off
echo ======================================
echo 🚀 Starting Chartink Webhook + ngrok
echo ======================================

:: ================= CONFIG ==================
:: Change SHOW_WINDOWS to 1 if you want windows visible
:: Change SHOW_WINDOWS to 0 if you want silent background
set SHOW_WINDOWS=0
:: ===========================================

:: Step 1 - Add ngrok authtoken (only runs once, harmless if repeated)
C:\Users\HP\OneDrive\Desktop\Shoonya\IntradayStockTrade\ngrok.exe config add-authtoken 33HdqVNQ992ftAD1kSBPzQLcFQv_4JgZ61NeWzzfrCdGEBUgW

if %SHOW_WINDOWS%==1 (
    echo 🔎 Running in VISIBLE mode (debug)

    :: Step 2 - Start Flask app (visible window)
    start cmd /k "cd C:\Users\HP\OneDrive\Desktop\Shoonya\IntradayStockTrade && C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe chartink_trading_bot.py"

    :: Small delay to ensure Flask starts
    timeout /t 5 >nul

    :: Step 3 - Start ngrok tunnel (visible window)
    start cmd /k "C:\Users\HP\OneDrive\Desktop\Shoonya\IntradayStockTrade\ngrok.exe" http 5000

) else (
    echo 🤫 Running in SILENT background mode

    :: Step 2 - Start Flask app silently (no window)
    start /b "" "C:\Users\HP\AppData\Local\Programs\Python\Python311\pythonw.exe" "C:\Users\HP\OneDrive\Desktop\Shoonya\IntradayStockTrade\chartink_webhook.py"

    :: Small delay to ensure Flask starts
    timeout /t 5 >nul

    :: Step 3 - Start ngrok tunnel silently
    start /b "" "C:\Users\HP\OneDrive\Desktop\Shoonya\IntradayStockTrade\ngrok.exe" http 5000 >nul 2>&1
)

:: Step 4 - Health check
echo 🩺 Checking if Flask server is running...
curl -s http://127.0.0.1:5000/health
if %errorlevel%==0 (
    echo ✅ Flask server is healthy!
) else (
    echo ❌ Flask server health check failed.
)

echo ======================================
echo ✅ Setup complete! (SHOW_WINDOWS=%SHOW_WINDOWS%)
echo ======================================
pause
