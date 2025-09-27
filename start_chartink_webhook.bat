@echo off
echo ======================================
echo 🚀 Starting Chartink Webhook + ngrok
echo ======================================

:: Step 1 - Add ngrok authtoken (only runs once, harmless if repeated)
C:\Users\HP\OneDrive\Desktop\Shoonya\IntradayStockTrade\ngrok.exe config add-authtoken 33HdqVNQ992ftAD1kSBPzQLcFQv_4JgZ61NeWzzfrCdGEBUgW

:: Step 2 - Start Flask app
echo 🔥 Starting Flask Webhook...
start cmd /k "cd C:\Users\HP\OneDrive\Desktop\Shoonya\IntradayStockTrade && C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe chartink_webhook.py"

:: Small delay to ensure Flask starts
timeout /t 5 >nul

:: Step 3 - Start ngrok tunnel
echo 🌐 Starting ngrok tunnel on port 5000...
start cmd /k "C:\Users\HP\Downloads\ngrok.exe http 5000"

echo ======================================
echo ✅ Setup complete! Alerts will show in Flask window
echo ======================================
pause
