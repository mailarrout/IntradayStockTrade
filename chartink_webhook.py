from flask import Flask, request, jsonify
from win10toast import ToastNotifier
import threading
import queue
import datetime
import time
import csv
import os
import pandas as pd
import pyotp
from api_helper import ShoonyaApiPy
from datetime import datetime

app = Flask(__name__)
toaster = ToastNotifier()

# Global broker client
broker_client = None
client_loaded = False

# Queue for alerts
alert_queue = queue.Queue()

# CSV file for alerts
csv_file = "app\Date_Chartink_rsi-cross-with-yesterday-high-break-one-min.csv"

# Ensure CSV file has headers
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Alert Name", "Stocks", "Trigger Prices", "Order Status"])

def load_broker_client():
    """Load broker client from ClientInfo.txt"""
    global broker_client, client_loaded
    
    try:
        print("Loading broker client...")
        
        if not os.path.exists("ClientInfo.txt"):
            print("❌ ClientInfo.txt not found")
            return False

        df = pd.read_csv("ClientInfo.txt")
        if df.empty:
            print("❌ ClientInfo.txt is empty")
            return False

        # Use first client from the file
        row = df.iloc[0]
        client_id = row["Client ID"]
        
        print(f"Logging in client: {client_id}")
        
        # Generate 2FA
        twoFA = pyotp.TOTP(row["token"]).now()
        
        # Create and login client
        broker_client = ShoonyaApiPy()
        
        ret = broker_client.login(
            userid=row["Client ID"],
            password=row["Password"],
            twoFA=twoFA,
            vendor_code=row["vc"],
            api_secret=row["app_key"],
            imei=row["imei"]
        )

        if ret and ret.get('stat') == 'Ok':
            client_loaded = True
            print("✅ Broker login successful")
            return True
        else:
            error_msg = ret.get('emsg', 'Unknown error') if ret else 'No response'
            print(f"❌ Login failed: {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading broker client: {e}")
        return False

def place_simple_buy_order(stock, trigger_price):
    """Place MARKET BUY order for 10 quantity"""
    global broker_client
    
    if not broker_client:
        print("❌ Broker client not available")
        return False, "NO_CLIENT"
    
    try:
        trading_symbol = f"{stock}-EQ" if not stock.endswith("-EQ") else stock
        
        print(f"📈 Placing ORDER: BUY {trading_symbol} Qty: 10")
        
        order_result = broker_client.place_order(
            buy_or_sell="B",
            product_type="I",
            exchange="NSE",
            tradingsymbol=trading_symbol,
            quantity=10,
            price_type="MKT",
            price=0,
            discloseqty=0,
            trigger_price=0,
            retention="DAY",
            remarks=f"Chartink_Webhook_{stock}"
        )
        
        if order_result and order_result.get('stat') == 'Ok':
            order_id = order_result.get('norenordno', 'UNKNOWN')
            print(f"✅ Order placed: {order_id}")
            return True, order_id
        else:
            print(f"❌ Order failed: {order_result}")
            return False, "FAILED"
            
    except Exception as e:
        print(f"❌ Order error: {e}")
        return False, "ERROR"

def alert_worker():
    """Process alerts from queue"""
    while True:
        alert = alert_queue.get()
        if alert is None:
            break

        try:
            stocks = alert.get('stocks', '').split(',')
            trigger_prices = alert.get('trigger_prices', '').split(',')
            
            if stocks and trigger_prices:
                stock = stocks[0].strip()
                trigger_price = trigger_prices[0].strip()
                
                print(f"🎯 Processing: {stock} at {trigger_price}")
                
                success, order_id = place_simple_buy_order(stock, trigger_price)
                order_status = "SUCCESS" if success else "FAILED"
                
                # Show notification
                title = f"Chartink Alert: {alert.get('alert_name', 'Unknown')}"
                message = f"BUY {stock} Qty: 10\nStatus: {'✅ Success' if success else '❌ Failed'}"
                toaster.show_toast(title, message, duration=5, threaded=True)
                
                # Log to CSV
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        alert.get("alert_name", ""),
                        alert.get("stocks", ""),
                        alert.get("trigger_prices", ""),
                        order_status
                    ])
                
        except Exception as e:
            print(f"❌ Alert processing error: {e}")
        
        alert_queue.task_done()
        time.sleep(0.5)

# Start worker thread
worker_thread = threading.Thread(target=alert_worker, daemon=True)
worker_thread.start()

@app.route('/chartink-webhook', methods=['POST'])
def chartink_webhook():
    """Main webhook endpoint"""
    try:
        data = request.get_json()
        
        print("\n" + "="*50)
        print("📢 CHARTINK ALERT RECEIVED")
        print("="*50)
        print(f"Alert: {data.get('alert_name', 'Unknown')}")
        print(f"Stocks: {data.get('stocks', 'None')}")
        print(f"Prices: {data.get('trigger_prices', 'None')}")
        print("="*50)
        
        # Add to queue
        alert_queue.put(data)
        
        return jsonify({
            "status": "success", 
            "message": "Alert queued for processing"
        }), 200
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "broker_connected": client_loaded,
        "alerts_queued": alert_queue.qsize(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }), 200

def start_webhook_server():
    """Start the webhook server"""
    # Load broker client first
    if not load_broker_client():
        print("❌ Cannot start webhook server - broker client failed")
        return False
    
    print("🚀 Starting Chartink Webhook Server...")
    print("📍 Webhook URL: http://0.0.0.0:5000/chartink-webhook")
    print("❤️  Health check: http://0.0.0.0:5000/health")
    
    # Start Flask server
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    return True


# run this from command window for testing
# curl -X POST https://etymologic-impoliticly-cooper.ngrok-free.dev/chartink-webhook -H "Content-Type: application/json" -d "{\"stocks\":\"RELIANCE\",\"trigger_prices\":\"2520.5\",\"scan_name\":\"Breakout\",\"alert_name\":\"Test Alert\",\"triggered_at\":\"10:15 am\"}"



if __name__ == '__main__':
    start_webhook_server()