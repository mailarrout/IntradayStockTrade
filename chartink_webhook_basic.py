from flask import Flask, request, jsonify
from win10toast import ToastNotifier
import threading
import queue
import datetime
import time
import csv
import os

app = Flask(__name__)
toaster = ToastNotifier()

# Queue to store incoming alerts
alert_queue = queue.Queue()

# CSV filename
csv_file = "app\Date_Chartink_rsi-cross-with-yesterday-high-break-one-min.csv"

# Ensure CSV file has headers if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Alert Name", "Stocks", "Trigger Prices", "Scan Name", "Triggered At"])

# Worker thread to display alerts with popup + sound
def alert_worker():
    while True:
        alert = alert_queue.get()  # Wait for an alert
        if alert is None:
            break  # Exit signal

        title = f"Chartink Alert: {alert['alert_name']}"
        message = f"Stocks: {alert['stocks']}\nTrigger Prices: {alert['trigger_prices']}"

        # Show Windows popup with sound
        toaster.show_toast(title, message, duration=10, threaded=True, icon_path=None)

        # Append to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                alert.get("alert_name"),
                alert.get("stocks"),
                alert.get("trigger_prices"),
                alert.get("scan_name"),
                alert.get("triggered_at")
            ])

        # Small delay to avoid popup overlap
        time.sleep(1)
        alert_queue.task_done()

# Start worker thread
threading.Thread(target=alert_worker, daemon=True).start()

@app.route('/chartink-webhook', methods=['POST'])
def chartink_webhook():
    try:
        data = request.get_json()
        
        # Print in terminal
        print("\n" + "="*50)
        print("📢 Chartink Alert Received at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*50)
        print(f"🔔 Alert: {data.get('alert_name')}")
        print(f"⏰ Triggered At: {data.get('triggered_at')}")
        print(f"📊 Stocks: {data.get('stocks')}")
        print(f"💰 Trigger Prices: {data.get('trigger_prices')}")
        print("="*50 + "\n")

        # Add alert to queue for popups + CSV
        alert_queue.put(data)

        return jsonify({"status": "success", "message": "Alert queued"}), 200
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
