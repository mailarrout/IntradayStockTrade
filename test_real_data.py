# test_real_data.py
import os
import sys
import glob
import logging
from datetime import datetime
import pandas as pd
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class TestUI:
    """Mock UI for testing"""
    def __init__(self):
        self.messages = []
        self.clients = []  # Add clients attribute
    
    def log(self, level, message):
        print(f"[{level}] {message}")
        self.messages.append(f"[{level}] {message}")
    
    def show_trade_alert(self, stock, entry_time, entry_price, stop_loss):
        print(f"🚨 TRADE ALERT: {stock} - Entry: {entry_price} at {entry_time}, SL: {stop_loss}")

def analyze_stock_file_test(file_path):
    """
    CORRECTED VERSION: Exactly matches LowVolumeCandleBreakOut.py logic
    """
    stock = os.path.basename(file_path).replace('.csv', '')
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"{stock}: Failed to read file: {e}")
        return None, False, False

    # Normalize column names (same as main strategy)
    df = df.rename(columns={
        "time": "Date", "into": "Open", "inth": "High", 
        "intl": "Low", "intc": "Close", "intv": "Volume"
    })
    
    # Date parsing (same as main strategy)
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        df = df.dropna(subset=['Date'])
        if df.empty:
            return None, True, False
    except:
        return None, True, False

    df = df.sort_values('Date').reset_index(drop=True)
    if df.empty:
        return None, True, False

    # Use all available data (remove date filter for testing)
    df_today = df
    
    # ✅ EXACT SAME LOGIC AS MAIN STRATEGY
    reference_candle = None
    reference_volume = None
    active_target = None
    monitoring_countdown = 0
    target_candle_type = None
    
    for idx, row in df_today.iterrows():
        candle_time = row['Date']
        candle_hour = candle_time.hour
        candle_minute = candle_time.minute
        
        # STEP 1: Reference candle (9:15, 9:20, 9:25)
        if candle_hour == 9 and candle_minute in [15, 20, 25]:
            if reference_candle is None:
                reference_candle = row
                reference_volume = float(row['Volume'])
            else:
                current_volume = float(row['Volume'])
                if current_volume < reference_volume:
                    reference_candle = row
                    reference_volume = current_volume
        
        # STEP 2: Target candles after 9:25
        elif (candle_hour > 9 or (candle_hour == 9 and candle_minute > 25)) and reference_candle is not None:
            current_volume = float(row['Volume'])
            
            if active_target is None and current_volume < reference_volume:
                is_green = float(row['Close']) > float(row['Open'])
                is_red = float(row['Close']) < float(row['Open'])
                
                if is_green or is_red:
                    active_target = row
                    target_candle_type = 'GREEN' if is_green else 'RED'
                    monitoring_countdown = 5
                    continue
        
        # STEP 3: Breakout monitoring - ✅ EXACT SAME AS MAIN STRATEGY
        if active_target is not None and monitoring_countdown > 0:
            monitoring_countdown -= 1
            
            current_close = float(row['Close'])
            target_open = float(active_target['Open'])
            
            breakout_occurred = False
            if target_candle_type == 'GREEN':
                # ✅ CORRECT: SELL when close < GREEN target's OPEN
                if current_close < target_open:
                    breakout_occurred = True
                    direction = 'SELL'
            elif target_candle_type == 'RED':
                # ✅ CORRECT: BUY when close > RED target's OPEN  
                if current_close > target_open:
                    breakout_occurred = True
                    direction = 'BUY'
            
            if breakout_occurred:
                entry_price = current_close
                
                # Stop loss calculation (same as main strategy)
                target_idx = df_today[df_today['Date'] == active_target['Date']].index[0]
                entry_idx = idx
                
                if direction == 'SELL':
                    highest_high = float(active_target['High'])
                    for candle_idx in range(target_idx, entry_idx + 1):
                        candle_high = float(df_today.iloc[candle_idx]['High'])
                        if candle_high > highest_high:
                            highest_high = candle_high
                    stop_loss = highest_high + 0.05
                    target_price = entry_price - (stop_loss - entry_price)
                else:
                    lowest_low = float(active_target['Low'])
                    for candle_idx in range(target_idx, entry_idx + 1):
                        candle_low = float(df_today.iloc[candle_idx]['Low'])
                        if candle_low < lowest_low:
                            lowest_low = candle_low
                    stop_loss = lowest_low - 0.05
                    target_price = entry_price + (entry_price - stop_loss)
                
                # Prepare result (same fields as main strategy)
                res = {
                    'Stock': stock,
                    'Entry_Point': row['Date'],
                    'Entry_Price': entry_price,
                    'Stop_Loss': stop_loss,
                    'Target_Price': target_price,
                    'Target_Candle_Time': active_target['Date'],
                    'Target_Candle_Open': float(active_target['Open']),
                    'Target_Candle_High': float(active_target['High']),
                    'Target_Candle_Low': float(active_target['Low']),
                    'Target_Candle_Close': float(active_target['Close']),
                    'Target_Candle_Volume': float(active_target['Volume']),
                    'Target_Candle_Type': target_candle_type,
                    'Direction': direction,
                    'Breakout_Candle_Time': candle_time.strftime('%H:%M'),
                    'Breakout_Price': entry_price
                }
                
                logger.info(f"{stock}: ✅ {direction} at {entry_price:.2f}")
                return res, False, True
        
        if active_target is not None and monitoring_countdown <= 0:
            active_target = None
            target_candle_type = None

    return None, False, False

def export_results_to_csv(signals, filename=None):
    """Export signals to CSV with proper error handling"""
    if not signals:
        print("No signals to export")
        return
    
    if filename is None:
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"strategy_results_{current_date}.csv"
    
    # Create DataFrame with all signal details
    rows = []
    for signal in signals:
        # Calculate risk/reward ratio
        entry_price = signal['Entry_Price']
        stop_loss = signal['Stop_Loss']
        target_price = signal.get('Target_Price', 0)
        
        if signal['Direction'] == 'SELL':
            risk = stop_loss - entry_price
            reward = entry_price - target_price
        else:  # BUY
            risk = entry_price - stop_loss
            reward = target_price - entry_price
        
        risk_reward = reward / risk if risk != 0 else 0
        
        # ✅ SAFE DATA EXTRACTION WITH DEFAULT VALUES
        row = {
            'Stock': signal.get('Stock', ''),
            'Direction': signal.get('Direction', ''),
            'Entry_Price': entry_price,
            'Stop_Loss': stop_loss,
            'Target_Price': target_price,
            'Risk': round(risk, 2),
            'Reward': round(reward, 2),
            'Risk_Reward_Ratio': round(risk_reward, 2),
            'Entry_Time': signal.get('Entry_Point', ''),
            'Target_Candle_Time': signal.get('Target_Candle_Time', ''),
            'Target_Candle_Type': signal.get('Target_Candle_Type', ''),
            'Target_Candle_Open': signal.get('Target_Candle_Open', 0),
            'Target_Candle_High': signal.get('Target_Candle_High', 0),
            'Target_Candle_Low': signal.get('Target_Candle_Low', 0),
            'Target_Candle_Close': signal.get('Target_Candle_Close', 0),
            'Target_Candle_Volume': signal.get('Target_Candle_Volume', 0),
            'Breakout_Candle_Time': signal.get('Breakout_Candle_Time', '')
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"✅ Results exported to: {filename}")
    return filename

def test_all_stocks():
    """Test with specific date folder and CSV export"""
    # ✅ SPECIFIC DATE FOLDER (change this to your actual folder)
    specific_date = "2025-09-26"  # Change to your folder date
    historical_data_dir = os.path.join('HistoricalData', specific_date)
    
    if not os.path.exists(historical_data_dir):
        print(f"❌ Folder not found: {historical_data_dir}")
        print("Available folders:")
        base_dir = 'HistoricalData'
        if os.path.exists(base_dir):
            folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
            for folder in folders:
                print(f"  {folder}")
        return

    csv_files = glob.glob(os.path.join(historical_data_dir, '*.csv'))
    if not csv_files:
        print(f"No CSV files in {historical_data_dir}")
        return

    print(f"Testing {len(csv_files)} stocks from {specific_date}...")
    print("=" * 60)
    
    signals = []
    invalid_stocks = []
    no_signal_stocks = []
    
    for file_path in csv_files:
        stock = os.path.basename(file_path).replace('.csv', '')
        print(f"Analyzing {stock}...", end=" ")
        
        try:
            result, is_invalid, is_triggered = analyze_stock_file_test(file_path)
            
            if result:
                signals.append(result)
                print("✅ SIGNAL")
            elif is_invalid:
                invalid_stocks.append(stock)
                print("❌ INVALID")
            else:
                no_signal_stocks.append(stock)
                print("➖ NO SIGNAL")
                
        except Exception as e:
            print(f"ERROR: {e}")
            no_signal_stocks.append(stock)

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY:")
    print(f"Total stocks analyzed: {len(csv_files)}")
    print(f"Signals found: {len(signals)}")
    print(f"Invalid stocks: {len(invalid_stocks)}")
    print(f"No signal stocks: {len(no_signal_stocks)}")

    # ✅ CSV EXPORT - ADDED BACK
    if signals:
        print("\n🔔 TRADE SIGNALS:")
        for signal in signals:
            # Calculate risk/reward for display
            entry_price = signal['Entry_Price']
            stop_loss = signal['Stop_Loss']
            target_price = signal['Target_Price']
            
            if signal['Direction'] == 'SELL':
                risk = stop_loss - entry_price
                reward = entry_price - target_price
            else:
                risk = entry_price - stop_loss
                reward = target_price - entry_price
            
            risk_reward = reward / risk if risk != 0 else 0
            
            print(f"  {signal['Stock']}: {signal['Direction']} at {entry_price:.2f}, SL: {stop_loss:.2f}, Target: {target_price:.2f}, R:R = 1:{risk_reward:.2f}")

        # ✅ EXPORT TO CSV
        export_results_to_csv(signals, f"strategy_results_{specific_date}.csv")
        
        print(f"\n✅ Results exported to: strategy_results_{specific_date}.csv")

    # Also export summary of all stocks
    export_stock_summary(csv_files, signals, invalid_stocks, no_signal_stocks, specific_date)

def export_stock_summary(csv_files, signals, invalid_stocks, no_signal_stocks, date):
    """Export complete summary of all stocks analyzed"""
    summary_data = []
    
    # Add signals
    for signal in signals:
        summary_data.append({
            'Stock': signal['Stock'],
            'Status': 'SIGNAL',
            'Direction': signal['Direction'],
            'Entry_Price': signal['Entry_Price'],
            'Entry_Time': signal['Entry_Point'],
            'Stop_Loss': signal['Stop_Loss'],
            'Target_Price': signal['Target_Price']
        })
    
    # Add invalid stocks
    for stock in invalid_stocks:
        summary_data.append({
            'Stock': stock,
            'Status': 'INVALID',
            'Direction': '',
            'Entry_Price': '',
            'Entry_Time': '',
            'Stop_Loss': '',
            'Target_Price': ''
        })
    
    # Add no signal stocks
    for stock in no_signal_stocks:
        summary_data.append({
            'Stock': stock,
            'Status': 'NO_SIGNAL',
            'Direction': '',
            'Entry_Price': '',
            'Entry_Time': '',
            'Stop_Loss': '',
            'Target_Price': ''
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f"stock_analysis_summary_{date}.csv", index=False)
    print(f"✅ Stock summary exported to: stock_analysis_summary_{date}.csv")

def export_results_to_csv(signals, filename=None):
    """Export signals to CSV for better analysis"""
    if not signals:
        print("No signals to export")
        return
    
    if filename is None:
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"strategy_results_{current_date}.csv"
    
    # Create DataFrame with all signal details
    rows = []
    for signal in signals:
        # Calculate risk/reward ratio
        entry_price = signal['Entry_Price']
        stop_loss = signal['Stop_Loss']
        target_price = signal['Target_Price']
        
        if signal['Direction'] == 'SELL':
            risk = stop_loss - entry_price
            reward = entry_price - target_price
        else:  # BUY
            risk = entry_price - stop_loss
            reward = target_price - entry_price
        
        risk_reward = reward / risk if risk != 0 else 0
        
        row = {
            'Stock': signal['Stock'],
            'Direction': signal['Direction'],
            'Entry_Price': entry_price,
            'Stop_Loss': stop_loss,
            'Target_Price': target_price,
            'Risk': round(risk, 2),
            'Reward': round(reward, 2),
            'Risk_Reward_Ratio': round(risk_reward, 2),
            'Entry_Time': signal['Entry_Point'],
            'Target_Candle_Time': signal['Target_Candle_Time'],
            'Target_Candle_Type': signal['Target_Candle_Type'],
            'Target_Candle_Open': signal['Target_Candle_Open'],
            'Target_Candle_High': signal['Target_Candle_High'],
            'Target_Candle_Low': signal['Target_Candle_Low'],
            'Target_Candle_Close': signal['Target_Candle_Close'],
            'Target_Candle_Volume': signal['Target_Candle_Volume'],
            'Breakout_Candle_Time': signal['Breakout_Candle_Time']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"✅ Results exported to: {filename}")
    return filename

def test_single_stock():
    """Test a specific stock file"""
    # Specify the stock file you want to test
    stock_name = input("Enter stock name to test (e.g., EXICOM): ").strip().upper()
    current_date = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join('HistoricalData', current_date, f'{stock_name}.csv')

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Available files:")
        data_dir = os.path.join('HistoricalData', current_date)
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            for f in files[:20]:  # Show first 20 files
                print(f"  {f.replace('.csv', '')}")
            if len(files) > 20:
                print(f"  ... and {len(files) - 20} more")
        return

    print(f"Testing single stock: {stock_name}")
    print("=" * 60)

    try:
        result, is_invalid, is_triggered = analyze_stock_file_test(file_path)

        if result:
            # Calculate risk/reward
            entry_price = result['Entry_Price']
            stop_loss = result['Stop_Loss']
            target_price = result['Target_Price']
            
            if result['Direction'] == 'SELL':
                risk = stop_loss - entry_price
                reward = entry_price - target_price
            else:  # BUY
                risk = entry_price - stop_loss
                reward = target_price - entry_price
                
            risk_reward = reward / risk if risk != 0 else None

            print("✅ TRADE SIGNAL FOUND!")
            print(f"Stock: {result['Stock']}")
            print(f"Direction: {result['Direction']}")
            print(f"Entry: {entry_price:.2f} at {result['Entry_Point']}")
            print(f"Stop Loss: {stop_loss:.2f}")
            print(f"Target Price: {target_price:.2f}")
            if risk_reward is not None:
                print(f"Risk/Reward: 1:{risk_reward:.2f}")
            else:
                print("Risk/Reward: N/A")
            print(f"Target Candle: {result['Target_Candle_Type']} at {result['Target_Candle_Time']}")

            # Export single result to CSV
            export_results_to_csv([result], f"single_stock_{stock_name}_{datetime.now().strftime('%H%M%S')}.csv")

        elif is_invalid:
            print("❌ Stock invalidated")
        elif is_triggered:
            print("ℹ️ Stock meets criteria but already triggered")
        else:
            print("ℹ️ No trade signal found")

    except Exception as e:
        print(f"Error testing {stock_name}: {e}")
        import traceback
        traceback.print_exc()

# ... (keep the rest of your existing functions unchanged - test_order_placement, cleanup_test_files, etc.)

def test_order_placement():
    """Test order placement without actual trading"""
    ui = TestUI()
    
    # Load clients first
    print("Loading broker clients for testing...")
    from client_loader import load_clients
    load_clients(ui, auto_load=True)
    
    if not hasattr(ui, 'clients') or not ui.clients:
        print("❌ No clients available")
        return
    
    client_name, client_id, client = ui.clients[0]
    
    # Test with a simple order
    test_symbol = "RELIANCE-EQ"
    test_qty = 1
    
    print(f"Testing order placement for {test_symbol}...")
    
    try:
        # Get current market price first
        try:
            quote = client.get_queries("NSE", test_symbol)
            if quote and isinstance(quote, list) and len(quote) > 0:
                ltp = float(quote[0].get('lp', 0))
                test_price = round(ltp * 0.95, 2)  # 5% below LTP to avoid immediate execution
                print(f"Current LTP: {ltp:.2f}, Using test price: {test_price:.2f}")
            else:
                test_price = 2500.00  # Fallback price
                print(f"Using fallback price: {test_price:.2f}")
        except:
            test_price = 2500.00
            print(f"Using fallback price: {test_price:.2f}")
        
        # Test BUY order
        print(f"Placing BUY order for {test_symbol} at {test_price:.2f}")
        order_result = client.place_order(
            buy_or_sell="B",
            product_type="I",
            exchange="NSE",
            tradingsymbol=test_symbol,
            quantity=test_qty,
            price_type="MKT",
            price=test_price,
            discloseqty= 0,
            trigger_price=0,
            retention="DAY",
            remarks="TEST_ORDER"
        )
        
        print("Order Result:", order_result)
        
        if order_result and order_result.get('stat') == 'Ok':
            order_id = order_result.get('norenordno')
            print(f"✅ Order placed successfully! Order ID: {order_id}")
            
            # Check order status
            time.sleep(3)
            try:
                order_status = client.get_order_history(order_id)
                print("Order Status:", order_status)
            except Exception as e:
                print(f"Error checking order status: {e}")
            
            # Cancel the test order
            print("Cancelling test order...")
            cancel_result = client.cancel_order(order_id)
            print("Cancel Result:", cancel_result)
            
        else:
            error_msg = order_result.get('emsg', 'Unknown error') if order_result else 'No response'
            print(f"❌ Order failed: {error_msg}")
            
    except Exception as e:
        print(f"❌ Order placement error: {e}")
        import traceback
        traceback.print_exc()

def cleanup_test_files():
    """Clean up test files safely"""
    current_date = datetime.now().strftime('%Y-%m-%d')
    test_files = [
        os.path.join('app', f'{current_date}_triggered_stocks.csv'),
        os.path.join('app', f'{current_date}_invalid_stocks.csv'),
        os.path.join('app', f'{current_date}_active_trades.csv'),
        os.path.join('app', f'{current_date}_active_trades_audit.csv'),
        os.path.join('app', f'{current_date}_low_volume_breakout_signals.csv')
    ]
    
    cleaned = 0
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned: {os.path.basename(file_path)}")
                cleaned += 1
            except Exception as e:
                print(f"Failed to clean {os.path.basename(file_path)}: {e}")
    
    if cleaned > 0:
        print(f"Cleaned {cleaned} test files")
    else:
        print("No test files to clean")

def test_with_real_order_placement():
    """Test the strategy with actual order placement"""
    ui = TestUI()
    
    print("Starting strategy with REAL order placement...")
    print("=" * 60)
    
    try:
        # Load clients for testing
        print("Loading broker clients...")
        from client_loader import load_clients
        load_clients(ui, auto_load=True)
        
        if not hasattr(ui, 'clients') or not ui.clients:
            print("❌ WARNING: No broker clients loaded. Running in simulation mode.")
            # Continue but orders won't be placed
        
        # Import and run the actual strategy
        from LowVolumeCandleBreakOut import run_strategy
        run_strategy(ui)
        
        print("\n" + "=" * 60)
        print("Strategy execution completed!")
        print("Check your Shoonya terminal for placed orders")
        
    except Exception as e:
        print(f"❌ Error during strategy execution: {e}")
        import traceback
        traceback.print_exc()

def test_order_placement():
    """Test order placement without actual trading"""
    ui = TestUI()
    
    # Load clients first
    print("Loading broker clients for testing...")
    try:
        from client_loader import load_clients
        load_clients(ui, auto_load=True)
    except Exception as e:
        print(f"❌ Error loading clients: {e}")
        return
    
    if not hasattr(ui, 'clients') or not ui.clients:
        print("❌ No clients available")
        return
    
    client_name, client_id, client = ui.clients[0]
    
    # Test with a simple order
    test_symbol = "RELIANCE-EQ"
    test_qty = 1
    
    print(f"Testing order placement for {test_symbol}...")
    
    try:
        # Get current market price first
        try:
            quote = client.get_quotes("NSE", test_symbol)
            if quote and isinstance(quote, list) and len(quote) > 0:
                ltp = float(quote[0].get('lp', 0))
                test_price = round(ltp * 0.95, 2)  # 5% below LTP to avoid immediate execution
                print(f"Current LTP: {ltp:.2f}, Using test price: {test_price:.2f}")
            else:
                test_price = 2500.00  # Fallback price
                print(f"Using fallback price: {test_price:.2f}")
        except Exception as e:
            test_price = 2500.00
            print(f"Using fallback price: {test_price:.2f}")
        
        # Test BUY order
        print(f"Placing BUY order for {test_symbol} at {test_price:.2f}")
        order_result = client.place_order(
            buy_or_sell="B",
            product_type="I",
            exchange="NSE",
            tradingsymbol=test_symbol,
            quantity=test_qty,
            price_type="LMT",  # Use limit order for safety
            price=test_price,
            discloseqty=0,
            trigger_price=0,
            retention="DAY",
            remarks="TEST_ORDER"
        )
        
        print("Order Result:", order_result)
        
        if order_result and order_result.get('stat') == 'Ok':
            order_id = order_result.get('norenordno')
            print(f"✅ Order placed successfully! Order ID: {order_id}")
            
            # Check order status
            time.sleep(3)
            try:
                order_status = client.get_order_history(order_id)
                print("Order Status:", order_status)
            except Exception as e:
                print(f"Error checking order status: {e}")
            
            # Cancel the test order
            print("Cancelling test order...")
            cancel_result = client.cancel_order(order_id)
            print("Cancel Result:", cancel_result)
            
        else:
            error_msg = order_result.get('emsg', 'Unknown error') if order_result else 'No response'
            print(f"❌ Order failed: {error_msg}")
            
    except Exception as e:
        print(f"❌ Order placement error: {e}")
        import traceback
        traceback.print_exc()

def cleanup_test_files():
    """Clean up test files safely"""
    current_date = datetime.now().strftime('%Y-%m-%d')
    test_files = [
        os.path.join('app', f'{current_date}_triggered_stocks.csv'),
        os.path.join('app', f'{current_date}_invalid_stocks.csv'),
        os.path.join('app', f'{current_date}_active_trades.csv'),
        os.path.join('app', f'{current_date}_active_trades_audit.csv'),
        os.path.join('app', f'{current_date}_low_volume_breakout_signals.csv'),
        # Add CSV result files
        os.path.join('strategy_results_*.csv'),
        os.path.join('single_stock_*.csv')
    ]
    
    cleaned = 0
    for file_pattern in test_files:
        if '*' in file_pattern:
            # Handle wildcard patterns
            import glob
            files_to_delete = glob.glob(file_pattern)
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"Cleaned: {os.path.basename(file_path)}")
                    cleaned += 1
                except Exception as e:
                    print(f"Failed to clean {os.path.basename(file_path)}: {e}")
        else:
            # Handle specific files
            if os.path.exists(file_pattern):
                try:
                    os.remove(file_pattern)
                    print(f"Cleaned: {os.path.basename(file_pattern)}")
                    cleaned += 1
                except Exception as e:
                    print(f"Failed to clean {os.path.basename(file_pattern)}: {e}")
    
    # Clean up app directory if empty
    app_dir = 'app'
    if os.path.exists(app_dir) and not os.listdir(app_dir):
        try:
            os.rmdir(app_dir)
            print(f"Cleaned empty directory: {app_dir}")
        except:
            pass
    
    if cleaned > 0:
        print(f"✅ Cleaned {cleaned} test files")
    else:
        print("ℹ️ No test files to clean")

def show_results_summary():
    """Show summary of recent test results"""
    import glob
    csv_files = glob.glob("strategy_results_*.csv")
    
    if not csv_files:
        print("No test result files found")
        return
    
    # Sort by modification time (newest first)
    csv_files.sort(key=os.path.getmtime, reverse=True)
    
    print("Recent Test Results:")
    print("=" * 50)
    
    for i, csv_file in enumerate(csv_files[:3]):  # Show last 3 results
        try:
            df = pd.read_csv(csv_file)
            file_time = os.path.getmtime(csv_file)
            date_str = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{i+1}. {os.path.basename(csv_file)}")
            print(f"   Date: {date_str}")
            print(f"   Signals: {len(df)}")
            print(f"   BUY: {len(df[df['Direction'] == 'BUY'])}")
            print(f"   SELL: {len(df[df['Direction'] == 'SELL'])}")
            
            if len(df) > 0:
                avg_rr = df['Risk_Reward_Ratio'].mean()
                print(f"   Avg Risk/Reward: 1:{avg_rr:.2f}")
            print()
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

if __name__ == "__main__":
    print("Low Volume Breakout Strategy Tester")
    print("=" * 50)
    print("1. Test all stocks (analysis only - NO ORDERS)")
    print("2. Test single stock (analysis only)")
    print("3. Run strategy with REAL ORDER PLACEMENT")
    print("4. Test order placement (simple test)")
    print("5. Clean up test files")
    print("6. Show recent results summary")
    print("7. Exit")
    
    choice = input("\nEnter your choice (1-7): ").strip()
    
    if choice == "1":
        test_all_stocks()
    elif choice == "2":
        test_single_stock()
    elif choice == "3":
        test_with_real_order_placement()
    elif choice == "4":
        test_order_placement()
    elif choice == "5":
        cleanup_test_files()
    elif choice == "6":
        show_results_summary()
    elif choice == "7":
        print("Exiting...")
    else:
        print("Invalid choice")