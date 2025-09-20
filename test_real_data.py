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
    Test version of analyze_stock_file that only takes file_path
    Returns tuple: (result_dict or None, is_invalid(bool), is_triggered(bool))
    """
    stock = os.path.basename(file_path).replace('.csv', '')
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"{stock}: Failed to read file: {e}")
        return None, False, False

    # normalize column names
    df = df.rename(columns={
        "time": "Date",
        "into": "Open",
        "inth": "High",
        "intl": "Low",
        "intc": "Close",
        "intv": "Volume"
    })
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S')
    except Exception:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df = df.sort_values('Date').reset_index(drop=True)
    if df.empty:
        logger.info(f"{stock}: Empty data, skipping")
        return None, True, False

    # --- Step A: identify first 3 candles (9:15, 9:20, 9:25) ---
    first_three = df[(df['Date'].dt.hour == 9) & (df['Date'].dt.minute.isin([15, 20, 25]))].sort_values('Date')
    if len(first_three) < 3:
        logger.info(f"{stock}: not enough first-3 candles")
        return None, False, False

    # locate the lowest-volume among those three
    lowest_idx = first_three['Volume'].idxmin()
    ref_row = df.loc[lowest_idx]
    ref_volume = float(ref_row['Volume'])
    ref_time = ref_row['Date']
    logger.debug(f"{stock}: Lowest of first-3 at {ref_time} with vol {ref_volume}")

    # Step B: find target GREEN/RED candle with lower volume after ref_time
    target_green, target_red = None, None
    for _, row in df.iterrows():
        if row['Date'] <= ref_time:
            continue
        vol = float(row['Volume'])
        if vol >= ref_volume:
            continue
        if float(row['Close']) > float(row['Open']) and target_green is None:
            target_green = row
            logger.debug(f"{stock}: Found Target GREEN at {row['Date']} vol {vol}")
        if float(row['Close']) < float(row['Open']) and target_red is None:
            target_red = row
            logger.debug(f"{stock}: Found Target RED at {row['Date']} vol {vol}")
        if target_green is not None and target_red is not None:
            break

    if target_green is None and target_red is None:
        logger.debug(f"{stock}: No target candles found")
        return None, False, False

    # Step C: scan only the next 3 candles for breakout condition (close vs target OPEN)
    start_time = min([t['Date'] for t in [target_green, target_red] if t is not None])
    target_idx = df[df['Date'] == start_time].index[0]

    max_candles_after_target = 3
    end_idx = min(target_idx + max_candles_after_target + 1, len(df))

    for idx in range(target_idx + 1, end_idx):
        row = df.iloc[idx]

        if target_green is not None:
            # SELL if later candle closes below target GREEN's Open
            if float(row['Close']) < float(target_green['Open']):
                entry_price = float(row['Close'])
                stop_loss = float(target_green['High']) + 0.5
                risk = abs(entry_price - stop_loss)
                target_price = entry_price - risk
                res = {
                    'Stock': stock,
                    'Entry_Point': row['Date'],
                    'Entry_Price': entry_price,
                    'Stop_Loss': stop_loss,
                    'Target_Price': target_price,
                    'Target_Candle_Time': target_green['Date'],
                    'Target_Candle_Open': float(target_green['Open']),
                    'Target_Candle_Type': 'GREEN',
                    'Direction': 'SELL'
                }
                logger.info(f"{stock}: GREEN target OPEN broken within 3 candles. Entry {entry_price}, SL {stop_loss}, Target {target_price}")
                return res, False, True

        if target_red is not None:
            # BUY if later candle closes above target RED's Open
            if float(row['Close']) > float(target_red['Open']):
                entry_price = float(row['Close'])
                stop_loss = float(target_red['Low']) - 0.5
                risk = abs(entry_price - stop_loss)
                target_price = entry_price + risk
                res = {
                    'Stock': stock,
                    'Entry_Point': row['Date'],
                    'Entry_Price': entry_price,
                    'Stop_Loss': stop_loss,
                    'Target_Price': target_price,
                    'Target_Candle_Time': target_red['Date'],
                    'Target_Candle_Open': float(target_red['Open']),
                    'Target_Candle_Type': 'RED',
                    'Direction': 'BUY'
                }
                logger.info(f"{stock}: RED target OPEN broken within 3 candles. Entry {entry_price}, SL {stop_loss}, Target {target_price}")
                return res, False, True

    # If no breakout within 3 candles → no trade
    logger.debug(f"{stock}: No breakout within 3 candles after target")
    return None, False, False

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

def test_all_stocks():
    """Test all stocks without any bookkeeping"""
    current_date = datetime.now().strftime('%Y-%m-%d')
    historical_data_dir = os.path.join('HistoricalData', current_date)
    
    if not os.path.exists(historical_data_dir):
        print(f"No HistoricalData for {current_date}: {historical_data_dir}")
        return

    csv_files = glob.glob(os.path.join(historical_data_dir, '*.csv'))
    if not csv_files:
        print("No CSV files to process")
        return

    print(f"Testing {len(csv_files)} stocks...")
    print("=" * 60)
    
    signals = []
    invalid_stocks = []
    triggered_stocks = []
    no_signal_stocks = []

    for file_path in csv_files:
        stock = os.path.basename(file_path).replace('.csv', '')
        print(f"Analyzing {stock}...", end=" ")
        
        try:
            result, is_invalid, is_triggered = analyze_stock_file_test(file_path)
            
            if result:
                signals.append(result)
                triggered_stocks.append(stock)
                print("✅ SIGNAL")
            elif is_invalid:
                invalid_stocks.append(stock)
                print("❌ INVALID")
            elif is_triggered:
                triggered_stocks.append(stock)
                print("🔁 TRIGGERED")
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
    print(f"Triggered stocks: {len(triggered_stocks)}")
    print(f"No signal stocks: {len(no_signal_stocks)}")

    if signals:
        print("\n🔔 TRADE SIGNALS:")
        for signal in signals:
            print(f"  {signal['Stock']}: {signal['Direction']} at {signal['Entry_Price']:.2f}, SL: {signal['Stop_Loss']:.2f}")

    if invalid_stocks:
        print(f"\n❌ INVALID STOCKS ({len(invalid_stocks)}): {', '.join(invalid_stocks[:10])}{'...' if len(invalid_stocks) > 10 else ''}")

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
            print("WARNING: No broker clients loaded. Running in simulation mode.")
            # Continue but orders won't be placed
        
        # Import and run the actual strategy
        from LowVolumeCandleBreakOut import run_strategy
        run_strategy(ui)
        
        print("\n" + "=" * 60)
        print("Strategy execution completed!")
        print("Check your Shoonya terminal for placed orders")
        
    except Exception as e:
        print(f"Error during strategy execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Low Volume Breakout Strategy Tester")
    print("=" * 50)
    print("1. Test all stocks (analysis only - NO ORDERS)")
    print("2. Test single stock (analysis only)")
    print("3. Run strategy with REAL ORDER PLACEMENT")
    print("4. Test order placement (simple test)")
    print("5. Clean up test files")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        test_all_stocks()
    elif choice == "2":
        test_single_stock()
    elif choice == "3":
        test_with_real_order_placement()  # ✅ This is the new option
    elif choice == "4":
        test_order_placement()
    elif choice == "5":
        cleanup_test_files()
    elif choice == "6":
        print("Exiting...")
    else:
        print("Invalid choice")