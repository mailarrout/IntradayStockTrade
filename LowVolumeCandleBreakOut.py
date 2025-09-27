# LowVolumeCandleBreakOut.py
import os
import glob
import logging
import pandas as pd
import time
from PyQt5.QtCore import QTimer
import pytz
from datetime import datetime, timedelta, time

# -----------------------
# Config / globals
# -----------------------

MAX_TRADES_PER_DAY = 5
TRADE_QUANTITY = 10
active_trades = {}  # trade_id -> trade info

# Monitoring variables
monitor_timer = None
stop_monitoring = False
strategy_is_running = False

# ✅ CORRECTED IST Formatter Setup
class ISTFormatter(logging.Formatter):
    """Custom formatter that converts timestamps to IST"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, pytz.timezone('Asia/Kolkata'))
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S IST')

# Get logger for this module
logger = logging.getLogger(__name__)

# ✅ Apply IST formatting ONLY after main logging is configured
def setup_ist_logging():
    """Setup IST formatting for this module's logs"""
    ist_formatter = ISTFormatter('%(asctime)s %(levelname)s %(message)s')
    
    # Apply to all existing handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(ist_formatter)
    
    # Also apply to this module's logger
    for handler in logger.handlers:
        handler.setFormatter(ist_formatter)

# Call this function when the strategy runs or after main app initialization
setup_ist_logging()


def validate_data_freshness(df, symbol):
    """Validate if we have current data (not delayed) - RELAXED for current candle"""
    try:
        if df.empty:
            logger.warning(f"{symbol}: Empty dataframe")
            return False
            
        if 'Date' not in df.columns:
            logger.warning(f"{symbol}: No Date column found")
            return False
            
        # Sort by date and get latest
        df_sorted = df.sort_values('Date', ascending=False)
        latest_time = df_sorted.iloc[0]['Date']
        
        if pd.isna(latest_time):
            logger.warning(f"{symbol}: Invalid latest timestamp")
            return False
            
        now = datetime.now()
        
        # Parse the date if it's still in string format
        if isinstance(latest_time, str):
            try:
                latest_time = pd.to_datetime(latest_time, format='%d-%m-%Y %H:%M:%S')
            except:
                latest_time = pd.to_datetime(latest_time, errors='coerce')
        
        if pd.isna(latest_time):
            logger.warning(f"{symbol}: Could not parse latest timestamp")
            return False
            
        # Check if latest candle is within current 5-minute slot
        current_slot_start = now.replace(second=0, microsecond=0)
        current_slot_start = current_slot_start.replace(minute=(now.minute // 5) * 5)
        
        # Allow up to 7 minutes delay (current candle + 2 minute buffer)
        time_diff = (now - latest_time).total_seconds() / 60
        
        if time_diff > 7:  # More than 7 minutes delay
            logger.error(f"STALE DATA: {symbol} delayed by {time_diff:.1f} minutes")
            return False
            
        # Check if latest candle is for current or previous time slot
        if latest_time >= (current_slot_start - timedelta(minutes=5)):
            logger.debug(f"{symbol}: Data is fresh (latest: {latest_time}, delay: {time_diff:.1f}m)")
            return True
        else:
            logger.warning(f"{symbol}: Data not current (latest: {latest_time}, expected after: {current_slot_start - timedelta(minutes=5)})")
            return False
            
    except Exception as e:
        logger.error(f"{symbol}: Error validating data freshness: {e}")
        return False

# -----------------------
# Monitoring control functions
# -----------------------
def start_monitoring(ui_reference):
    """Start the 5-second monitoring timer"""
    global monitor_timer, stop_monitoring
    
    stop_monitoring = False
    
    # Create timer if it doesn't exist
    if monitor_timer is None:
        monitor_timer = QTimer()
        monitor_timer.timeout.connect(lambda: check_trades_periodically(ui_reference))
    
    monitor_timer.start(5000)  # 5 seconds
    logger.info("Started trade monitoring (5-second intervals)")

def stop_monitoring_trades():
    """Stop the monitoring timer"""
    global monitor_timer, stop_monitoring
    
    stop_monitoring = True
    if monitor_timer and monitor_timer.isActive():
        monitor_timer.stop()
        logger.info("Stopped trade monitoring")

def check_trades_periodically(ui_reference):
    """Wrapper function for periodic monitoring"""
    if stop_monitoring:
        return
        
    current_date = datetime.now().strftime('%Y-%m-%d')
    try:
        monitor_active_trades(ui_reference, current_date)
    except Exception as e:
        logger.error(f"Periodic monitoring error: {e}")

# -----------------------
# Public entry
# -----------------------
def run_strategy(ui_reference):
    """Main entry point for the strategy."""
    global strategy_is_running
    
    # Prevent concurrent execution
    if strategy_is_running:
        logger.warning("Strategy is already running, skipping")
        return
        
    try:
        strategy_is_running = True
        logger.info("Starting LowVolumeCandleBreakOut strategy")
        current_date = datetime.now().strftime('%Y-%m-%d')
        historical_data_dir = os.path.join('HistoricalData', current_date)
        if not os.path.exists(historical_data_dir):
            logger.warning(f"No HistoricalData for {current_date}: {historical_data_dir}")
            return

        csv_files = glob.glob(os.path.join(historical_data_dir, '*.csv'))
        if not csv_files:
            logger.warning("No CSV files to process")
            return

        # bookkeeping load
        invalid_stocks = load_invalid_stocks(current_date)
        triggered_stocks = load_triggered_stocks(current_date)
        load_active_trades(current_date)

        logger.info(f"Found {len(csv_files)} files. Invalid: {len(invalid_stocks)} Triggered: {len(triggered_stocks)} ActiveTrades: {len(active_trades)}")
        
        # DEBUG: Log some triggered stocks
        if triggered_stocks:
            logger.info(f"Triggered stocks: {triggered_stocks[:5]}{'...' if len(triggered_stocks) > 5 else ''}")

        signals = []
        newly_invalid = []
        newly_triggered = []

        for file_path in csv_files:
            stock = os.path.basename(file_path).replace('.csv', '')
            if stock in invalid_stocks:
                logger.debug(f"Skipping {stock} (invalid)")
                continue
            if stock in triggered_stocks:
                logger.debug(f"Skipping {stock} (already triggered)")
                continue
            if len(active_trades) >= MAX_TRADES_PER_DAY:
                logger.info("Reached max active trades for the day")
                break

            result, is_invalid, is_triggered = analyze_stock_file(file_path)
            if result:
                logger.info(f"Signal generated for {stock}")
                signals.append(result)
                newly_triggered.append(stock)
            elif is_invalid:
                newly_invalid.append(stock)
            elif is_triggered:
                newly_triggered.append(stock)

        if newly_invalid:
            save_invalid_stocks(newly_invalid, current_date)
            logger.info(f"Saved {len(newly_invalid)} invalid stocks: {newly_invalid}")
        if newly_triggered:
            save_triggered_stocks(newly_triggered, current_date)
            logger.info(f"Saved {len(newly_triggered)} triggered stocks: {newly_triggered}")

        # Persist signals to CSV for record and attempt execution
        if signals:
            save_signals_to_csv(signals, current_date)

        executed = execute_trades(signals, ui_reference, current_date)

        logger.info(f"Strategy run complete. Signals: {len(signals)}, Executed: {executed}")

        # Start monitoring existing active trades (5-second intervals)
        # Only start monitoring if we have active trades
        if active_trades:
            start_monitoring(ui_reference)
        else:
            logger.info("No active trades to monitor")

    except Exception as e:
        logger.exception(f"Error running strategy: {e}")
    finally:
        strategy_is_running = False

# -----------------------
# Strategy core
# -----------------------
def analyze_stock_file(file_path):
    stock = os.path.basename(file_path).replace('.csv', '')
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"{stock}: Failed to read file: {e}")
        return None, False, False

    # DEBUG: Log data timing information
    if not df.empty and 'Date' in df.columns:
        latest_time = df['Date'].max()
        current_time = datetime.now()
        logger.debug(f"{stock}: Latest data point: {latest_time}, Current time: {current_time}")
        logger.debug(f"{stock}: Total candles in file: {len(df)}")

    # ✅ CORRECTED DATA FRESHNESS CHECK
    if not validate_data_freshness(df, stock):
        logger.warning(f"{stock}: Skipping due to stale or invalid data")
        return None, False, False

    # Normalize column names
    df = df.rename(columns={
        "time": "Date",
        "into": "Open",
        "inth": "High",
        "intl": "Low",
        "intc": "Close",
        "intv": "Volume"
    })
    
    # ✅ CORRECTED DATE PARSING with proper error handling
    try:
        # Handle the exact format from API: "26-09-2025 09:30:00"
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        # Drop rows where date parsing failed
        df = df.dropna(subset=['Date'])
        if df.empty:
            logger.warning(f"{stock}: No valid dates after parsing")
            return None, True, False
    except Exception as e:
        logger.error(f"{stock}: Error parsing dates: {e}")
        return None, True, False

    df = df.sort_values('Date').reset_index(drop=True)
    if df.empty:
        logger.info(f"{stock}: Empty data after processing, skipping")
        return None, True, False

    # Log the available time range for debugging
    logger.debug(f"{stock}: Data range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Filter only today's data to avoid processing old data
    today = datetime.now().date()
    df_today = df[df['Date'].dt.date == today]
    if df_today.empty:
        logger.info(f"{stock}: No data for today {today}")
        return None, False, False
        
    logger.debug(f"{stock}: Today's candles: {len(df_today)}")

    # ✅ CORRECTED: Process candles SEQUENTIALLY (like real-time)
    df_today = df_today.sort_values('Date').reset_index(drop=True)
    
    # Variables to track state
    reference_candle = None
    reference_volume = None
    active_target = None
    monitoring_countdown = 0
    target_candle_type = None
    
    # Process each candle in chronological order
    for idx, row in df_today.iterrows():
        candle_time = row['Date']
        candle_hour = candle_time.hour
        candle_minute = candle_time.minute
        
        logger.debug(f"{stock} Processing: {candle_time.strftime('%H:%M')} - O:{row['Open']} H:{row['High']} L:{row['Low']} C:{row['Close']} V:{row['Volume']:,.0f}")
        
        # STEP 1: Find reference candle (9:15, 9:20, 9:25)
        if candle_hour == 9 and candle_minute in [15, 20, 25]:
            if reference_candle is None:
                # First candle of the three
                reference_candle = row
                reference_volume = float(row['Volume'])
                logger.debug(f"{stock}: First reference candidate at {candle_time.strftime('%H:%M')} - Vol: {reference_volume:,.0f}")
            else:
                # Compare with existing reference to find lowest volume
                current_volume = float(row['Volume'])
                if current_volume < reference_volume:
                    reference_candle = row
                    reference_volume = current_volume
                    logger.debug(f"{stock}: New reference at {candle_time.strftime('%H:%M')} - Vol: {current_volume:,.0f}")
        
        # STEP 2: After 9:25, look for target candles (but only if we have a reference)
        elif (candle_hour > 9 or (candle_hour == 9 and candle_minute > 25)) and reference_candle is not None:
            current_volume = float(row['Volume'])
            
            # If we're not currently monitoring a target, look for new ones
            if active_target is None and current_volume < reference_volume:
                is_green = float(row['Close']) > float(row['Open'])
                is_red = float(row['Close']) < float(row['Open'])
                
                if is_green or is_red:
                    active_target = row
                    target_candle_type = 'GREEN' if is_green else 'RED'
                    monitoring_countdown = 5  # Monitor next 5 candles
                    
                    logger.info(f"{stock}: 🎯 TARGET FOUND at {candle_time.strftime('%H:%M')} - {target_candle_type} "
                              f"(Vol: {current_volume:,.0f} < Ref: {reference_volume:,.0f})")
                    logger.info(f"{stock}: Target Open: {row['Open']:.2f}, Monitoring next 5 candles for breakout")
                    
                    # Skip to next candle (continue monitoring)
                    continue
        
        # STEP 3: If we have an active target, monitor for breakout
        if active_target is not None and monitoring_countdown > 0:
            monitoring_countdown -= 1
            
            # Check breakout conditions based on target type
            breakout_occurred = False
            current_close = float(row['Close'])
            target_open = float(active_target['Open'])
            
            if target_candle_type == 'GREEN':
                # SELL signal: Close goes below GREEN target's Open
                if current_close < target_open:
                    breakout_occurred = True
                    direction = 'SELL'
                    logger.info(f"{stock}: ✅ BREAKOUT at {candle_time.strftime('%H:%M')} - "
                              f"Close {current_close:.2f} < Target Open {target_open:.2f}")
            
            elif target_candle_type == 'RED':
                # BUY signal: Close goes above RED target's Open  
                if current_close > target_open:
                    breakout_occurred = True
                    direction = 'BUY'
                    logger.info(f"{stock}: ✅ BREAKOUT at {candle_time.strftime('%H:%M')} - "
                              f"Close {current_close:.2f} > Target Open {target_open:.2f}")
            
            if breakout_occurred:
                # Calculate stop loss and target
                entry_price = current_close
                
                # Find highest high (for SELL) or lowest low (for BUY) in the range
                target_idx = df_today[df_today['Date'] == active_target['Date']].index[0]
                entry_idx = idx
                
                if direction == 'SELL':
                    highest_high = float(active_target['High'])
                    for candle_idx in range(target_idx, entry_idx + 1):
                        candle_high = float(df_today.iloc[candle_idx]['High'])
                        if candle_high > highest_high:
                            highest_high = candle_high
                    stop_loss = highest_high + 0.05
                    target_price = entry_price - (stop_loss - entry_price)  # 1:1 RR
                else:  # BUY
                    lowest_low = float(active_target['Low'])
                    for candle_idx in range(target_idx, entry_idx + 1):
                        candle_low = float(df_today.iloc[candle_idx]['Low'])
                        if candle_low < lowest_low:
                            lowest_low = candle_low
                    stop_loss = lowest_low - 0.05
                    target_price = entry_price + (entry_price - stop_loss)  # 1:1 RR
                
                # Prepare result
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
                
                logger.info(f"{stock}: 🚀 TRADE SIGNAL | {direction} at {entry_price:.2f} | "
                          f"SL: {stop_loss:.2f} | Target: {target_price:.2f}")
                
                return res, False, True
        
        # STEP 4: If monitoring period ended without breakout, reset
        if active_target is not None and monitoring_countdown <= 0:
            logger.info(f"{stock}: ❌ No breakout detected for {target_candle_type} target at "
                      f"{active_target['Date'].strftime('%H:%M')} (Open: {active_target['Open']:.2f})")
            active_target = None
            target_candle_type = None

    # If we reach here, no trade signal was generated
    if reference_candle is None:
        logger.info(f"{stock}: No valid reference candle found (9:15, 9:20, 9:25)")
    elif active_target is None:
        logger.info(f"{stock}: No target candle found with volume lower than reference")
    
    return None, False, False

# -----------------------
# Bookkeeping functions
# -----------------------
def load_triggered_stocks(current_date):
    path = os.path.join('app', f'{current_date}_triggered_stocks.csv')
    if not os.path.exists(path):
        return []
    try:
        # Check if file is empty
        if os.path.getsize(path) == 0:
            return []
        with open(path, 'r') as f:
            stocks = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(stocks)} triggered stocks from file")
            return stocks
    except Exception as e:
        logger.error(f"load_triggered_stocks error: {e}")
        return []

def save_triggered_stocks(stocks, current_date):
    app_dir = 'app'
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
    path = os.path.join(app_dir, f'{current_date}_triggered_stocks.csv')
    existing = load_triggered_stocks(current_date)
    all_items = list(dict.fromkeys(existing + stocks))
    with open(path, 'w') as f:
        for s in all_items:
            f.write(f"{s}\n")

def load_invalid_stocks(current_date):
    path = os.path.join('app', f'{current_date}_invalid_stocks.csv')
    if not os.path.exists(path):
        return []
    try:
        # Check if file is empty
        if os.path.getsize(path) == 0:
            return []
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"load_invalid_stocks error: {e}")
        return []

def save_invalid_stocks(stocks, current_date):
    app_dir = 'app'
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
    path = os.path.join(app_dir, f'{current_date}_invalid_stocks.csv')
    existing = load_invalid_stocks(current_date)
    all_items = list(dict.fromkeys(existing + stocks))
    with open(path, 'w') as f:
        for s in all_items:
            f.write(f"{s}\n")

def load_active_trades(current_date):
    global active_trades
    active_trades = {}
    
    # Try to load from both files (regular and audit)
    paths = [
        os.path.join('app', f'{current_date}_active_trades.csv'),
        os.path.join('app', f'{current_date}_active_trades_audit.csv')
    ]
    
    for path in paths:
        if not os.path.exists(path):
            continue
            
        try:
            # Check if file is empty or contains only headers
            if os.path.getsize(path) == 0:
                logger.warning(f"Active trades file is empty: {path}")
                continue
                
            # Try to read the file
            df = pd.read_csv(path)
            if df.empty:
                logger.warning(f"Active trades file contains no data: {path}")
                continue
                
            # Get the latest status for each trade_id from audit file
            if 'audit_timestamp' in df.columns:
                df = df.sort_values('audit_timestamp').groupby('trade_id').last().reset_index()
            
            for _, row in df.iterrows():
                tid = row['trade_id']
                # Only load ACTIVE trades
                if row.get('status') == 'ACTIVE':
                    active_trades[tid] = {
                        'stock': row['stock'],
                        'entry_price': float(row['entry_price']),
                        'stop_loss': float(row['stop_loss']),
                        'target_price': float(row.get('target_price')) if pd.notna(row.get('target_price')) and row.get('target_price') != '' else None,
                        'quantity': int(row['quantity']),
                        'remaining_qty': int(row.get('remaining_qty', row['quantity'])),
                        'entry_time': row['entry_time'],
                        'status': row['status'],
                        'order_id': str(row.get('order_id', '')),
                        'sl_order_id': str(row.get('sl_order_id', '')),
                        'profit_booked': bool(row.get('profit_booked', False)),
                        'original_sl': float(row.get('original_sl', row['stop_loss']))
                    }
                    
            logger.info(f"Loaded {len(active_trades)} active trades from {path}")
            break  # Stop after successfully loading from one file
            
        except pd.errors.EmptyDataError:
            logger.warning(f"Active trades file is empty: {path}")
            continue
        except Exception as e:
            logger.error(f"load_active_trades error with {path}: {e}")
            continue

def save_active_trades(current_date, action="update", trade_id=None, reason=""):
    global active_trades
    app_dir = 'app'
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
    path = os.path.join(app_dir, f'{current_date}_active_trades_audit.csv')
    
    try:
        # Prepare current active trades data with audit info
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rows = []
        
        for tid, info in active_trades.items():
            row_data = {
                'trade_id': tid,
                'stock': info['stock'],
                'entry_price': info['entry_price'],
                'stop_loss': info['stop_loss'],
                'target_price': info.get('target_price', ''),
                'quantity': info['quantity'],  # signed (+BUY / -SELL)
                'remaining_qty': info.get('remaining_qty', info['quantity']),
                'entry_time': info['entry_time'],
                'status': info['status'],
                'order_id': info.get('order_id', ''),
                'sl_order_id': info.get('sl_order_id', ''),
                'profit_booked': info.get('profit_booked', False),
                'original_sl': info.get('original_sl', info['stop_loss']),
                'direction': info.get('direction', ''),  # NEW COLUMN
                # AUDIT FIELDS
                'audit_timestamp': current_time,
                'audit_action': action,
                'audit_reason': reason if tid == trade_id else '',
                'total_active_trades': len(active_trades)
            }

            rows.append(row_data)
        
        df_new = pd.DataFrame(rows)
        
        # Check if audit file exists
        if os.path.exists(path):
            # Read existing audit data and append new records
            df_existing = pd.read_csv(path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(path, index=False)
            logger.info(f"Appended {len(rows)} trade records to audit file")
        else:
            # Create new audit file
            df_new.to_csv(path, index=False)
            logger.info(f"Created new audit file with {len(rows)} trade records")
            
    except Exception as e:
        logger.error(f"save_active_trades error: {e}")

def save_signals_to_csv(signals, current_date):
    app_dir = 'app'
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
    path = os.path.join(app_dir, f'{current_date}_low_volume_breakout_signals.csv')
    
    try:
        # Convert to DataFrame
        df_new = pd.DataFrame(signals)
        
        # Add missing columns for backward compatibility
        required_cols = [
            'Stock', 'Entry_Point', 'Entry_Price', 'Stop_Loss',
            'Target_Price', 'Target_Candle_Time', 'Target_Candle_Open',
            'Target_Candle_Type', 'Direction'
        ]
        for col in required_cols:
            if col not in df_new.columns:
                df_new[col] = None
        
        # Check if file already exists
        if os.path.exists(path):
            # Read existing data
            df_existing = pd.read_csv(path)
            # Append new signals
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Remove duplicates based on Stock and Entry_Point
            df_combined = df_combined.drop_duplicates(subset=['Stock', 'Entry_Point'], keep='last')
            df_combined.to_csv(path, index=False)
            logger.info(f"Appended {len(signals)} signals to existing file {path}")
        else:
            # Create new file
            df_new.to_csv(path, index=False)
            logger.info(f"Created new signals file {path} with {len(signals)} signals")
        
        # Log the signals with target prices
        for signal in signals:
            target_price = signal.get('Target_Price', 'N/A')
            target_open = signal.get('Target_Candle_Open', 'N/A')
            logger.info(
                f"Signal: {signal['Stock']} - "
                f"Entry: {signal['Entry_Price']:.2f}, "
                f"SL: {signal['Stop_Loss']:.2f}, "
                f"Target: {target_price if target_price != 'N/A' else 'N/A'}, "
                f"TargetOpen: {target_open if target_open != 'N/A' else 'N/A'}"
            )
            
    except Exception as e:
        logger.error(f"save_signals_to_csv error: {e}")

# -----------------------
# Execution: place orders
# -----------------------
# -----------------------
# Execution: place orders
# -----------------------
def execute_trades(signals, ui_reference, current_date):
    """Place orders for signals with target price tracking"""
    global active_trades
    executed_count = 0

    if not signals:
        return 0

    if not hasattr(ui_reference, 'clients') or not ui_reference.clients:
        logger.warning("No clients available for order placement")
        return 0

    client_name, client_id, client = ui_reference.clients[0]
    
    # Get current positions to check for existing trades
    try:
        positions = client.get_positions()
        existing_positions = {}
        if positions:
            for pos in positions:
                netqty = int(float(pos.get('netqty', 0)))
                symbol = pos.get('tsym', '').replace('-EQ', '')
                if symbol and netqty != 0:
                    existing_positions[symbol] = netqty
        logger.info(f"Existing positions: {existing_positions}")
    except Exception as e:
        logger.error(f"Error checking broker positions: {e}")
        existing_positions = {}

    # Count active trades (both in our system and broker)
    active_trade_count = len(active_trades) + len(existing_positions)
    if active_trade_count >= MAX_TRADES_PER_DAY:
        logger.info(f"Already reached max {MAX_TRADES_PER_DAY} trades for the day")
        return 0

    remaining_slots = MAX_TRADES_PER_DAY - active_trade_count
    logger.info(f"Remaining trade slots: {remaining_slots}")

    for res in signals:
        if remaining_slots <= 0:
            break
            
        stock = res['Stock']
        direction = res.get('Direction', 'SELL')
        
        # Check if we already have active trade for this stock
        if any(t['stock'] == stock for t in active_trades.values()):
            logger.info(f"Already have active trade for {stock}, skipping")
            continue
        
        # Check if broker already has position for this stock
        if stock in existing_positions:
            netqty = existing_positions[stock]
            if (direction == 'SELL' and netqty < 0) or (direction == 'BUY' and netqty > 0):
                logger.info(f"Already have position for {stock} (qty: {netqty}), skipping")
                continue

        entry_price = float(res['Entry_Price'])
        stop_loss = float(res['Stop_Loss'])
        target_price = float(res.get('Target_Price', 0))
        qty = TRADE_QUANTITY

        try:
            trading_symbol = f"{stock}-EQ" if not stock.endswith("-EQ") else stock
            buy_or_sell = "S" if direction == 'SELL' else "B"
            print(trading_symbol)
            logger.info(f"Placing market order {buy_or_sell} for {trading_symbol} qty {qty}")
            order_result = client.place_order(
                buy_or_sell=buy_or_sell,
                product_type="I",
                exchange="NSE",
                tradingsymbol=trading_symbol,
                quantity=qty,
                price_type="LMT",  # ✅ CHANGE: Limit order instead of Market
                price=entry_price, # ✅ CHANGE: Use the actual entry price
                discloseqty=0,
                trigger_price=0,
                retention="DAY",                
                remarks=f"LowVolBreak_{stock}"
            )

            if not order_result or order_result.get('stat') != 'Ok':
                logger.error(f"Order failed for {stock}: {order_result}")
                continue

            order_id = order_result.get('norenordno', '')
            logger.info(f"Order placed for {stock}, order_id {order_id}")
      
            # Wait and verify order execution
            time.sleep(3)
            
            # Check if order was executed by verifying position
            position_created = False
            try:
                # Check positions again to see if our order was filled
                positions_after = client.get_positions()
                if positions_after:
                    for pos in positions_after:
                        pos_symbol = pos.get('tsym', '').replace('-EQ', '')
                        pos_netqty = int(float(pos.get('netqty', 0)))
                        if pos_symbol == stock:
                            if (direction == 'SELL' and pos_netqty < 0) or (direction == 'BUY' and pos_netqty > 0):
                                position_created = True
                                logger.info(f"Position verified for {stock}: qty {pos_netqty}")
                                break
            except Exception as e:
                logger.error(f"Error verifying position for {stock}: {e}")
                # Continue anyway but don't place SL order

            if not position_created:
                logger.warning(f"Position not created for {stock}, order may not be filled")
                continue

            # ✅ CORRECTED: Place proper STOP LOSS order (not reverse order)
            # For SELL trade: BUY stop loss (to cover short position)
            # For BUY trade: SELL stop loss (to exit long position)
            sl_side = "B" if direction == 'SELL' else "S"  # This is CORRECT
            sl_price = stop_loss
            
            logger.info(f"Placing SL order for {stock} at {sl_price:.2f}")
            sl_order_result = client.place_order(
                buy_or_sell=sl_side,
                product_type="I",
                exchange="NSE",
                tradingsymbol=trading_symbol,
                quantity=qty,
                discloseqty=0,
                price_type="SL-LMT",  # Stop Loss Limit order
                price=sl_price,
                trigger_price=sl_price,
                remarks=f"LowVolBreak_SL_{stock}"
            )

            sl_order_id = ""
            if sl_order_result and sl_order_result.get('stat') == 'Ok':
                sl_order_id = sl_order_result.get('norenordno', '')
                logger.info(f"SL order placed for {stock}, order_id {sl_order_id}")
            else:
                logger.error(f"SL order failed for {stock}: {sl_order_result}")

            trade_id = f"{stock}_{datetime.now().strftime('%H%M%S')}"

            # signed quantity (+ for BUY, - for SELL)
            signed_qty = qty if direction == "BUY" else -qty

            active_trades[trade_id] = {
                'stock': stock,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'quantity': signed_qty,       # signed qty
                'remaining_qty': signed_qty,  # signed remaining qty
                'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'ACTIVE',
                'order_id': order_id,
                'sl_order_id': sl_order_id,
                'profit_booked': False,
                'original_sl': stop_loss,
                'direction': direction        # explicit BUY/SELL
            }


            executed_count += 1
            remaining_slots -= 1
            logger.info(f"TRADE EXECUTED: {stock} {direction} Entry={entry_price:.2f} SL={stop_loss:.2f} Target={target_price:.2f} Qty={qty}")

            if hasattr(ui_reference, 'show_trade_alert'):
                ui_reference.show_trade_alert(stock, res['Entry_Point'], entry_price, stop_loss)

        except Exception as e:
            logger.exception(f"Failed to execute trade for {stock}: {e}")

    if executed_count > 0:
        save_active_trades(current_date, action="new_trades", reason="Trade executed")
    return executed_count

# -----------------------
# Monitoring & exits
# -----------------------
def monitor_active_trades(ui_reference, current_date):
    """Monitor active trades with real-time LTP every 5 seconds"""
    global active_trades
    
    if not active_trades:
        return

    if not hasattr(ui_reference, 'clients') or not ui_reference.clients:
        return
        
    client_name, client_id, client = ui_reference.clients[0]
    
    try:
        # Get real-time LTP for all active trade symbols
        active_symbols = [trade['stock'] for trade in active_trades.values() if trade['status'] == 'ACTIVE']
        
        if not active_symbols:
            return
            
        ltp_map = {}
        for stock in active_symbols:
            try:
                trading_symbol = f"{stock}-EQ"
                quote = client.get_quotes("NSE", trading_symbol)
                if quote and isinstance(quote, list) and len(quote) > 0:
                    ltp = float(quote[0].get('lp', 0))
                    ltp_map[stock] = ltp
                    logger.debug(f"LTP for {stock}: {ltp:.2f}")
            except Exception as e:
                logger.error(f"Failed to get LTP for {stock}: {e}")
                continue

        for tid, trade in list(active_trades.items()):
            if trade['status'] != 'ACTIVE':
                continue
                
            stock = trade['stock']
            current_ltp = ltp_map.get(stock, 0)
            
            if current_ltp == 0:
                continue
                
            entry_price = float(trade['entry_price'])
            stop_loss = float(trade['stop_loss'])
            target_price = float(trade.get('target_price', 0))
            direction = trade.get('direction', 'SELL')
            quantity = trade['quantity']
            profit_booked = trade.get('profit_booked', False)

            logger.debug(f"Monitoring {stock} | LTP: {current_ltp:.2f} | Entry: {entry_price:.2f}")

            # Check exit conditions
            exit_reason = None
            
            if direction == 'SELL':
                if current_ltp <= target_price and not profit_booked:
                    exit_reason = "TARGET_HIT"
                elif current_ltp >= stop_loss:
                    exit_reason = "SL_HIT"
                    
            elif direction == 'BUY':
                if current_ltp >= target_price and not profit_booked:
                    exit_reason = "TARGET_HIT"
                elif current_ltp <= stop_loss:
                    exit_reason = "SL_HIT"

            # Execute exit if condition met
            if exit_reason:
                execute_exit_trade(ui_reference, trade, current_ltp, exit_reason, current_date)
                
    except Exception as e:
        logger.error(f"Monitor error: {e}")

def execute_exit_trade(ui_reference, trade, exit_price, reason, current_date):
    """Execute trade exit"""
    global active_trades
    
    stock = trade['stock']
    direction = trade.get('direction', 'SELL')
    quantity = trade['quantity']
    entry_price = float(trade['entry_price'])
    
    trading_symbol = f"{stock}-EQ"
    buy_or_sell = "B" if direction == 'SELL' else "S"  # Reverse position
    
    try:
        client_name, client_id, client = ui_reference.clients[0]
        
        order_result = client.place_order(
            buy_or_sell=buy_or_sell,
            product_type="I",
            exchange="NSE",
            tradingsymbol=trading_symbol,
            quantity=quantity,
            price_type="MKT",
            price=0,
            discloseqty= 0,
            trigger_price=0, 
            retention="DAY",                       
            remarks=f"Exit_{reason}_{stock}"
        )
        
        if order_result and order_result.get('stat') == 'Ok':
            # Update trade status
            trade['status'] = 'EXITED'
            trade['exit_price'] = exit_price
            trade['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            trade['exit_reason'] = reason

            # Calculate PnL
            if direction == 'SELL':
                pnl = (entry_price - exit_price) * quantity
            else:
                pnl = (exit_price - entry_price) * quantity

            trade['pnl'] = pnl

            logger.info(f"{stock}: {reason} at {exit_price:.2f}. PnL: {pnl:+.2f}")

            # ✅ Log exit details with signed qty and direction
            log_profit_booking(
                stock=stock,
                entry_price=entry_price,
                exit_price=exit_price,
                qty_exited=quantity,
                remaining_qty=0,            # adjust if you implement partial exits
                current_date=current_date,
                reason=reason,
                direction=direction         # <-- NEW argument
            )

            # Simple audit logging
            save_active_trades(
                current_date,
                action="exit",
                trade_id=None,
                reason=f"{reason}_PnL_{pnl:+.2f}"
            )

        else:
            logger.error(f"{stock}: Exit order failed: {order_result}")

            
    except Exception as e:
        logger.error(f"{stock}: Error placing exit order: {e}")

def log_profit_booking(stock, entry_price, exit_price, qty_exited, remaining_qty, current_date, reason, direction):
    """Simple log function for all exits"""
    app_dir = 'app'
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
        
    log_file = os.path.join(app_dir, f'{current_date}_trade_exits.csv')
    
    pnl = (exit_price - entry_price) * qty_exited
    
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock': stock,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'quantity': qty_exited if direction == "BUY" else -qty_exited,   # signed
        'remaining_qty': remaining_qty if direction == "BUY" else -remaining_qty,
        'pnl': pnl,
        'direction': direction,   # ✅ NEW
        'exit_reason': reason
         }

    
    try:
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([log_entry])
        df.to_csv(log_file, index=False)
    except Exception as e:
        logger.error(f"Error logging trade exit: {e}")