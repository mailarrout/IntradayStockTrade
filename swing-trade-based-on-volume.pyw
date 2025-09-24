# swing-trade-based-on-volume.py

import os
import pandas as pd
import pytz
import json
import time
import subprocess
import sys
from datetime import datetime, timedelta
from client_loader import load_clients

# ================= CONFIGURATION =================
# File Paths
TRADES_CSV = "SwingTrade/swing_trading_positions.csv"
HISTORICAL_DATA_CSV = "SwingTrade/historical_price_data.csv"
SWING_TRADE_VOLUME_CSV = "SwingTrade/swing_trade_volume.csv"

# Strategy Parameters
RISK_REWARD_RATIO = 1
MAX_DAYS_AFTER_SIGNAL = 3
MAX_CANDLES_AFTER_LOW_VOLUME = 5
MAX_MONITORING_DAYS = 15
EXCHANGE = "NSE"

# STATUS CONSTANTS
STATUS_MONITORING = "Monitoring"
STATUS_TAKE_TRADE_NEXT_DAY = "Take trade Next day"
STATUS_IN_HOLDING = "In Holding"
STATUS_CLOSED = "Closed"
STATUS_EXPIRED = "Expired"

# COMMENT CONSTANTS
COMMENT_WAITING_FIRST_3_DAYS = "Waiting for first 3 days data"
COMMENT_WAITING_LOW_VOLUME = "Waiting for low volume candle (lower than first 3 days)"
COMMENT_WAITING_TARGET_CANDLE = "Waiting for target candle (lower volume + close < open)"
COMMENT_WAITING_BREAKOUT = "Waiting for breakout candle (close > target candle open)"
COMMENT_READY_FOR_ENTRY = "Ready for entry next day"
COMMENT_IN_POSITION = "In position - monitoring for exit"
COMMENT_TARGET_HIT = "Target hit"
COMMENT_SL_HIT = "Stop loss hit"
COMMENT_MONITORING_EXPIRED = "Monitoring period expired"
# ================================================

def run_chartink_scraper():
    """Run the Chartink scraper before processing"""
    print(" Running Chartink scraper to get latest signals...")
    # return True
    try:
        # Run the scraper script
        result = subprocess.run([
            sys.executable, "scanner-swing-trade-based-on-volume.py"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print(" Chartink scraper completed successfully")
            return True
        else:
            print(f"❌ Chartink scraper failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Chartink scraper timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running Chartink scraper: {e}")
        return False

def wait_for_swing_trade_file(timeout=300):
    """Wait for swing_trade_volume.csv to be loaded with timeout"""
    print(" Waiting for swing_trade_volume.csv to be loaded...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(SWING_TRADE_VOLUME_CSV):
            try:
                # Check if file has content and today's date
                df = pd.read_csv(SWING_TRADE_VOLUME_CSV)
                today_date = datetime.now().date().strftime('%Y-%m-%d')
                
                if not df.empty and today_date in df['date'].values:
                    print(f" swing_trade_volume.csv loaded with {len(df)} symbols for today")
                    return True
                else:
                    print("  File exists but no today's data yet, waiting...")
            except Exception as e:
                print(f"Error reading file: {e}, waiting...")
        
        time.sleep(10)  # Wait 10 seconds before checking again
    
    print("❌ Timeout waiting for swing_trade_volume.csv")
    return False

def read_swing_trade_signals():
    """Read signals from swing_trade_volume.csv with date-based filtering"""
    if not os.path.exists(SWING_TRADE_VOLUME_CSV):
        print(f"❌ Error: File not found at {os.path.abspath(SWING_TRADE_VOLUME_CSV)}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(SWING_TRADE_VOLUME_CSV)
        print(f"Successfully read {len(df)} signals from swing_trade_volume.csv")
        
        # Filter for today's signals only
        today_date = datetime.now().date().strftime('%Y-%m-%d')
        today_signals = df[df['date'] <= today_date]
        
        print(f"Found {len(today_signals)} signals for today ({today_date})")
        
        # Use date from CSV as signal_date
        today_signals = today_signals.copy()
        today_signals['signal_date'] = pd.to_datetime(today_signals['date']).dt.date
        today_signals['symbol'] = today_signals['symbol'].astype(str)
        
        # Add required columns with default values
        today_signals['sector'] = 'N/A'
        today_signals['marketcap'] = 'N/A'
        
        return today_signals
        
    except Exception as e:
        print(f"Error reading swing_trade_volume.csv: {e}")
        return pd.DataFrame()

def initialize_files():
    """Initialize both trades and historical data files"""
    os.makedirs(os.path.dirname(TRADES_CSV), exist_ok=True)
    
    if not os.path.exists(TRADES_CSV):
        columns = [
            'symbol', 'sector', 'marketcap', 'signal_date', 
            'day1_volume', 'day2_volume', 'day3_volume', 
            'low_volume_date', 'low_volume_volume', 'low_volume_open', 'low_volume_high', 'low_volume_low', 'low_volume_close',
            'target_candle_date', 'target_candle_volume', 'target_candle_open', 'target_candle_high', 'target_candle_low', 'target_candle_close',
            'entry_date', 'entry_price', 'entry_volume', 'entry_open', 'entry_high', 'entry_low', 'entry_close',
            'stop_loss', 'target_price', 
            'exit_date', 'exit_price', 'exit_volume', 'exit_open', 'exit_high', 'exit_low', 'exit_close',
            'exit_type', 'pnl', 'pnl_pct', 'status', 'comments', 'days_held', 'last_updated'
        ]
        pd.DataFrame(columns=columns).to_csv(TRADES_CSV, index=False)
        print(f"Initialized new trades file: {TRADES_CSV}")
    
    if not os.path.exists(HISTORICAL_DATA_CSV):
        columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        pd.DataFrame(columns=columns).to_csv(HISTORICAL_DATA_CSV, index=False)
        print(f"Initialized historical data file: {HISTORICAL_DATA_CSV}")

def read_trading_positions():
    """Read existing trading positions from CSV with date-based deduplication"""
    if not os.path.exists(TRADES_CSV):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(TRADES_CSV)
        
        # Remove duplicates based on symbol and last_updated date (keep latest)
        if 'last_updated' in df.columns:
            df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
            df = df.sort_values('last_updated').drop_duplicates(subset=['symbol'], keep='last')
        
        date_cols = ['signal_date', 'low_volume_date', 'target_candle_date',
                    'entry_date', 'exit_date', 'last_updated']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
        return df
    except Exception as e:
        print(f"Error reading trades file: {e}")
        return pd.DataFrame()

def save_trading_positions(df):
    """Save trading positions to CSV with date-based deduplication"""
    # Remove duplicates before saving
    if 'last_updated' in df.columns:
        df = df.sort_values('last_updated').drop_duplicates(subset=['symbol'], keep='last')
    df.to_csv(TRADES_CSV, index=False)
    print(f"Saved {len(df)} positions to {TRADES_CSV}")

def read_historical_data(symbol=None):
    """Read historical price data from CSV"""
    if not os.path.exists(HISTORICAL_DATA_CSV):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(HISTORICAL_DATA_CSV)
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        if symbol:
            df = df[df['symbol'] == symbol]
        
        return df.sort_values(['symbol', 'date'])
    except Exception as e:
        print(f"Error reading historical data: {e}")
        return pd.DataFrame()

def save_historical_data(new_data):
    """Save or update historical price data"""
    if os.path.exists(HISTORICAL_DATA_CSV):
        existing_data = pd.read_csv(HISTORICAL_DATA_CSV)
        existing_data['date'] = pd.to_datetime(existing_data['date']).dt.date
    else:
        existing_data = pd.DataFrame()
    
    if not existing_data.empty:
        combined = pd.concat([existing_data, new_data]).drop_duplicates(
            subset=['symbol', 'date'], keep='last'
        )
    else:
        combined = new_data
    
    combined.to_csv(HISTORICAL_DATA_CSV, index=False)
    return combined

def validate_daily_data(data, symbol, date):
    """Validate that we have complete OHLCV data for the day"""
    if data is None or len(data) == 0:
        raise ValueError(f"❌ No data returned for {symbol} on {date}")
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in data.columns or data[col].isnull().any():
            raise ValueError(f"❌ Incomplete data for {symbol} on {date} - missing {col}")
        
        if col != 'volume' and (data[col] <= 0).any():
            raise ValueError(f"❌ Invalid {col} value for {symbol} on {date}")
    
    return True

def fetch_and_store_daily_data(symbol, start_date, end_date):
    """Fetch daily data only if dates are valid"""
    # Check if we're trying to fetch invalid date range or future dates
    today = datetime.now().date()
    if start_date > end_date:
        print(f"  ⚠️  No new data needed for {symbol} (start_date > end_date)")
        return read_historical_data(symbol)
    
    # Don't try to fetch future dates
    if start_date > today or end_date > today:
        end_date = min(end_date, today)
        if start_date > end_date:
            print(f"  ⚠️  No new data needed for {symbol} (dates in future)")
            return read_historical_data(symbol)
    try:
        ui_ref = type('Obj', (object,), {})()
        ui_ref.clients = []
        load_clients(ui_ref, auto_load=True)
        api = ui_ref.clients[0][2]
        
        ist = pytz.timezone("Asia/Kolkata")
        start_dt = ist.localize(datetime.combine(start_date, datetime.min.time()))
        end_dt = ist.localize(datetime.combine(end_date, datetime.max.time()))
        
        start_epoch = int(start_dt.timestamp())
        end_epoch = int(end_dt.timestamp())
        
        print(f"  Fetching data for {symbol} from {start_date} to {end_date}...")
        
        bulk_data = api.get_daily_price_series(
            exchange=EXCHANGE,
            tradingsymbol=f"{symbol}-EQ",
            startdate=start_epoch,
            enddate=end_epoch
        )
        
        if bulk_data and isinstance(bulk_data, list) and len(bulk_data) > 0:
            all_days_data = []
            
            for data_item in bulk_data:
                if isinstance(data_item, str):
                    try:
                        data_dict = json.loads(data_item)
                        
                        price_data = {
                            'symbol': symbol,
                            'date': pd.to_datetime(data_dict.get('time', ''), format='%d-%b-%Y').date(),
                            'open': float(data_dict.get('into', 0)),
                            'high': float(data_dict.get('inth', 0)),
                            'low': float(data_dict.get('intl', 0)),
                            'close': float(data_dict.get('intc', 0)),
                            'volume': float(data_dict.get('intv', 0))
                        }
                        
                        temp_df = pd.DataFrame([price_data])
                        validate_daily_data(temp_df, symbol, price_data['date'])
                        
                        all_days_data.append(price_data)
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        continue
            
            if all_days_data:
                new_df = pd.DataFrame(all_days_data)
                saved_data = save_historical_data(new_df)
                print(f"  Stored {len(new_df)} days of validated data for {symbol}")
                return saved_data[saved_data['symbol'] == symbol]
        
        raise ValueError(f"No bulk data returned for {symbol} from {start_date} to {end_date}")
        
    except Exception as e:
        print(f"  ❌ Fetch error for {symbol}: {e}")
        raise

def get_todays_data(symbol):
    """Get today's data specifically with validation"""
    today = datetime.now().date()
    
    # First check if we already have today's data
    existing_data = read_historical_data(symbol)
    if not existing_data.empty and today in existing_data['date'].values:
        today_data = existing_data[existing_data['date'] == today]
        validate_daily_data(today_data, symbol, today)
        return today_data
    
    # Only fetch if we don't have today's data yet
    try:
        data = fetch_and_store_daily_data(symbol, today, today)
        if data is not None and len(data) > 0:
            validate_daily_data(data, symbol, today)
        return data
    except Exception as e:
        print(f"  ⚠️  Could not fetch today's data for {symbol}: {e}")
        # Return existing data if available
        return existing_data[existing_data['date'] == today] if not existing_data.empty else None

def get_volume_for_dates(historical_data, signal_date):
    """Get volumes for available dates from signal date onward"""
    volumes = []
    
    # Check up to 3 days from signal date, but only if data exists
    for i in range(3):
        check_date = signal_date + timedelta(days=i)
        day_data = historical_data[historical_data['date'] == check_date]
        
        if not day_data.empty:
            validate_daily_data(day_data, historical_data.iloc[0]['symbol'], check_date)
            volumes.append(day_data['volume'].iloc[0])
        else:
            volumes.append(None)
    
    # Return volumes for available days (could be 1, 2, or 3 values)
    return volumes[0] if len(volumes) > 0 else None, \
           volumes[1] if len(volumes) > 1 else None, \
           volumes[2] if len(volumes) > 2 else None

def check_for_new_signals():
    """Check for new signals from swing_trade_volume.csv and add them to monitoring"""
    print("🔍 Checking for new signals from swing_trade_volume.csv...")
    signals_df = read_swing_trade_signals()
    positions_df = read_trading_positions()
    
    if signals_df.empty:
        print("No signals found in swing_trade_volume.csv")
        return positions_df
    
    # Filter out symbols that are already being monitored (not closed/expired)
    active_statuses = [STATUS_MONITORING, STATUS_TAKE_TRADE_NEXT_DAY, STATUS_IN_HOLDING]
    active_symbols = positions_df[positions_df['status'].isin(active_statuses)]['symbol'].unique() if not positions_df.empty else []
    
    # DEBUG: Show what symbols are being compared
    print(f"Signals symbols: {signals_df['symbol'].tolist()}")
    print(f"Active symbols: {active_symbols.tolist() if len(active_symbols) > 0 else 'None'}")
    
    # Filter out symbols that are already active (not closed)
    new_signals = signals_df[~signals_df['symbol'].isin(active_symbols)]
    
    print(f"New symbols to add: {new_signals['symbol'].tolist() if not new_signals.empty else 'None'}")
    
    if new_signals.empty:
        print("No new symbols to monitor (all signals are already being tracked)")
        return positions_df
    
    print(f"Found {len(new_signals)} new signals to monitor")
    
    for _, signal in new_signals.iterrows():
        symbol = signal['symbol']
        signal_date = signal['signal_date']
        
        print(f"  Processing {symbol} (Signal date: {signal_date})")
        
        try:
            # Fetch data from signal date to today
            today = datetime.now().date()
            historical_data = fetch_and_store_daily_data(symbol, signal_date, today)
            
            # Even if no data is available, still add the symbol to monitoring
            if historical_data is None or historical_data.empty:
                day1_volume, day2_volume, day3_volume = None, None, None
                print(f"   No data available yet for {symbol} - adding to monitoring anyway")
            else:
                day1_volume, day2_volume, day3_volume = get_volume_for_dates(historical_data, signal_date)
            
            new_position = {
                'symbol': symbol,
                'sector': signal.get('sector', 'N/A'),
                'marketcap': signal.get('marketcap', 'N/A'),
                'signal_date': signal_date,
                'day1_volume': day1_volume,
                'day2_volume': day2_volume,
                'day3_volume': day3_volume,
                'low_volume_date': None,
                'low_volume_volume': None,
                'low_volume_open': None,
                'low_volume_high': None,
                'low_volume_low': None,
                'low_volume_close': None,
                'target_candle_date': None,
                'target_candle_volume': None,
                'target_candle_open': None,
                'target_candle_high': None,
                'target_candle_low': None,
                'target_candle_close': None,
                'entry_date': None,
                'entry_price': None,
                'entry_volume': None,
                'entry_open': None,
                'entry_high': None,
                'entry_low': None,
                'entry_close': None,
                'stop_loss': None,
                'target_price': None,
                'exit_date': None,
                'exit_price': None,
                'exit_volume': None,
                'exit_open': None,
                'exit_high': None,
                'exit_low': None,
                'exit_close': None,
                'exit_type': None,
                'pnl': None,
                'pnl_pct': None,
                'status': STATUS_MONITORING,
                'comments': "Waiting for market data" if day1_volume is None else COMMENT_WAITING_FIRST_3_DAYS,
                'days_held': 0,
                'last_updated': datetime.now().date()
            }
            
            positions_df = pd.concat([positions_df, pd.DataFrame([new_position])], ignore_index=True)
            print(f"  ✅ Added {symbol} to monitoring (Signal: {signal_date})")
        
        except Exception as e:
            print(f"  ❌ Failed to add {symbol} to monitoring: {e}")
            continue
    
    save_trading_positions(positions_df)
    return positions_df

def find_initial_low_volume_candle(historical_data, signal_date):
    """Find initial lowest volume candle within available days AFTER signal date"""
    today = datetime.now().date()
    analysis_start = signal_date
    analysis_end = min(signal_date + timedelta(days=MAX_DAYS_AFTER_SIGNAL), today)
    
    analysis_data = historical_data[
        (historical_data['date'] >= analysis_start) & 
        (historical_data['date'] <= analysis_end)
    ]
    
    if analysis_data.empty:
        return None
    
    for _, candle in analysis_data.iterrows():
        validate_daily_data(pd.DataFrame([candle]), historical_data.iloc[0]['symbol'], candle['date'])
    
    low_volume_candle = analysis_data.loc[analysis_data['volume'].idxmin()]
    return low_volume_candle

def find_target_candle(historical_data, current_target_volume, current_target_date):
    """Find target candle - continuously look for lower volume with close < open"""
    if current_target_date:
        subsequent_data = historical_data[historical_data['date'] > current_target_date]
    else:
        return None
    
    if subsequent_data.empty:
        return None
    
    target_candle = None
    
    for _, candle in subsequent_data.iterrows():
        validate_daily_data(pd.DataFrame([candle]), historical_data.iloc[0]['symbol'], candle['date'])
        
        if candle['volume'] < current_target_volume and candle['close'] < candle['open']:
            target_candle = candle
            current_target_volume = candle['volume']
            print(f"  New target candle: {candle['date']} (Vol: {candle['volume']:,.0f}, Close < Open: ✓)")
    
    return target_candle

def find_entry_candle(historical_data, target_candle):
    """Find entry candle - first candle where close > target candle's open"""
    target_date = target_candle['date']
    
    subsequent_data = historical_data[historical_data['date'] > target_date].head(MAX_CANDLES_AFTER_LOW_VOLUME)
    
    if subsequent_data.empty:
        return None
    
    for _, candle in subsequent_data.iterrows():
        validate_daily_data(pd.DataFrame([candle]), historical_data.iloc[0]['symbol'], candle['date'])
        
        if candle['close'] > target_candle['open']:
            return candle
    
    return None

def process_monitoring_stocks(positions_df):
    """Process stocks that are in monitoring status"""
    print("\n Processing monitoring stocks...")
    
    monitoring_stocks = positions_df[positions_df['status'] == STATUS_MONITORING]
    
    if monitoring_stocks.empty:
        print("No stocks in monitoring status")
        return positions_df
    
    today = datetime.now().date()
    
    for idx, position in monitoring_stocks.iterrows():
        symbol = position['symbol']
        signal_date = position['signal_date']
        
        print(f"\n  Processing {symbol} (Signal: {signal_date})")
        
        # Check monitoring period
        days_since_signal = (today - signal_date).days
        
        if days_since_signal > MAX_MONITORING_DAYS:
            print(f"   Monitoring period expired ({days_since_signal} days)")
            positions_df.loc[idx, 'status'] = STATUS_EXPIRED
            positions_df.loc[idx, 'comments'] = COMMENT_MONITORING_EXPIRED
            positions_df.loc[idx, 'last_updated'] = today
            continue
        
        try:
            # Fetch complete data from signal_date to today for proper analysis
            start_date = signal_date
            end_date = today

            # Ensure we don't try to fetch future dates
            if start_date > end_date:
                print(f"  ⚠️  No new data needed for {symbol} (start_date > end_date)")
                historical_data = read_historical_data(symbol)
            else:
                # Fetch complete range of data for proper analysis
                print(f"  📊 Fetching complete data for {symbol} from {start_date} to {end_date}")
                historical_data = fetch_and_store_daily_data(symbol, start_date, end_date)
            
            if historical_data is None or historical_data.empty:
                print(f"   No data available for {symbol}")
                positions_df.loc[idx, 'comments'] = "Error: No data available"
                positions_df.loc[idx, 'last_updated'] = today
                continue
            
            print(f"  📊 Available data: {len(historical_data)} days (from {historical_data['date'].min()} to {historical_data['date'].max()})")
            
            # Check if we have first 3 days data
            if pd.isna(position['day1_volume']) or pd.isna(position['day2_volume']) or pd.isna(position['day3_volume']):
                print(f"  ⏳ Waiting for first 3 days data")
                
                # Get volumes for first 3 available days after signal date
                day1_volume, day2_volume, day3_volume = get_volume_for_dates(historical_data, signal_date)
                
                # Update volumes if we have the data
                if day1_volume is not None:
                    positions_df.loc[idx, 'day1_volume'] = day1_volume
                if day2_volume is not None:
                    positions_df.loc[idx, 'day2_volume'] = day2_volume
                if day3_volume is not None:
                    positions_df.loc[idx, 'day3_volume'] = day3_volume
                
                # Check if we now have all 3 days of data
                if (pd.notna(positions_df.loc[idx, 'day1_volume']) and 
                    pd.notna(positions_df.loc[idx, 'day2_volume']) and 
                    pd.notna(positions_df.loc[idx, 'day3_volume'])):
                    print(f"  ✅ Now have all 3 days of volume data")
                    positions_df.loc[idx, 'comments'] = COMMENT_WAITING_LOW_VOLUME
                else:
                    positions_df.loc[idx, 'comments'] = COMMENT_WAITING_FIRST_3_DAYS
                
                positions_df.loc[idx, 'last_updated'] = today
                continue
            
            # STEP 1: Find initial low volume candle
            low_volume_candle = find_initial_low_volume_candle(historical_data, signal_date)
            
            if low_volume_candle is None:
                print(f"   No initial low volume candle found after signal")
                positions_df.loc[idx, 'comments'] = COMMENT_WAITING_LOW_VOLUME
                positions_df.loc[idx, 'last_updated'] = today
                continue
            
            print(f"  ✅ Initial low volume: {low_volume_candle['date']} (Vol: {low_volume_candle['volume']:,.0f})")
            
            # Update low volume candle details
            positions_df.loc[idx, 'low_volume_date'] = low_volume_candle['date']
            positions_df.loc[idx, 'low_volume_volume'] = low_volume_candle['volume']
            positions_df.loc[idx, 'low_volume_open'] = low_volume_candle['open']
            positions_df.loc[idx, 'low_volume_high'] = low_volume_candle['high']
            positions_df.loc[idx, 'low_volume_low'] = low_volume_candle['low']
            positions_df.loc[idx, 'low_volume_close'] = low_volume_candle['close']
            
            # STEP 2: Continuously look for target candles
            current_target_volume = low_volume_candle['volume']
            current_target_date = low_volume_candle['date']
            target_candle = low_volume_candle
            
            while True:
                new_target_candle = find_target_candle(historical_data, current_target_volume, current_target_date)
                if new_target_candle is None:
                    break
                target_candle = new_target_candle
                current_target_volume = target_candle['volume']
                current_target_date = target_candle['date']
            
            if target_candle['date'] == low_volume_candle['date']:
                print(f"   No better target candle found yet")
                positions_df.loc[idx, 'comments'] = COMMENT_WAITING_TARGET_CANDLE
                positions_df.loc[idx, 'last_updated'] = today
                continue
            
            print(f"   Final target candle: {target_candle['date']} (Vol: {target_candle['volume']:,.0f}, Close < Open: ✓)")
            
            # Update target candle details
            positions_df.loc[idx, 'target_candle_date'] = target_candle['date']
            positions_df.loc[idx, 'target_candle_volume'] = target_candle['volume']
            positions_df.loc[idx, 'target_candle_open'] = target_candle['open']
            positions_df.loc[idx, 'target_candle_high'] = target_candle['high']
            positions_df.loc[idx, 'target_candle_low'] = target_candle['low']
            positions_df.loc[idx, 'target_candle_close'] = target_candle['close']
            
            # STEP 3: Find entry candle
            entry_candle = find_entry_candle(historical_data, target_candle)
            
            if entry_candle is None:
                print(f"  No entry signal yet (waiting for close > {target_candle['open']:.2f})")
                positions_df.loc[idx, 'comments'] = COMMENT_WAITING_BREAKOUT
                positions_df.loc[idx, 'last_updated'] = today
                continue
            
            # ENTRY SIGNAL FOUND
            entry_price = entry_candle['close']
            stop_loss = min(low_volume_candle['low'], target_candle['low'])
            risk = entry_price - stop_loss
            
            if risk <= 0:
                print(f"  Invalid risk calculation")
                positions_df.loc[idx, 'last_updated'] = today
                continue
                
            target_price = entry_price + (RISK_REWARD_RATIO * risk)
            
            # Update position with all entry details
            positions_df.loc[idx, 'entry_date'] = entry_candle['date']
            positions_df.loc[idx, 'entry_price'] = entry_price
            positions_df.loc[idx, 'entry_volume'] = entry_candle['volume']
            positions_df.loc[idx, 'entry_open'] = entry_candle['open']
            positions_df.loc[idx, 'entry_high'] = entry_candle['high']
            positions_df.loc[idx, 'entry_low'] = entry_candle['low']
            positions_df.loc[idx, 'entry_close'] = entry_candle['close']
            positions_df.loc[idx, 'stop_loss'] = stop_loss
            positions_df.loc[idx, 'target_price'] = target_price
            positions_df.loc[idx, 'status'] = STATUS_TAKE_TRADE_NEXT_DAY
            positions_df.loc[idx, 'comments'] = COMMENT_READY_FOR_ENTRY
            positions_df.loc[idx, 'last_updated'] = today
            
            print(f"  ENTRY SIGNAL - Trade next day at {entry_price:.2f}")
            print(f"  SL: {stop_loss:.2f}, Target: {target_price:.2f}")
        
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")
            positions_df.loc[idx, 'comments'] = f"Error: {str(e)}"
            positions_df.loc[idx, 'last_updated'] = today
            continue
    
    save_trading_positions(positions_df)
    return positions_df

def execute_pending_trades(positions_df):
    """Execute trades marked as 'Take trade Next day'"""
    print("\n Executing pending trades...")
    
    pending_trades = positions_df[positions_df['status'] == STATUS_TAKE_TRADE_NEXT_DAY]
    
    if pending_trades.empty:
        print("No pending trades to execute")
        return positions_df
    
    today = datetime.now().date()
    
    for idx, position in pending_trades.iterrows():
        symbol = position['symbol']
        entry_date = position['entry_date']
        
        if pd.isna(entry_date) or entry_date != today:
            print(f"   {symbol}: Not yet time to execute (Entry: {entry_date}, Today: {today})")
            continue
        
        print(f"   {symbol}: Executing trade at {position['entry_price']:.2f}")
        positions_df.loc[idx, 'status'] = STATUS_IN_HOLDING
        positions_df.loc[idx, 'comments'] = COMMENT_IN_POSITION
        positions_df.loc[idx, 'last_updated'] = today
    
    save_trading_positions(positions_df)
    return positions_df

def monitor_held_positions(positions_df):
    """Monitor positions that are in holding and check for exits"""
    print("\n Monitoring held positions...")
    
    held_positions = positions_df[positions_df['status'] == STATUS_IN_HOLDING]
    
    if held_positions.empty:
        print("No positions in holding")
        return positions_df
    
    today = datetime.now().date()
    
    for idx, position in held_positions.iterrows():
        symbol = position['symbol']
        
        try:
            today_data = get_todays_data(symbol)
            
            if today_data is None or len(today_data) == 0:
                print(f" {symbol}: No data available for today yet")
                # Still update last_updated but don't process exit conditions
                positions_df.loc[idx, 'last_updated'] = today
                positions_df.loc[idx, 'comments'] = "Waiting for today's data"
                continue
            
                today_candle = today_data.iloc[0]
                
                # Check for exit conditions
                exit_info = None
                
                if today_candle['high'] >= target_price:
                    exit_info = {
                        'date': today,
                        'type': 'Target',
                        'price': target_price,
                        'pnl': target_price - entry_price,
                        'volume': today_candle['volume'],
                        'open': today_candle['open'],
                        'high': today_candle['high'],
                        'low': today_candle['low'],
                        'close': today_candle['close']
                    }
                    positions_df.loc[idx, 'comments'] = COMMENT_TARGET_HIT
                elif today_candle['low'] <= stop_loss:
                    exit_info = {
                        'date': today,
                        'type': 'SL',
                        'price': stop_loss,
                        'pnl': stop_loss - entry_price,
                        'volume': today_candle['volume'],
                        'open': today_candle['open'],
                        'high': today_candle['high'],
                        'low': today_candle['low'],
                        'close': today_candle['close']
                    }
                    positions_df.loc[idx, 'comments'] = COMMENT_SL_HIT
                
                if exit_info:
                    pnl_pct = (exit_info['pnl'] / entry_price) * 100
                    days_held = (today - entry_date).days
                    
                    print(f" {symbol}: {exit_info['type']} HIT at {exit_info['price']:.2f}")
                    print(f"  P&L: {exit_info['pnl']:.2f} points ({pnl_pct:.1f}%), Days: {days_held}")
                    
                    # Update position with complete exit details
                    positions_df.loc[idx, 'exit_date'] = exit_info['date']
                    positions_df.loc[idx, 'exit_price'] = exit_info['price']
                    positions_df.loc[idx, 'exit_volume'] = exit_info['volume']
                    positions_df.loc[idx, 'exit_open'] = exit_info['open']
                    positions_df.loc[idx, 'exit_high'] = exit_info['high']
                    positions_df.loc[idx, 'exit_low'] = exit_info['low']
                    positions_df.loc[idx, 'exit_close'] = exit_info['close']
                    positions_df.loc[idx, 'exit_type'] = exit_info['type']
                    positions_df.loc[idx, 'pnl'] = exit_info['pnl']
                    positions_df.loc[idx, 'pnl_pct'] = pnl_pct
                    positions_df.loc[idx, 'days_held'] = days_held
                    positions_df.loc[idx, 'status'] = STATUS_CLOSED
                    positions_df.loc[idx, 'last_updated'] = today
                
                else:
                    # Update days held
                    days_held = (today - entry_date).days
                    positions_df.loc[idx, 'days_held'] = days_held
                    positions_df.loc[idx, 'last_updated'] = today
                    positions_df.loc[idx, 'comments'] = COMMENT_IN_POSITION
                    
                    current_pnl = today_candle['close'] - entry_price
                    current_pnl_pct = (current_pnl / entry_price) * 100
                    
                    print(f" {symbol}: Still holding - P&L: {current_pnl:.2f} ({current_pnl_pct:.1f}%), Days: {days_held}")
            
        except Exception as e:
            print(f"  ❌ Error monitoring {symbol}: {e}")
            continue
        
        save_trading_positions(positions_df)
        return positions_df

def run_daily_trading_system():
    """Main function to run the daily trading system with Chartink integration"""
    print("=" * 80)
    print(" DAILY TRADING SYSTEM STARTED")
    print(f"Date: {datetime.now().date()}")
    print("=" * 80)
    
    try:
        # STEP 1: Run Chartink scraper first
        if not run_chartink_scraper():
            print("❌ Cannot proceed without Chartink data")
            return
        
        # STEP 2: Wait for CSV file to be loaded
        if not wait_for_swing_trade_file():
            print(" Cannot proceed without swing_trade_volume.csv")
            return
        
        # STEP 3: Initialize and run your existing system
        initialize_files()
        positions_df = read_trading_positions()
        
        positions_df = check_for_new_signals()
        positions_df = process_monitoring_stocks(positions_df)
        positions_df = execute_pending_trades(positions_df)
        positions_df = monitor_held_positions(positions_df)
        
        # Print summary
        print("\n" + "=" * 80)
        print(" DAILY SUMMARY")
        print("=" * 80)
        
        status_counts = positions_df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        # Print comments summary
        print(f"\n   Current Status Comments:")
        for status in [STATUS_MONITORING, STATUS_TAKE_TRADE_NEXT_DAY, STATUS_IN_HOLDING]:
            status_positions = positions_df[positions_df['status'] == status]
            if not status_positions.empty:
                for _, pos in status_positions.iterrows():
                    print(f"     {pos['symbol']}: {pos['comments']}")
        
        closed_positions = positions_df[positions_df['status'] == STATUS_CLOSED]
        if not closed_positions.empty:
            total_pnl = closed_positions['pnl'].sum()
            avg_pnl = closed_positions['pnl'].mean()
            winning_trades = len(closed_positions[closed_positions['pnl'] > 0])
            win_rate = (winning_trades / len(closed_positions)) * 100
            
            print(f"\n   Closed Trades Performance:")
            print(f"     Total P&L: {total_pnl:.2f} points")
            print(f"     Average P&L: {avg_pnl:.2f} points")
            print(f"     Win Rate: {win_rate:.1f}% ({winning_trades}/{len(closed_positions)})")
        
        print(f"\n Daily trading system completed at {datetime.now()}")
    
    except Exception as e:
        print(f"CRITICAL ERROR in trading system: {e}")
        raise

# Run the daily trading system
if __name__ == "__main__":
    run_daily_trading_system()