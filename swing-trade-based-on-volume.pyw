import os
import pandas as pd
import pytz
import json
import time
from datetime import datetime, timedelta
from client_loader import load_clients

# ================= CONFIGURATION =================
RISK_REWARD_RATIO = 1  # Change this to 1 for 1:1, 2 for 1:2, 3 for 1:3, etc.
MAX_DAYS_AFTER_REFERENCE = 15  # Only look for low volume candles within 3 days after reference
MAX_CANDLES_AFTER_LOW_VOLUME = 3  # Maximum candles allowed after low volume for valid entry
TRADES_CSV = "app/swing_trading_positions.csv"  # File to track all trading positions in app directory
# ================================================

def initialize_trades_file():
    """Initialize the trades CSV file if it doesn't exist"""
    # Create app directory if it doesn't exist
    os.makedirs(os.path.dirname(TRADES_CSV), exist_ok=True)
    
    if not os.path.exists(TRADES_CSV):
        columns = [
            'symbol', 'sector', 'marketcap', 'signal_date', 'reference_date',
            'low_volume_date', 'entry_date', 'entry_price', 'stop_loss', 
            'target_price', 'exit_date', 'exit_price', 'exit_type', 'pnl', 
            'pnl_pct', 'status', 'days_held', 'last_updated'
        ]
        pd.DataFrame(columns=columns).to_csv(TRADES_CSV, index=False)
        print(f"Initialized new trades file: {TRADES_CSV}")

def read_backtest_signals():
    """Read backtest signals from CSV file"""
    file_path = "app/Backtest Swing Trade Based on Volume, Technical Analysis Scanner.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {os.path.abspath(file_path)}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully read {len(df)} signals")
        
        # Parse date column
        df['signal_date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %I:%M %p', errors='coerce')
        df = df.dropna(subset=['signal_date'])
        
        return df
        
    except Exception as e:
        print(f"Error reading signals file: {e}")
        return pd.DataFrame()

def read_trading_positions():
    """Read existing trading positions from CSV"""
    if not os.path.exists(TRADES_CSV):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(TRADES_CSV)
        # Convert date columns from string to datetime
        date_cols = ['signal_date', 'reference_date', 'low_volume_date', 
                    'entry_date', 'exit_date', 'last_updated']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"Error reading trades file: {e}")
        return pd.DataFrame()

def save_trading_positions(df):
    """Save trading positions to CSV"""
    df.to_csv(TRADES_CSV, index=False)
    print(f"Saved {len(df)} positions to {TRADES_CSV}")

def fetch_bulk_daily_data(symbol, start_date, end_date):
    """Fetch multiple days of data in a single API call using date range"""
    try:
        ui_ref = type('Obj', (object,), {})()
        ui_ref.clients = []
        load_clients(ui_ref, auto_load=True)
        api = ui_ref.clients[0][2]
        
        # Convert dates to timestamps
        ist = pytz.timezone("Asia/Kolkata")
        start_dt = ist.localize(datetime.combine(start_date, datetime.min.time()))
        end_dt = ist.localize(datetime.combine(end_date, datetime.max.time()))
        
        start_epoch = int(start_dt.timestamp())
        end_epoch = int(end_dt.timestamp())
        
        print(f"  Fetching bulk data for {symbol} from {start_date} to {end_date}...")
        
        # Single API call for the entire date range
        bulk_data = api.get_daily_price_series(
            exchange="NSE",
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
                        df = pd.DataFrame([data_dict])
                        
                        # Convert numeric columns
                        numeric_cols = ['into', 'inth', 'intl', 'intc', 'intv']
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Parse date from time column
                        if 'time' in df.columns:
                            df['date'] = pd.to_datetime(df['time'], format='%d-%b-%Y', errors='coerce').dt.date
                        
                        all_days_data.append(df)
                        
                    except json.JSONDecodeError:
                        continue
            
            if all_days_data:
                combined_df = pd.concat(all_days_data, ignore_index=True)
                print(f"  ✅ Retrieved {len(combined_df)} days in bulk")
                return combined_df
        
        print(f"  ❌ No bulk data returned")
        return None
        
    except Exception as e:
        print(f"  ❌ Bulk fetch error: {e}")
        return None

def check_for_new_signals():
    """Check for new signals and add them to monitoring"""
    print("🔍 Checking for new signals...")
    signals_df = read_backtest_signals()
    positions_df = read_trading_positions()
    
    if signals_df.empty:
        print("No new signals found")
        return positions_df
    
    # Check which signals are already being tracked
    existing_symbols = positions_df['symbol'].unique() if not positions_df.empty else []
    new_signals = signals_df[~signals_df['symbol'].isin(existing_symbols)]
    
    if new_signals.empty:
        print("No new symbols to monitor")
        return positions_df
    
    # Add new signals to monitoring
    print(f"Found {len(new_signals)} new signals to monitor")
    
    for _, signal in new_signals.iterrows():
        new_position = {
            'symbol': signal['symbol'],
            'sector': signal.get('sector', 'N/A'),
            'marketcap': signal.get('marketcapname', 'N/A'),
            'signal_date': signal['signal_date'],
            'reference_date': None,
            'low_volume_date': None,
            'entry_date': None,
            'entry_price': None,
            'stop_loss': None,
            'target_price': None,
            'exit_date': None,
            'exit_price': None,
            'exit_type': None,
            'pnl': None,
            'pnl_pct': None,
            'status': 'Monitoring',
            'days_held': 0,
            'last_updated': datetime.now()
        }
        
        positions_df = pd.concat([positions_df, pd.DataFrame([new_position])], ignore_index=True)
        print(f"  ✅ Added {signal['symbol']} to monitoring (Signal date: {signal['signal_date'].date()})")
    
    save_trading_positions(positions_df)
    return positions_df

def process_monitoring_stocks(positions_df):
    """Process stocks that are in monitoring status"""
    print("\n📊 Processing monitoring stocks...")
    
    monitoring_stocks = positions_df[positions_df['status'] == 'Monitoring']
    
    if monitoring_stocks.empty:
        print("No stocks in monitoring status")
        return positions_df
    
    today = datetime.now().date()
    
    for idx, position in monitoring_stocks.iterrows():
        symbol = position['symbol']
        signal_date = position['signal_date'].date()
        
        # Check if we should still monitor this stock (within 3 days of signal)
        days_since_signal = (today - signal_date).days
        if days_since_signal > MAX_DAYS_AFTER_REFERENCE:
            print(f"  ❌ {symbol}: Monitoring period expired")
            positions_df.loc[idx, 'status'] = 'Expired'
            positions_df.loc[idx, 'last_updated'] = datetime.now()
            continue
        
        # Fetch data for analysis
        start_date = signal_date
        end_date = min(signal_date + timedelta(days=MAX_DAYS_AFTER_REFERENCE + 2), today)
        
        bulk_df = fetch_bulk_daily_data(symbol, start_date, end_date)
        
        if bulk_df is None or len(bulk_df) == 0:
            print(f"  ❌ {symbol}: No data available")
            continue
        
        bulk_df = bulk_df.sort_values('date').drop_duplicates('date')
        
        # Look for reference candles (open > 5% from previous close)
        bulk_df = bulk_df.copy()
        bulk_df['prev_close'] = bulk_df['intc'].shift(1)
        bulk_df['open_change_pct'] = ((bulk_df['into'] - bulk_df['prev_close']) / bulk_df['prev_close']) * 100
        
        reference_candles = bulk_df[bulk_df['open_change_pct'] > 5]
        
        if len(reference_candles) == 0:
            print(f"  ❌ {symbol}: No reference candles found yet")
            continue
        
        # Take the first reference candle
        ref_candle = reference_candles.iloc[0]
        ref_date = ref_candle['date']
        ref_index = bulk_df.index.get_loc(reference_candles.index[0])
        
        # Update reference date in positions
        positions_df.loc[idx, 'reference_date'] = ref_date
        
        # Find low volume candle within MAX_DAYS_AFTER_REFERENCE days after reference
        low_volume_candle = None
        low_volume_idx = None
        
        for i in range(ref_index + 1, min(ref_index + 1 + MAX_DAYS_AFTER_REFERENCE, len(bulk_df))):
            current_candle = bulk_df.iloc[i]
            
            if low_volume_candle is None or current_candle['intv'] < low_volume_candle['intv']:
                low_volume_candle = current_candle
                low_volume_idx = i
        
        if low_volume_candle is None:
            print(f"  ❌ {symbol}: No low volume candle found yet")
            continue
        
        # Update low volume date in positions
        positions_df.loc[idx, 'low_volume_date'] = low_volume_candle['date']
        
        # Find breakout candle - must break the high of the low volume candle
        breakout_found = False
        breakout_candle = None
        
        for i in range(low_volume_idx + 1, min(low_volume_idx + 1 + MAX_CANDLES_AFTER_LOW_VOLUME, len(bulk_df))):
            potential_breakout = bulk_df.iloc[i]
            
            if potential_breakout['inth'] > low_volume_candle['inth']:
                breakout_found = True
                breakout_candle = potential_breakout
                break
        
        if not breakout_found:
            print(f"  ❌ {symbol}: No breakout found yet")
            continue
        
        # ENTRY SIGNAL FOUND - Mark for trading next day
        entry_price = breakout_candle['intc']  # Enter at CLOSE of breakout day
        stop_loss = min(ref_candle['intl'], low_volume_candle['intl'])
        risk = entry_price - stop_loss
        target_price = entry_price + (RISK_REWARD_RATIO * risk)
        
        # Update position with entry details
        positions_df.loc[idx, 'entry_date'] = breakout_candle['date']
        positions_df.loc[idx, 'entry_price'] = entry_price
        positions_df.loc[idx, 'stop_loss'] = stop_loss
        positions_df.loc[idx, 'target_price'] = target_price
        positions_df.loc[idx, 'status'] = 'Take trade Next day'
        positions_df.loc[idx, 'last_updated'] = datetime.now()
        
        print(f"  ✅ {symbol}: ENTRY SIGNAL - Trade next day at {entry_price:.2f}")
        print(f"      SL: {stop_loss:.2f}, Target: {target_price:.2f}")
    
    save_trading_positions(positions_df)
    return positions_df

def execute_pending_trades(positions_df):
    """Execute trades marked as 'Take trade Next day'"""
    print("\n💼 Executing pending trades...")
    
    pending_trades = positions_df[positions_df['status'] == 'Take trade Next day']
    
    if pending_trades.empty:
        print("No pending trades to execute")
        return positions_df
    
    today = datetime.now().date()
    
    for idx, position in pending_trades.iterrows():
        symbol = position['symbol']
        entry_date = position['entry_date']
        
        if pd.isna(entry_date) or entry_date.date() != today:
            print(f"  ⏳ {symbol}: Not yet time to execute (Entry date: {entry_date})")
            continue
        
        # Here you would place the actual trade order with your broker API
        # For now, we'll just update the status
        print(f"  ✅ {symbol}: Executing trade at {position['entry_price']:.2f}")
        positions_df.loc[idx, 'status'] = 'In Holding'
        positions_df.loc[idx, 'last_updated'] = datetime.now()
    
    save_trading_positions(positions_df)
    return positions_df

def monitor_held_positions(positions_df):
    """Monitor positions that are in holding and check for exits"""
    print("\n👀 Monitoring held positions...")
    
    held_positions = positions_df[positions_df['status'] == 'In Holding']
    
    if held_positions.empty:
        print("No positions in holding")
        return positions_df
    
    today = datetime.now().date()
    
    for idx, position in held_positions.iterrows():
        symbol = position['symbol']
        entry_date = position['entry_date'].date()
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        target_price = position['target_price']
        
        # Fetch today's data
        today_data = fetch_bulk_daily_data(symbol, today, today)
        
        if today_data is None or len(today_data) == 0:
            print(f"  ❌ {symbol}: No data for today")
            continue
        
        today_candle = today_data.iloc[0]
        
        # Check for exit conditions
        exit_info = None
        
        if today_candle['inth'] >= target_price:
            exit_info = {
                'date': today,
                'type': 'Target',
                'price': target_price,
                'pnl': target_price - entry_price
            }
        elif today_candle['intl'] <= stop_loss:
            exit_info = {
                'date': today,
                'type': 'SL',
                'price': stop_loss,
                'pnl': stop_loss - entry_price
            }
        
        if exit_info:
            pnl_pct = (exit_info['pnl'] / entry_price) * 100
            days_held = (today - entry_date).days
            
            print(f"  🔴 {symbol}: {exit_info['type']} HIT at {exit_info['price']:.2f}")
            print(f"      P&L: {exit_info['pnl']:.2f} points ({pnl_pct:.1f}%), Days: {days_held}")
            
            # Update position with exit details
            positions_df.loc[idx, 'exit_date'] = exit_info['date']
            positions_df.loc[idx, 'exit_price'] = exit_info['price']
            positions_df.loc[idx, 'exit_type'] = exit_info['type']
            positions_df.loc[idx, 'pnl'] = exit_info['pnl']
            positions_df.loc[idx, 'pnl_pct'] = pnl_pct
            positions_df.loc[idx, 'days_held'] = days_held
            positions_df.loc[idx, 'status'] = 'Closed'
            positions_df.loc[idx, 'last_updated'] = datetime.now()
        
        else:
            # Update days held
            days_held = (today - entry_date).days
            positions_df.loc[idx, 'days_held'] = days_held
            positions_df.loc[idx, 'last_updated'] = datetime.now()
            
            current_pnl = today_candle['intc'] - entry_price
            current_pnl_pct = (current_pnl / entry_price) * 100
            
            print(f"  📊 {symbol}: Still holding - P&L: {current_pnl:.2f} ({current_pnl_pct:.1f}%), Days: {days_held}")
    
    save_trading_positions(positions_df)
    return positions_df

def run_daily_trading_system():
    """Main function to run the daily trading system"""
    print("=" * 80)
    print("📈 DAILY TRADING SYSTEM STARTED")
    print("=" * 80)
    
    # Initialize trades file if needed
    initialize_trades_file()
    
    # Load current positions
    positions_df = read_trading_positions()
    
    # Step 1: Check for new signals
    positions_df = check_for_new_signals()
    
    # Step 2: Process monitoring stocks
    positions_df = process_monitoring_stocks(positions_df)
    
    # Step 3: Execute pending trades
    positions_df = execute_pending_trades(positions_df)
    
    # Step 4: Monitor held positions
    positions_df = monitor_held_positions(positions_df)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 DAILY SUMMARY")
    print("=" * 80)
    
    status_counts = positions_df['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    closed_positions = positions_df[positions_df['status'] == 'Closed']
    if not closed_positions.empty:
        total_pnl = closed_positions['pnl'].sum()
        avg_pnl = closed_positions['pnl'].mean()
        winning_trades = len(closed_positions[closed_positions['pnl'] > 0])
        win_rate = (winning_trades / len(closed_positions)) * 100
        
        print(f"\n  📈 Closed Trades Performance:")
        print(f"     Total P&L: {total_pnl:.2f} points")
        print(f"     Average P&L: {avg_pnl:.2f} points")
        print(f"     Win Rate: {win_rate:.1f}% ({winning_trades}/{len(closed_positions)})")
    
    print("\n✅ Daily trading system completed")

# Run the daily trading system
if __name__ == "__main__":
    run_daily_trading_system()