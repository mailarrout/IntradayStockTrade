import os
import pandas as pd
import pytz
import json
import time
from datetime import datetime, timedelta
from client_loader import load_clients

# ================= CONFIGURATION =================
RISK_REWARD_RATIO = 1  # Change this to 1 for 1:1, 2 for 1:2, 3 for 1:3, etc.
MAX_DAYS_AFTER_REFERENCE = 3  # Only look for low volume candles within 3 days after reference
MAX_CANDLES_AFTER_LOW_VOLUME = 5  # Maximum candles allowed after low volume for valid entry
# ================================================

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

def process_single_symbol_fast(symbol, signal_date, sector, marketcap, idx, total):
    """Fast processing with bulk data fetching"""
    print(f"\n{'='*80}")
    print(f"📊 PROCESSING {idx}/{total}: {symbol}")
    print(f"   Signal Date: {signal_date}")
    print(f"{'='*80}")
    
    # Calculate date range (signal date to 20 days after)
    start_date = signal_date
    end_date = min(signal_date + timedelta(days=20), datetime.now().date())
    
    # Fetch all data in one go
    bulk_df = fetch_bulk_daily_data(symbol, start_date, end_date)
    
    if bulk_df is None or len(bulk_df) == 0:
        print("   ❌ No data available")
        return None
    
    # Sort and clean the data
    bulk_df = bulk_df.sort_values('date').drop_duplicates('date')
    
    print(f"\n📊 COMPLETE DATA FOR {symbol} ({len(bulk_df)} days):")
    print("   Date        | Open     | High     | Low      | Close    | Volume")
    print("   " + "-" * 60)
    for idx, row in bulk_df.iterrows():
        print(f"   {row['date']} | {row['into']:7.2f} | {row['inth']:7.2f} | {row['intl']:7.2f} | {row['intc']:7.2f} | {row['intv']:9,.0f}")
    
    # Now apply strategy logic to the bulk data
    print(f"\n🔍 APPLYING STRATEGY LOGIC...")
    
    # Add previous close and percentage change
    bulk_df = bulk_df.copy()
    bulk_df['prev_close'] = bulk_df['intc'].shift(1)
    bulk_df['open_change_pct'] = ((bulk_df['into'] - bulk_df['prev_close']) / bulk_df['prev_close']) * 100
    
    # Look for reference candles (open > 5% from previous close)
    reference_candles = bulk_df[bulk_df['open_change_pct'] > 5]
    
    if len(reference_candles) == 0:
        print("   ❌ No reference candles found (open > 5% from previous close)")
        return None
    
    print(f"   ✅ Found {len(reference_candles)} reference candle(s)")
    
    # Process each reference candle
    for ref_idx, ref_candle in reference_candles.iterrows():
        ref_date = ref_candle['date']
        ref_index = bulk_df.index.get_loc(ref_idx)
        
        # Find low volume candle within MAX_DAYS_AFTER_REFERENCE days after reference
        days_after_ref = 0
        low_volume_candle = None
        low_volume_idx = None
        
        # Look for the lowest volume candle in the next MAX_DAYS_AFTER_REFERENCE days
        for i in range(ref_index + 1, min(ref_index + 1 + MAX_DAYS_AFTER_REFERENCE, len(bulk_df))):
            current_candle = bulk_df.iloc[i]
            days_after_ref += 1
            
            if low_volume_candle is None or current_candle['intv'] < low_volume_candle['intv']:
                low_volume_candle = current_candle
                low_volume_idx = i
        
        if low_volume_candle is None:
            print(f"   ❌ No low volume candle found within {MAX_DAYS_AFTER_REFERENCE} days after reference")
            continue
        
        print(f"   ✅ Found low volume candle on {low_volume_candle['date']} (Volume: {low_volume_candle['intv']:,.0f})")
        
        # Find breakout candle - must break the high of the low volume candle
        breakout_found = False
        breakout_candle = None
        
        # Check candles after the low volume candle for breakout
        for i in range(low_volume_idx + 1, min(low_volume_idx + 1 + MAX_CANDLES_AFTER_LOW_VOLUME, len(bulk_df))):
            potential_breakout = bulk_df.iloc[i]
            
            if potential_breakout['inth'] > low_volume_candle['inth']:
                breakout_found = True
                breakout_candle = potential_breakout
                break
        
        if not breakout_found:
            print(f"   ❌ No breakout within {MAX_CANDLES_AFTER_LOW_VOLUME} candles after low volume")
            continue
            
        # ENTRY SIGNAL - Enter on the CLOSE of the breakout candle
        entry_price = breakout_candle['intc']  # Enter at CLOSE of breakout day
        stop_loss = min(ref_candle['intl'], low_volume_candle['intl'])
        risk = entry_price - stop_loss
        
        if risk <= 0:
            print(f"   ❌ Invalid risk calculation (entry: {entry_price}, SL: {stop_loss})")
            continue
            
        target_price = entry_price + (RISK_REWARD_RATIO * risk)
        
        print(f"   ✅ TRADE SIGNAL FOUND!")
        print(f"      Reference: {ref_date}")
        print(f"      Low Volume: {low_volume_candle['date']} (High: {low_volume_candle['inth']:.2f})")
        print(f"      Entry: {breakout_candle['date']} at CLOSE {entry_price:.2f} (Breakout of {low_volume_candle['inth']:.2f})")
        print(f"      Stop Loss: {stop_loss:.2f}")
        print(f"      Target: {target_price:.2f}")
        print(f"      Risk: {risk:.2f} points")
        print(f"      Reward: {risk * RISK_REWARD_RATIO:.2f} points")
        print(f"      Risk-Reward: 1:{RISK_REWARD_RATIO}")
        
        # Check exit conditions (start from the day AFTER entry)
        data_after_entry = bulk_df[bulk_df['date'] > breakout_candle['date']]
        
        exit_info = None
        for exit_idx, exit_row in data_after_entry.iterrows():
            if exit_row['inth'] >= target_price:
                exit_info = {
                    'date': exit_row['date'],
                    'type': 'Target',
                    'price': target_price,
                    'pnl': target_price - entry_price
                }
                break
            
            if exit_row['intl'] <= stop_loss:
                exit_info = {
                    'date': exit_row['date'],
                    'type': 'SL',
                    'price': stop_loss,
                    'pnl': stop_loss - entry_price
                }
                break
        
        if exit_info:
            pnl_pct = (exit_info['pnl'] / entry_price) * 100
            days_held = (exit_info['date'] - breakout_candle['date']).days
            
            print(f"      🔴 {exit_info['type']} HIT on {exit_info['date']} at {exit_info['price']:.2f}")
            print(f"      P&L: {exit_info['pnl']:.2f} points ({pnl_pct:.1f}%), Days: {days_held}")
            
            return {
                'symbol': symbol,
                'sector': sector,
                'marketcap': marketcap,
                'signal_date': signal_date,
                'entry_date': breakout_candle['date'],
                'entry_price': entry_price,
                'exit_date': exit_info['date'],
                'exit_price': exit_info['price'],
                'exit_type': exit_info['type'],
                'pnl': exit_info['pnl'],
                'pnl_pct': pnl_pct,
                'days_held': days_held
            }
        else:
            # No exit triggered
            if len(data_after_entry) > 0:
                last_row = data_after_entry.iloc[-1]
                days_held = (last_row['date'] - breakout_candle['date']).days
                unrealized_pnl = last_row['intc'] - entry_price
                unrealized_pnl_pct = (unrealized_pnl / entry_price) * 100
                
                print(f"      🟡 NO EXIT - Last price: {last_row['intc']:.2f} on {last_row['date']}")
                print(f"      Unrealized P&L: {unrealized_pnl:.2f} points ({unrealized_pnl_pct:.1f}%), Days: {days_held}")
                
                return {
                    'symbol': symbol,
                    'sector': sector,
                    'marketcap': marketcap,
                    'signal_date': signal_date,
                    'entry_date': breakout_candle['date'],
                    'entry_price': entry_price,
                    'exit_date': last_row['date'],
                    'exit_price': last_row['intc'],
                    'exit_type': 'No Exit',
                    'pnl': unrealized_pnl,
                    'pnl_pct': unrealized_pnl_pct,
                    'days_held': days_held
                }
    
    print("   ❌ No valid trade setup found")
    return None

def run_fast_test():
    """Run fast test with bulk data fetching"""
    print(f"=== FAST BACKTEST WITH BULK DATA FETCHING ===")
    print(f"Using Risk-Reward Ratio: 1:{RISK_REWARD_RATIO}")
    print(f"Looking for low volume within {MAX_DAYS_AFTER_REFERENCE} days after reference")
    print(f"Maximum candles after low volume for valid entry: {MAX_CANDLES_AFTER_LOW_VOLUME}")
    
    signals_df = read_backtest_signals()
    
    if signals_df.empty:
        print("No signals found")
        return
    
    # Test with first 5 symbols for quick validation
    test_symbols = signals_df.head(5)
    results = []
    
    for idx, signal in test_symbols.iterrows():
        symbol = signal['symbol']
        signal_date = signal['signal_date'].date()
        sector = signal.get('sector', 'N/A')
        marketcap = signal.get('marketcapname', 'N/A')
        
        result = process_single_symbol_fast(symbol, signal_date, sector, marketcap, idx+1, len(test_symbols))
        
        if result:
            results.append(result)
            print(f"🎯 TRADE FOUND FOR {symbol}")
        
        print(f"\n{'='*80}")
        time.sleep(1)  # Small delay between symbols
    
    # Print summary
    if results:
        print(f"\n📈 BACKTEST SUMMARY:")
        print(f"   Found {len(results)} trades out of {len(test_symbols)} symbols")
        
        total_pnl = sum(r['pnl'] for r in results)
        avg_pnl = total_pnl / len(results)
        winning_trades = sum(1 for r in results if r['pnl'] > 0)
        win_rate = (winning_trades / len(results)) * 100
        
        print(f"   Total P&L: {total_pnl:.2f} points")
        print(f"   Average P&L: {avg_pnl:.2f} points")
        print(f"   Win Rate: {win_rate:.1f}% ({winning_trades}/{len(results)})")
    
    print(f"✅ Completed fast test.")

# Run the fast test
if __name__ == "__main__":
    run_fast_test()