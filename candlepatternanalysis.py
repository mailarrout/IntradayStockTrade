import pandas as pd
import os
import glob
from datetime import datetime

def analyze_stock_data(file_path, log_file):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        log_file.write(f"File not found: {file_path}\n")
        return False
    except Exception as e:
        log_file.write(f"Error reading {file_path}: {e}\n")
        return False
    
    # Get the stock name from filename
    stock_name = os.path.basename(file_path).replace('.csv', '')
    
    # Rename columns as specified
    plot_df = df.rename(columns={
        "time": "Date",
        "into": "Open",
        "inth": "High",
        "intl": "Low",
        "intc": "Close",
        "intv": "Volume"
    })
    
    # Convert Date column to datetime
    plot_df['Date'] = pd.to_datetime(plot_df['Date'], format='%d-%m-%Y %H:%M:%S')
    
    # Sort by date (oldest first)
    plot_df = plot_df.sort_values('Date')
    
    # Reset index for easier navigation
    plot_df.reset_index(drop=True, inplace=True)
    
    # Find the 9:15 AM candle
    nine_fifteen_idx = None
    nine_fifteen_candle = None
    
    for idx, row in plot_df.iterrows():
        if row['Date'].hour == 9 and row['Date'].minute == 15:
            nine_fifteen_idx = idx
            nine_fifteen_candle = row
            break
    
    if nine_fifteen_candle is None:
        log_file.write(f"{stock_name}: 9:15 AM candle not found\n")
        return False
    
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"ANALYZING: {stock_name}\n")
    log_file.write(f"{'='*80}\n")
    log_file.write(f"9:15 AM candle - Time: {nine_fifteen_candle['Date']}, Low: {nine_fifteen_candle['Low']}, High: {nine_fifteen_candle['High']}\n")
    
    # Initialize variables for sequential processing
    high_volume_count = 0
    reference_candle = None
    previous_reference = None
    trade_triggered = False
    day_low_broken = False
    stock_ignored = False
    
    # Process candles sequentially after 9:15 AM
    for idx in range(nine_fifteen_idx + 1, len(plot_df)):
        current_candle = plot_df.iloc[idx]
        previous_candle = plot_df.iloc[idx - 1]
        
        # CRITICAL: Check if ANY candle closes above 9:15 high - if yes, ignore stock immediately
        if current_candle['Close'] > nine_fifteen_candle['High']:
            log_file.write(f"❌ IGNORED: Candle closed above 9:15 high at: {current_candle['Date']}\n")
            log_file.write(f"   Close: {current_candle['Close']}, 9:15 High: {nine_fifteen_candle['High']}\n")
            stock_ignored = True
            break
        
        # Check if current candle closed below 9:15 low (FIRST CRITERIA)
        if not day_low_broken and current_candle['Close'] < nine_fifteen_candle['Low']:
            day_low_broken = True
            log_file.write(f"✓ 9:15 Low broken at: {current_candle['Date']}, Close: {current_candle['Close']}\n")
        
        # Only proceed with pattern analysis if 9:15 low was broken AND stock not ignored
        if not day_low_broken or stock_ignored:
            continue
        
        # Check if candle has higher volume than previous candle
        is_high_volume = (current_candle['Volume'] > previous_candle['Volume'])
        
        if is_high_volume:
            high_volume_count += 1
            
            if high_volume_count == 2:
                previous_reference = reference_candle
                reference_candle = {
                    'time': current_candle['Date'],
                    'low': current_candle['Low'],
                    'high': current_candle['High'],
                    'volume': current_candle['Volume'],
                    'index': idx
                }
                log_file.write(f"Second high volume candle at: {reference_candle['time']}, Low: {reference_candle['low']}\n")
        
        # If we have a reference candle, check for break of reference low
        if reference_candle is not None and not trade_triggered:
            # Check if current candle is within next 5 candles of reference
            within_reference_range = (idx <= reference_candle['index'] + 5)
            
            # Check if any candle (high volume or not) breaks the current reference low
            if current_candle['Close'] < reference_candle['low']:
                log_file.write(f"🚨 TRADE SIGNAL! Candle broke reference low at: {current_candle['Date']}\n")
                log_file.write(f"   Reference Low: {reference_candle['low']}, Current Close: {current_candle['Close']}\n")
                log_file.write(f"   ENTRY at candle close: {current_candle['Date']}\n")
                trade_triggered = True
                break
            
            # Check for additional high volume candles in next 3-5 candles for reference update
            if within_reference_range and is_high_volume:
                # Update reference to this new high volume candle
                previous_reference = reference_candle
                reference_candle = {
                    'time': current_candle['Date'],
                    'low': current_candle['Low'],
                    'high': current_candle['High'],
                    'volume': current_candle['Volume'],
                    'index': idx
                }
                log_file.write(f"Reference updated to: {reference_candle['time']}, Low: {reference_candle['low']}\n")
    
    if stock_ignored:
        log_file.write(f"{stock_name}: IGNORED - Candle closed above 9:15 high\n")
        return False
    
    if not day_low_broken:
        log_file.write(f"{stock_name}: No candle closed below 9:15 low during market hours\n")
        return False
    
    if not trade_triggered:
        log_file.write(f"No trade signal generated for {stock_name}\n")
    
    return True

def main():
    # Create app directory if it doesn't exist
    app_dir = 'app'
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(app_dir, f'analysis_log_{timestamp}.txt')
    
    # Find all CSV files in HistoricalData directory and subdirectories
    historical_data_dir = 'HistoricalData'
    csv_files = []
    
    if os.path.exists(historical_data_dir):
        # Get all CSV files in all subdirectories
        csv_files = glob.glob(os.path.join(historical_data_dir, '**', '*.csv'), recursive=True)
    else:
        print(f"HistoricalData directory not found: {historical_data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to analyze")
    print(f"Logging results to: {log_file_path}")
    
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Stock Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Found {len(csv_files)} CSV files to analyze\n")
        log_file.write("=" * 80 + "\n")
        
        # List all files found
        log_file.write("FILES FOUND:\n")
        for i, file_path in enumerate(csv_files, 1):
            log_file.write(f"{i}. {file_path}\n")
        log_file.write("=" * 80 + "\n")
        
        # Analyze each file
        analyzed_count = 0
        for i, file_path in enumerate(csv_files, 1):
            print(f"Analyzing file {i}/{len(csv_files)}: {os.path.basename(file_path)}")
            if analyze_stock_data(file_path, log_file):
                analyzed_count += 1
        
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"ANALYSIS COMPLETED - {analyzed_count}/{len(csv_files)} stocks met all criteria\n")
        log_file.write("=" * 80 + "\n")
    
    print(f"Analysis complete. {analyzed_count}/{len(csv_files)} stocks met all criteria.")
    print(f"Results logged to: {log_file_path}")

if __name__ == "__main__":
    main()