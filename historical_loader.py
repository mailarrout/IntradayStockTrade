import os
from datetime import datetime, date, time, timedelta
import pytz
import pandas as pd
import logging
from PyQt5.QtCore import QTimer
#from FiveMinInvertedHammer import run_strategy
# from FiveMinVolumeReference import run_strategy
from LowVolumeCandleBreakOut import run_strategy


last_strategy_run = None
STRATEGY_COOLDOWN_MINUTES = 1  # 2 minutes cooldown between strategy runs

# Get logger for this module
logger = logging.getLogger(__name__)

continuous_timer = None

def load_token_map(symbol_file="NSE_symbols.txt"):
    """Load NSE symbol-token mapping from file"""
    try:
        if not os.path.exists(symbol_file):
            raise FileNotFoundError(f"Symbol file not found: {symbol_file}")

        logger.debug(f"Loading symbol mapping from: {symbol_file}")
        df = pd.read_csv(symbol_file)
        token_map = {row["Symbol"].strip().upper(): str(row["Token"]) for _, row in df.iterrows()}
        logger.info(f"Loaded {len(token_map)} symbols from {symbol_file}")
        return token_map
        
    except FileNotFoundError as e:
        logger.error(f"Symbol file not found: {symbol_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading symbol mapping: {str(e)}")
        raise

def start_continuous_fetching(ui_reference, symbols):
    """Start the 5-minute interval timer for exact candle timing"""
    global continuous_timer
    
    logger.info("Starting continuous fetching")
    stop_continuous_fetching()
    
    continuous_timer = QTimer()
    
    def fetch_immediately():
        logger.debug("Continuous fetch timer triggered")
        fetch_historical_for_symbols(ui_reference, symbols)
    
    # Calculate time until next exact 5-minute interval (XX:00, XX:05, etc.)
    now = datetime.now()
    # Get the next 5-minute boundary
    minutes_to_next = 5 - (now.minute % 5)
    next_time = now.replace(minute=now.minute + minutes_to_next, second=0, microsecond=0)
    
    # If we're exactly at a boundary, wait for the next one
    if next_time <= now:
        next_time = next_time + timedelta(minutes=5)
    
    delay_ms = (next_time - now).total_seconds() * 1000
    
    continuous_timer.timeout.connect(fetch_immediately)
    continuous_timer.start(300000)  # 5 minutes (300 seconds)
    
    # Trigger first fetch at exact 5-minute boundary
    QTimer.singleShot(int(delay_ms), fetch_immediately)
    logger.info(f"First fetch scheduled at: {next_time.strftime('%H:%M:%S')}")

def stop_continuous_fetching():
    """Stop the continuous timer"""
    global continuous_timer
    if continuous_timer and continuous_timer.isActive():
        continuous_timer.stop()
        logger.info("Continuous fetching stopped")
    else:
        logger.debug("No active continuous timer to stop")

def fetch_historical_for_symbols(ui_reference, symbols, target_date=None, interval_minutes=5):
    """Fetch historical OHLC for given stock symbols using existing clients"""
    logger.info("Starting historical data fetch for %d symbols: %s", len(symbols), symbols)
    
    if not hasattr(ui_reference, "clients") or not ui_reference.clients:
        logger.error("No logged-in clients available. Please load clients first.")
        return False

    # Pick the first client from client_loader
    client_name, client_id, client = ui_reference.clients[0]
    logger.info(f"Using client {client_name} ({client_id}) for historical fetch")

    # Default to today's date if not provided
    if target_date is None:
        target_date = date.today()

    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)
    
    # Market hours
    market_open = ist.localize(datetime.combine(target_date, time(9, 15)))
    market_close = ist.localize(datetime.combine(target_date, time(15, 30)))
    
    logger.debug(f"Market hours: {market_open.strftime('%H:%M')} to {market_close.strftime('%H:%M')}")
    logger.debug(f"Current IST time: {now_ist.strftime('%H:%M:%S')}")
    
    # Set start time to market open or current time if after market close
    if now_ist > market_close:
        starttime_ist = market_open
        endtime_ist = market_close
        logger.info("Market closed - fetching full day data")
    elif now_ist < market_open:
        # Before market opens, get previous day's data or nothing
        logger.info("Market not open yet")
        return False
    else:
        # During market hours, get data from market open to current time
        starttime_ist = market_open
        endtime_ist = now_ist
        logger.info("Market open - fetching data from market open to current time")
    
    # Convert to UTC timestamps for the API
    starttime = int(starttime_ist.astimezone(pytz.UTC).timestamp())
    endtime = int(endtime_ist.astimezone(pytz.UTC).timestamp())

    logger.info(f"Fetching OHLC data for {len(symbols)} symbols from {starttime_ist.strftime('%H:%M')} to {endtime_ist.strftime('%H:%M')}")

    # Output folder with date-based subfolder
    output_dir = os.path.join(os.getcwd(), "HistoricalData", target_date.strftime('%Y-%m-%d'))
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Output directory: {output_dir}")

    # Load token mapping
    try:
        token_map = load_token_map()
        logger.debug("Successfully loaded token mapping for %d symbols", len(token_map))
    except Exception as e:
        logger.error(f"Failed to load token mapping: {str(e)}")
        return False

    success_count = 0
    failed_count = 0
    failed_symbols = []
    
    for symbol in symbols:
        sym = symbol.strip().upper()
        if sym not in token_map:
            logger.warning(f"Token not found for {sym}. Skipping.")
            failed_count += 1
            failed_symbols.append(sym)
            continue

        token = token_map[sym]
        logger.info(f"Fetching {sym} (token={token}) from {starttime_ist.strftime('%H:%M')} to {endtime_ist.strftime('%H:%M')}")

        try:
            logger.debug(f"Requesting time price series for {sym}")
            data = client.get_time_price_series(
                exchange="NSE",
                token=token,
                starttime=starttime,
                endtime=endtime,
                interval=5
            )
            
            if data and isinstance(data, list):
                df = pd.DataFrame(data)
                if not df.empty:
                    # DON'T convert timestamp - keep the original format "dd-mm-yyyy HH:MM:SS"
                    # The API returns time in the format we need: "02-06-2020 15:46:23"
                    
                    # Save to file
                    filename = f"{sym}.csv"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Always overwrite with complete data for the day
                    df.to_csv(filepath, index=False)
                    
                    success_count += 1
                    logger.info(f"Saved {len(df)} candles for {sym} to {filepath}")
                else:
                    logger.warning(f"No data for {sym} in this time range")
                    failed_count += 1
                    failed_symbols.append(sym)
            else:
                logger.warning(f"No data returned for {sym}")
                failed_count += 1
                failed_symbols.append(sym)
                
        except Exception as e:
            logger.error(f"Failed fetching {sym}: {str(e)}")
            failed_count += 1
            failed_symbols.append(sym)

    # ✅ Moved outside the loop - process all symbols first
    if success_count > 0:
        logger.info(f"Completed historical fetch. Success: {success_count}, Failed: {failed_count}")
        
        current_time_ist = datetime.now(ist)
        current_time_only = current_time_ist.time()
        market_open_time = time(9, 15)
        market_close_time = time(15, 30)
        
        # Check if we're in market hours AND cooldown has passed
        global last_strategy_run
        time_since_last = None
        if last_strategy_run:
            time_since_last = (current_time_ist - last_strategy_run).total_seconds() / 60
        
        # Allow strategy run if:
        # 1. We're in market hours (9:15-15:30)
        # 2. Either no previous run OR cooldown period has passed
        # 3. We're at a 5-minute boundary (XX:00, XX:05, etc.)
        current_minute = current_time_ist.minute
        at_5_min_boundary = (current_minute % 5 == 0)
        
        can_run_strategy = (
            market_open_time <= current_time_only <= market_close_time and
            at_5_min_boundary and
            (last_strategy_run is None or (time_since_last and time_since_last >= STRATEGY_COOLDOWN_MINUTES))
        )
        
        if can_run_strategy:
            logger.info(f"Strategy run scheduled (at 5-min boundary: {current_time_ist.strftime('%H:%M')})")
            last_strategy_run = current_time_ist
            # Run strategy immediately
            run_strategy(ui_reference)
        else:
            if not at_5_min_boundary:
                logger.info(f"Skipping strategy run (not at 5-min boundary: {current_time_ist.strftime('%H:%M')})")
            elif last_strategy_run and time_since_last < STRATEGY_COOLDOWN_MINUTES:
                logger.info(f"Skipping strategy run (cooldown: {time_since_last:.1f}m/{STRATEGY_COOLDOWN_MINUTES}m remaining)")
            else:
                logger.info("Skipping strategy run (outside market hours)")
        
        return True
    
    else:
        logger.warning("No historical data fetched for any symbols")
        return False