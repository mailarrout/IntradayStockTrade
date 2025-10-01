# chartink_trading_bot.py
import os, csv, time, threading, queue, pandas as pd, pyotp, pytz
from datetime import datetime, date, timedelta
from flask import Flask, request, jsonify
from win10toast import ToastNotifier
from api_helper import ShoonyaApiPy
import logging

# ================== TRADING CONSTANTS ==================
REWARD_RATIO = 3  # Target = Buy Price + (Candle Range * REWARD_RATIO)
SL_BUFFER = 0.05  # SL = Previous Candle Low - SL_BUFFER
FALLBACK_SL_PERCENT = 0.5  # 0.5% fallback SL if no candle data
FALLBACK_TARGET_PERCENT = 1.0  # 1.0% fallback target if no candle data
ORDER_QUANTITY = 10  # Default quantity for orders

# ================== CONFIGURATION ==================
app = Flask(__name__)
toaster = ToastNotifier()

# Global variables
broker_client = None
client_loaded = False
alert_queue = queue.Queue()
position_monitor_running = False
shutdown_flag = False

# Files
today_date = datetime.now().strftime("%Y-%m-%d")
alerts_file = f"app/{today_date}_alerts.csv"
positions_file = f"app/{today_date}_positions_state.csv"
position_book_file = f"app/{today_date}_position.csv"
candle_analysis_file = f"app/{today_date}_candle_analysis.csv"
log_file = f"logs/{today_date}_trading_bot.log"

# Ensure directories exist
os.makedirs("app", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ================== LOGGING SETUP ==================
# Force log timestamps to IST
ist = pytz.timezone("Asia/Kolkata")

def ist_time_converter(*args):
    return datetime.now(ist).timetuple()

logging.Formatter.converter = ist_time_converter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")


# ================== TIME UTILITIES ==================
def get_ist_time():
    """Get current time in IST"""
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist)

def get_ist_timestamp():
    """Get current timestamp in IST for logging"""
    return get_ist_time().strftime("%Y-%m-%d %H:%M:%S")

def is_market_closing_time():
    """Check if it's between 3:15 PM and 3:20 PM IST"""
    ist_time = get_ist_time()
    logger.debug(f"Checking market closing time - Current IST: {ist_time.strftime('%H:%M:%S')}")
    # 5-minute window from 3:15 to 3:20 PM
    result = (ist_time.hour == 15 and ist_time.minute >= 15 and ist_time.minute <= 17)
    if result:
        logger.info(f"Market closing time detected: {ist_time.strftime('%H:%M:%S')}")
    return result

def initialize_positions_file():
    """Create today's positions file with headers if it doesn't exist"""
    if not os.path.exists(positions_file):
        with open(positions_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Symbol", "BuyPrice", "EntryTime",
                "StopLoss", "Target", "Status",
                "ExitPrice", "ExitTime", "ExitOrderId"
            ])
        logger.info(f"Created new positions file: {positions_file}")


# ================== SYMBOL TOKEN MAPPING ==================
def load_symbol_token_mapping():
    """Load symbol to token mapping from NSE_symbols.txt file"""
    logger.info("Starting symbol token mapping load")
    symbol_to_token = {}
    try:
        if os.path.exists("NSE_symbols.txt"):
            logger.info("NSE_symbols.txt file found, loading data")
            df = pd.read_csv("NSE_symbols.txt")
            logger.info(f"Loaded {len(df)} rows from NSE_symbols.txt")
            
            for _, row in df.iterrows():
                symbol = row['Symbol'].strip().upper()
                token = str(row['Token']).strip()
                trading_symbol = row['TradingSymbol'].strip().upper()
                
                symbol_to_token[symbol] = token
                symbol_to_token[trading_symbol] = token
                
                logger.debug(f"Mapped symbol: {symbol}, trading_symbol: {trading_symbol} -> token: {token}")
            
            logger.info(f"Successfully loaded {len(symbol_to_token)} symbol mappings from NSE_symbols.txt")
        else:
            logger.warning("NSE_symbols.txt file not found")
            
    except Exception as e:
        logger.error(f"Error loading symbol mapping: {e}")
    
    return symbol_to_token

SYMBOL_TO_TOKEN = load_symbol_token_mapping()

def get_symbol_token(symbol):
    """Get token for symbol from NSE_symbols.txt mapping"""
    try:
        clean_symbol = symbol.replace('-EQ', '').strip().upper()
        logger.debug(f"Looking up token for symbol: {symbol} -> clean: {clean_symbol}")
        
        token = SYMBOL_TO_TOKEN.get(clean_symbol)
        
        if token:
            logger.info(f"Found token for {clean_symbol}: {token}")
            return token
        else:
            logger.warning(f"Token not found in mapping for: {clean_symbol}")
            logger.debug(f"Available symbols in mapping: {list(SYMBOL_TO_TOKEN.keys())[:10]}...")  # First 10 only
            return None
            
    except Exception as e:
        logger.error(f"Error getting token for {symbol}: {e}")
        return None

# ================== PREVIOUS CANDLE ANALYSIS ==================
def get_previous_candle_hloc(symbol, reference_time=None):
    try:
        logger.info(f"Fetching previous candle for: {symbol}")

        clean_symbol = symbol.replace('-EQ', '').strip()
        token = get_symbol_token(clean_symbol)
        if not token:
            logger.warning(f"No token found for {symbol}, cannot fetch candle data")
            return None

        ist = pytz.timezone("Asia/Kolkata")
        if reference_time is None:
            reference_time = get_ist_time()
            logger.debug(f"No reference time provided, using current time: {reference_time}")

        # Align to candle minute
        current_minute = reference_time.replace(second=0, microsecond=0)
        previous_candle_end = current_minute - timedelta(minutes=1)
        previous_candle_start = previous_candle_end - timedelta(minutes=1)

        logger.info(f"Candle time range: {previous_candle_start.strftime('%H:%M:%S')} to {previous_candle_end.strftime('%H:%M:%S')}")

        # Convert IST → UTC → Epoch
        starttime = int(previous_candle_start.astimezone(pytz.UTC).timestamp())
        endtime = int(previous_candle_end.astimezone(pytz.UTC).timestamp())

        logger.info(f"Candle time range (UTC epoch): {starttime} to {endtime}")

        logger.info(f"Calling broker API for time price series: token={token}, interval=1")
        data = broker_client.get_time_price_series(
            exchange="NSE",
            token=token,
            starttime=starttime,
            endtime=endtime,
            interval=1
        )

        logger.debug(f"Raw candle data received: {data}")

        if data and isinstance(data, list) and len(data) > 0:
            candle = data[0]
            hloc = {
                'high': float(candle.get('inth', 0)),
                'low': float(candle.get('intl', 0)),
                'open': float(candle.get('into', 0)),
                'close': float(candle.get('intc', 0)),
                'timestamp': previous_candle_end.strftime('%H:%M:%S'),
                'date_time': previous_candle_end.strftime('%Y-%m-%d %H:%M:%S')
            }
            if hloc['high'] > 0 and hloc['low'] > 0:
                logger.info(f"Candle found: {symbol} - High: {hloc['high']}, Low: {hloc['low']}, Open: {hloc['open']}, Close: {hloc['close']}")
                return hloc
            else:
                logger.warning(f"Invalid candle data for {symbol}: {hloc}")
                return None
        else:
            logger.warning(f"No previous candle data found for {symbol}")
            return None

    except Exception as e:
        logger.error(f"Error fetching previous candle for {symbol}: {e}")
        return None

def calculate_sl_target_from_previous_candle(buy_price, previous_candle, symbol):
    """Calculate SL and Target based on previous candle's low with buffer and reward ratio"""
    logger.info(f"Starting SL/Target calculation for {symbol}, Buy Price: {buy_price}")
    try:
        if not previous_candle:
            logger.warning(f"No previous candle for {symbol}, using fallback SL/Target")
            stop_loss = round(buy_price * (1 - FALLBACK_SL_PERCENT/100), 2)
            target = round(buy_price * (1 + FALLBACK_TARGET_PERCENT/100), 2)
            logger.info(f"Fallback SL/Target for {symbol}: SL={stop_loss}, Target={target}")
        else:
            prev_low = previous_candle['low']
            prev_high = previous_candle['high']
            
            logger.info(f"Previous candle data - High: {prev_high}, Low: {prev_low}")
            
            stop_loss = round(prev_low - SL_BUFFER, 2)
            candle_range = prev_high - prev_low
            target = round(buy_price + (candle_range * REWARD_RATIO), 2)
            
            logger.info(f"SL/Target calculation for {symbol}:")
            logger.info(f"Buy Price: {buy_price}, Prev Low: {prev_low}, Prev High: {prev_high}")
            logger.info(f"Candle Range: {candle_range:.2f}, SL: {stop_loss}, Target: {target}")
            logger.info(f"Using REWARD_RATIO: {REWARD_RATIO}, SL_BUFFER: {SL_BUFFER}")
            
            if stop_loss >= buy_price:
                stop_loss = round(buy_price * (1 - FALLBACK_SL_PERCENT/100), 2)
                logger.warning(f"Adjusted SL above buy price for {symbol}. New SL: {stop_loss}")
            
            if target <= buy_price:
                target = round(buy_price * (1 + FALLBACK_TARGET_PERCENT/100), 2)
                logger.warning(f"Adjusted target below buy price for {symbol}. New Target: {target}")
        
        logger.info(f"Final levels for {symbol}: Buy={buy_price}, SL={stop_loss}, Target={target}")
        return stop_loss, target
        
    except Exception as e:
        logger.error(f"Error calculating SL/Target for {symbol}: {e}")
        fallback_sl = round(buy_price * (1 - FALLBACK_SL_PERCENT/100), 2)
        fallback_target = round(buy_price * (1 + FALLBACK_TARGET_PERCENT/100), 2)
        logger.info(f"Using emergency fallback: SL={fallback_sl}, Target={fallback_target}")
        return fallback_sl, fallback_target

def log_candle_analysis_to_csv(symbol, buy_price, previous_candle, sl, target):
    """Log candle analysis details to CSV for backtesting and analysis"""
    try:
        file_exists = os.path.exists(candle_analysis_file)
        logger.debug(f"Logging candle analysis to CSV. File exists: {file_exists}")
        
        with open(candle_analysis_file, "a", newline="") as f:
            writer = csv.writer(f)
            
            if not file_exists:
                logger.info(f"Creating new candle analysis file: {candle_analysis_file}")
                writer.writerow([
                    "Timestamp", "Symbol", "Candle_Time", "Open", "High", "Low", "Close", 
                    "Buy_Price", "Stop_Loss", "Target", "Candle_Range", "SL_Type", "Calculation_Time"
                ])
            
            if previous_candle:
                candle_range = previous_candle['high'] - previous_candle['low']
                sl_type = "PREV_CANDLE_LOW"
                logger.debug(f"Using previous candle data for {symbol}, range: {candle_range}")
            else:
                candle_range = 0
                sl_type = "FALLBACK"
                logger.debug(f"No previous candle for {symbol}, using fallback")
            
            row_data = [
                get_ist_timestamp(),
                symbol,
                previous_candle['date_time'] if previous_candle else "N/A",
                previous_candle['open'] if previous_candle else 0,
                previous_candle['high'] if previous_candle else 0,
                previous_candle['low'] if previous_candle else 0,
                previous_candle['close'] if previous_candle else 0,
                buy_price,
                sl,
                target,
                round(candle_range, 2),
                sl_type,
                get_ist_timestamp()
            ]
            
            writer.writerow(row_data)
            logger.info(f"Candle analysis logged to CSV: {symbol} - SL: {sl}, Target: {target}, Type: {sl_type}")
        
        logger.debug(f"Candle analysis successfully written to: {candle_analysis_file}")
        
    except Exception as e:
        logger.error(f"Error logging candle analysis to CSV: {e}")

# ================== BROKER LOGIN ==================
def login_broker():
    global broker_client, client_loaded
    logger.info("Starting broker login process")
    try:
        logger.info("Reading client info from ClientInfo.txt")
        df = pd.read_csv("ClientInfo.txt")
        row = df.iloc[0]
        
        logger.info("Generating 2FA token")
        twoFA = pyotp.TOTP(row["token"]).now()
        logger.debug(f"2FA token generated: {twoFA}")

        client = ShoonyaApiPy()
        logger.info("Attempting broker login...")
        ret = client.login(
            userid=row["Client ID"], password=row["Password"], twoFA=twoFA,
            vendor_code=row["vc"], api_secret=row["app_key"], imei=row["imei"]
        )
        
        logger.debug(f"Login API response: {ret}")
        
        if ret and ret.get("stat") == "Ok":
            broker_client = client
            client_loaded = True
            logger.info("Broker login successful")
            return True
        else:
            logger.error(f"Login failed: {ret}")
            return False
    except Exception as e:
        logger.error(f"Login error: {e}")
        return False

# ================== ORDER MANAGEMENT ==================
def place_buy_order(stock, trigger_price):
    logger.info(f"PLACE BUY ORDER - Stock: {stock}, Trigger Price: {trigger_price}")
    if not broker_client:
        logger.error("No broker client available for placing buy order")
        return False, "NO_CLIENT"
    
    try:
        trading_symbol = f"{stock}-EQ" if not stock.endswith('-EQ') else stock
        logger.info(f"Placing BUY order: {trading_symbol} @ {trigger_price}, Qty: {ORDER_QUANTITY}")
        
        result = broker_client.place_order(
            buy_or_sell="B", product_type="I", exchange="NSE",
            tradingsymbol=trading_symbol, quantity=ORDER_QUANTITY, discloseqty=0, price_type="MKT",
            price=0, trigger_price=0, retention="DAY", remarks=f"Chartink_{stock}"
        )
        
        logger.debug(f"Buy order API response: {result}")
        
        if result and result.get("stat") == "Ok":
            order_id = result.get("norenordno", "UNKNOWN")
            logger.info(f"BUY ORDER PLACED SUCCESS: {trading_symbol} - ID: {order_id}")
            return True, trading_symbol
        else:
            error_msg = result.get("emsg", "FAILED")
            logger.error(f"BUY ORDER FAILED: {trading_symbol} - {error_msg}")
            return False, error_msg
    except Exception as e:
        logger.error(f"BUY ORDER EXCEPTION: {e}")
        return False, "ERROR"

def place_sell_order(symbol, quantity=ORDER_QUANTITY):
    logger.info(f"PLACE SELL ORDER - Symbol: {symbol}, Quantity: {quantity}")
    try:
        trading_symbol = symbol if symbol.endswith('-EQ') else f"{symbol}-EQ"
        logger.info(f"Placing SELL order: {trading_symbol}, Qty: {quantity}")
        
        result = broker_client.place_order(
            buy_or_sell="S", product_type="I", exchange="NSE",
            tradingsymbol=trading_symbol, quantity=quantity, discloseqty=0, price_type="MKT",
            price=0, trigger_price=0, retention="DAY", remarks=f"Exit_{symbol}"
        )
        
        logger.debug(f"Sell order API response: {result}")
        
        if result and result.get("stat") == "Ok":
            order_id = result.get("norenordno", "UNKNOWN")
            logger.info(f"SELL ORDER PLACED SUCCESS: {trading_symbol} - ID: {order_id}")
            return True, order_id
        else:
            error_msg = result.get("emsg", "FAILED")
            logger.error(f"SELL ORDER FAILED: {trading_symbol} - {error_msg}")
            return False, error_msg
    except Exception as e:
        logger.error(f"SELL ORDER EXCEPTION: {e}")
        return False, "ERROR"

# ================== POSITION MANAGEMENT ==================
def confirm_and_setup_position(trading_symbol, trigger_price):
    """Confirm position and calculate SL based on previous candle low using trade book fill time"""
    logger.info(f"Starting position setup for {trading_symbol}")

    for attempt in range(3):
        logger.info(f"Position setup attempt {attempt + 1}/3 for {trading_symbol}")
        try:
            # --- Step 1: Get trade book ---
            logger.info("Fetching trade book...")
            trade_book = broker_client.get_trade_book() or []
            logger.debug(f"Trade book entries: {len(trade_book)}")

            matched_trade = None
            for trade in trade_book:
                tsym = trade.get("tsym", "").strip().upper()
                logger.debug(f"Checking trade: {tsym} vs {trading_symbol.upper()}")
                if tsym == trading_symbol.upper():
                    matched_trade = trade
                    logger.info(f"Found matching trade: {trade}")
                    break

            if not matched_trade:
                logger.warning(f"Attempt {attempt+1}: Trade not found for {trading_symbol}")
                time.sleep(2)
                continue

            # --- Step 2: Extract buy price & fill time ---
            buy_price = float(matched_trade.get("flprc", 0) or matched_trade.get("avgprc", 0))
            if buy_price == 0:
                buy_price = trigger_price
                logger.warning(f"Using trigger price as fallback buy price for {trading_symbol}")
            else:
                logger.info(f"Extracted buy price from trade: {buy_price}")

            fill_time_str = matched_trade.get("fltm")
            logger.info(f"Trade fill time: {fill_time_str}")
            ist = pytz.timezone("Asia/Kolkata")
            fill_time = datetime.strptime(fill_time_str, "%d-%m-%Y %H:%M:%S").replace(tzinfo=ist)

            # --- Step 3: Get previous candle based on fill time ---
            logger.info(f"Fetching previous candle for fill time: {fill_time}")
            previous_candle = get_previous_candle_hloc(trading_symbol, reference_time=fill_time)

            # --- Step 4: Calculate SL/Target ---
            logger.info("Calculating SL and Target levels")
            stop_loss, target = calculate_sl_target_from_previous_candle(
                buy_price, previous_candle, trading_symbol
            )

            # --- Step 5: Log analysis ---
            logger.info("Logging candle analysis to CSV")
            log_candle_analysis_to_csv(trading_symbol, buy_price, previous_candle, stop_loss, target)

            logger.info(f"Writing position to state file: {positions_file}")
            with open(positions_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    trading_symbol, buy_price, get_ist_timestamp(),
                    stop_loss, target, "ACTIVE"
                ])

            logger.info(f"POSITION SETUP COMPLETE: {trading_symbol}")
            logger.info(f"Final Position - Buy: {buy_price}, SL: {stop_loss}, Target: {target}")
            return True

        except Exception as e:
            logger.error(f"Error in position setup attempt {attempt+1} for {trading_symbol}: {e}")
            time.sleep(2)

    logger.error(f"FAILED to setup position after 3 attempts: {trading_symbol}")
    return False

def get_ltp(symbol):
    """Get LTP for a symbol"""
    logger.debug(f"Fetching LTP for: {symbol}")
    try:
        clean_symbol = symbol.replace('-EQ', '').strip()
        
        # First try to get from positions
        logger.debug("Checking positions for LTP...")
        positions = broker_client.get_positions() or []
        for pos in positions:
            if pos.get("tsym", "").strip().upper() == symbol.upper():
                ltp = float(pos.get("lp", 0))
                if ltp > 0:
                    logger.debug(f"LTP from positions: {symbol} = {ltp}")
                    return ltp
        
        # Fallback to quotes API
        logger.debug(f"Getting quote for: {clean_symbol}")
        quote = broker_client.get_quotes('NSE', clean_symbol)
        if quote and 'lp' in quote:
            ltp = float(quote['lp'])
            logger.debug(f"LTP from quotes: {symbol} = {ltp}")
            return ltp
        
        logger.warning(f"Could not fetch LTP for {symbol}")
        return None
    except Exception as e:
        logger.error(f"Error getting LTP for {symbol}: {e}")
        return None

# ================== POSITION BOOK RECORDING ==================
def save_position_book():
    """Save complete position book in required format"""
    logger.info("Saving position book...")
    try:
        positions = broker_client.get_positions() or []
        logger.info(f"Found {len(positions)} positions to save")
        
        with open(position_book_file, "a", newline="") as f:
            writer = csv.writer(f)
            
            for pos in positions:
                symbol = pos.get("tsym", "")
                token = pos.get("token", "")
                netqty = int(pos.get("netqty", 0))
                buyqty = int(pos.get("daybuyqty", 0))
                sellqty = int(pos.get("daysellqty", 0))
                buyavg = float(pos.get("daybuyavgprc", 0))
                sellavg = float(pos.get("daysellavgprc", 0))
                ltp = float(pos.get("lp", 0))
                mtm = float(pos.get("urmtom", 0))
                pnl = float(pos.get("rpnl", 0))
                
                logger.debug(f"Processing position: {symbol}, NetQty: {netqty}, LTP: {ltp}")
                
                if buyqty > 0 and sellqty > 0:
                    raw_mtm = sellqty * (sellavg - buyavg)
                else:
                    raw_mtm = 0
                
                is_non_zero = "Yes" if netqty != 0 else "No"
                
                writer.writerow([
                    get_ist_timestamp(),
                    symbol,
                    token,
                    is_non_zero,
                    netqty,
                    buyqty,
                    sellqty,
                    buyavg,
                    sellavg,
                    "I",
                    "",
                    ltp,
                    mtm,
                    pnl,
                    raw_mtm
                ])
        
        logger.info(f"Position book saved with {len(positions)} records to {position_book_file}")
        
    except Exception as e:
        logger.error(f"Error saving position book: {e}")

# ================== MARKET CLOSE FUNCTIONS ==================
def cancel_all_open_orders():
    """Cancel all open orders at 3:15 PM"""
    logger.info("Starting cancel all open orders procedure")
    try:
        logger.info("Fetching order book...")
        order_book = broker_client.get_order_book() or []
        logger.info(f"Found {len(order_book)} orders in order book")
        
        cancelled_count = 0
        
        for order in order_book:
            order_status = order.get("status", "").upper()
            logger.debug(f"Checking order: {order.get('tsym', '')} - Status: {order_status}")
            if order_status in ["PENDING", "TRANSIT", "OPEN"]:
                order_no = order.get("norenordno")
                symbol = order.get("tsym", "")
                
                logger.info(f"Cancelling order: {symbol} - {order_no}")
                result = broker_client.cancel_order(orderno=order_no)
                if result and result.get("stat") == "Ok":
                    logger.info(f"SUCCESS: Cancelled order: {symbol} - {order_no}")
                    cancelled_count += 1
                else:
                    logger.error(f"FAILED to cancel order: {symbol} - {order_no}")
        
        logger.info(f"Order cancellation completed: {cancelled_count} orders cancelled")
        return True
        
    except Exception as e:
        logger.error(f"Error cancelling orders: {e}")
        return False

def close_all_positions():
    """Close all open positions at 3:15 PM"""
    logger.info("Starting close all positions procedure")
    try:
        positions = broker_client.get_positions() or []
        logger.info(f"Found {len(positions)} positions to check")
        
        closed_count = 0
        
        for pos in positions:
            symbol = pos.get("tsym", "")
            netqty = int(pos.get("netqty", 0))
            logger.info(f"Checking position: {symbol}, NetQty: {netqty}")
            
            if netqty > 0:
                logger.info(f"Closing long position: {symbol}, Qty: {netqty}")
                success, order_id = place_sell_order(symbol, netqty)
                if success:
                    logger.info(f"SUCCESS: Closed long position: {symbol} - Qty: {netqty}")
                    closed_count += 1
                else:
                    logger.error(f"FAILED to close long position: {symbol}")
            
            elif netqty < 0:
                logger.info(f"Closing short position: {symbol}, Qty: {abs(netqty)}")
                trading_symbol = symbol if symbol.endswith('-EQ') else f"{symbol}-EQ"
                result = broker_client.place_order(
                    buy_or_sell="B", product_type="I", exchange="NSE",
                    tradingsymbol=trading_symbol, quantity=abs(netqty),discloseqty=0, price_type="MKT",
                    price=0, trigger_price=0, retention="DAY", remarks=f"EOD_Close_{symbol}"
                )
                if result and result.get("stat") == "Ok":
                    logger.info(f"SUCCESS: Closed short position: {symbol} - Qty: {abs(netqty)}")
                    closed_count += 1
                else:
                    logger.error(f"FAILED to close short position: {symbol}")
        
        logger.info(f"Position closing completed: {closed_count} positions closed")
        return True
        
    except Exception as e:
        logger.error(f"Error closing positions: {e}")
        return False

def check_all_positions_closed():
    """Check if all positions are closed (netqty = 0 for all stocks)"""
    logger.info("Checking if all positions are closed")
    try:
        positions = broker_client.get_positions() or []
        logger.info(f"Checking {len(positions)} positions")
        
        for pos in positions:
            symbol = pos.get("tsym", "")
            netqty = int(pos.get("netqty", 0))
            if netqty != 0:
                logger.info(f"Position still open: {symbol} - NetQty: {netqty}")
                return False
        
        logger.info("SUCCESS: All positions are closed")
        return True
        
    except Exception as e:
        logger.error(f"Error checking positions: {e}")
        return False

def market_close_procedure():
    """Execute complete market close procedure"""
    global shutdown_flag
    
    logger.info("=== STARTING MARKET CLOSE PROCEDURE ===")
    logger.info("Current time: " + get_ist_timestamp())
    
    # Step 1: Cancel all open orders
    logger.info("STEP 1: Cancelling all open orders")
    cancel_all_open_orders()
    time.sleep(2)
    
    # Step 2: Close all positions
    logger.info("STEP 2: Closing all positions")
    close_all_positions()
    
    # Step 3: Wait for positions to close
    logger.info("STEP 3: Waiting for positions to close")
    max_attempts = 30
    for attempt in range(max_attempts):
        logger.info(f"Position close check attempt {attempt + 1}/{max_attempts}")
        if check_all_positions_closed():
            logger.info("SUCCESS: All positions closed")
            break
        
        logger.info(f"Waiting for positions to close... ({attempt + 1}/{max_attempts})")
        time.sleep(10)
    
    # Step 4: Save final position book
    logger.info("STEP 4: Saving final position book")
    save_position_book()
    
    # Step 5: Set shutdown flag
    shutdown_flag = True
    logger.info("Shutdown flag set to True")
    
    logger.info("=== MARKET CLOSE PROCEDURE COMPLETED ===")
    
    # Force stop the Flask server and exit
    import os
    import signal
    logger.info("Shutting down Flask application...")
    os.kill(os.getpid(), signal.SIGINT)  # This sends KeyboardInterrupt to stop Flask

# ================== POSITION MONITORING ==================
def monitor_positions():
    """Monitor active positions for SL/Target hits"""
    global position_monitor_running
    
    logger.info("=== STARTING POSITION MONITORING ===")
    processed_exits = set()
    
    while position_monitor_running and not shutdown_flag:
        try:
            logger.debug("Position monitoring cycle started")
            
            # Check for market closing time
            if is_market_closing_time():
                logger.info("3:15 PM detected - initiating market close procedure")
                market_close_procedure()
                break
            
            # Check if positions file exists
            if not os.path.exists(positions_file):
                logger.warning(f"Positions file not found: {positions_file}")
                time.sleep(10)
                continue
            
            # Read positions file
            logger.debug("Reading positions file")
            df = pd.read_csv(positions_file)
            logger.debug(f"Loaded {len(df)} positions from file")
            updated = False
            
            for idx, row in df.iterrows():
                symbol = row["Symbol"]
                status = row["Status"]
                position_id = f"{symbol}_{idx}"
                
                logger.debug(f"Checking position: {symbol}, Status: {status}, ID: {position_id}")
                
                if status != "ACTIVE" or position_id in processed_exits:
                    logger.debug(f"Skipping position - not active or already processed: {symbol}")
                    continue
                
                # Get current position from broker
                logger.debug(f"Fetching broker position for: {symbol}")
                positions = broker_client.get_positions() or []
                current_qty = None
                matched_pos = None

                for pos in positions:
                    tsym = pos.get("tsym", "").strip().upper()
                    netqty = pos.get("netqty", 0)
                    logger.debug(f"Broker position: {tsym} vs {symbol.upper()} (netqty={netqty})")
                    if tsym == symbol.upper():
                        current_qty = int(netqty)
                        matched_pos = pos
                        logger.debug(f"Found matching broker position: {symbol}, NetQty: {current_qty}")
                        break

                if current_qty is None:
                    logger.warning(f"No matching broker position found for {symbol}, skipping exit check")
                    continue
                
                # Check if user manually exited
                if current_qty <= 0:
                    logger.info(f"User manually exited position: {symbol}, NetQty: {current_qty}")
                    # Try LTP first, fallback to broker data
                    exit_price = get_ltp(symbol)
                    if not exit_price or exit_price <= 0:
                        if matched_pos:
                            exit_price = float(matched_pos.get("lp", 0)) or 0.0
                        else:
                            exit_price = 0.0
                    
                    df.at[idx, "Status"] = "USER_EXIT"
                    df.at[idx, "ExitPrice"] = exit_price
                    df.at[idx, "ExitTime"] = get_ist_timestamp()
                    updated = True
                    processed_exits.add(position_id)
                    logger.info(f"USER EXIT recorded: {symbol} @ {exit_price} (NetQty={current_qty})")
                    continue
                
                # --- Normal monitoring flow if position still open ---
                logger.debug(f"Monitoring active position: {symbol}")
                ltp = get_ltp(symbol)
                if ltp is None:
                    logger.warning(f"Could not get LTP for {symbol}, skipping this cycle")
                    continue
                
                buy_price = float(row["BuyPrice"])
                stop_loss = float(row["StopLoss"])
                target = float(row["Target"])
                
                logger.debug(f"Position {symbol}: LTP={ltp}, Buy={buy_price}, SL={stop_loss}, Target={target}")
                
                # Check for SL hit
                if ltp <= stop_loss:
                    logger.warning(f"!!! SL HIT !!!: {symbol} @ {ltp} (SL: {stop_loss})")
                    logger.info(f"Placing exit order for SL: {symbol}, Qty: {current_qty}")
                    success, order_id = place_sell_order(symbol, current_qty)
                    if success:
                        df.at[idx, "Status"] = "SL_HIT"
                        df.at[idx, "ExitPrice"] = ltp
                        df.at[idx, "ExitTime"] = get_ist_timestamp()
                        df.at[idx, "ExitOrderId"] = order_id
                        logger.info(f"SL EXIT SUCCESS: {symbol} @ {ltp}, Order: {order_id}")
                    else:
                        df.at[idx, "Status"] = "SL_FAILED"
                        logger.error(f"SL EXIT FAILED: {symbol}")
                    updated = True
                    processed_exits.add(position_id)
                    
                # Check for target hit
                elif ltp >= target:
                    logger.info(f"!!! TARGET HIT !!!: {symbol} @ {ltp} (Target: {target})")
                    logger.info(f"Placing exit order for target: {symbol}, Qty: {current_qty}")
                    success, order_id = place_sell_order(symbol, current_qty)
                    if success:
                        df.at[idx, "Status"] = "TARGET_HIT"
                        df.at[idx, "ExitPrice"] = ltp
                        df.at[idx, "ExitTime"] = get_ist_timestamp()
                        df.at[idx, "ExitOrderId"] = order_id
                        logger.info(f"TARGET EXIT SUCCESS: {symbol} @ {ltp}, Order: {order_id}")
                    else:
                        df.at[idx, "Status"] = "TARGET_FAILED"
                        logger.error(f"TARGET EXIT FAILED: {symbol}")
                    updated = True
                    processed_exits.add(position_id)
            
            # Save updates to positions file
            if updated:
                logger.info("Saving updated positions to file")
                for col in ["ExitPrice", "ExitTime", "ExitOrderId"]:
                    if col not in df.columns:
                        df[col] = ""
                df.to_csv(positions_file, index=False)
                logger.info("Position book updated successfully")
            
            # Periodic position book save
            if int(time.time()) % 60 == 0:
                logger.debug("Periodic position book save")
                save_position_book()
            
            logger.debug("Position monitoring cycle completed, sleeping for 10 seconds")
            time.sleep(10)
            
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
            time.sleep(10)

# ================== ALERT PROCESSING ==================
def process_alerts():
    """Process Chartink alerts from queue"""
    logger.info("=== STARTING ALERT PROCESSOR ===")
    
    while not shutdown_flag:
        logger.debug("Alert processor checking queue...")
        try:
            alert = alert_queue.get(timeout=10)
            logger.info(f"Processing alert from queue: {alert}")
            
            if alert is None:
                logger.info("Received None alert, continuing...")
                continue
            
            symbol = alert.get("symbol", "").strip()
            trigger_price = float(alert.get("trigger_price", 0))
            
            logger.info(f"Alert details - Symbol: {symbol}, Trigger Price: {trigger_price}")
            
            if not symbol or trigger_price <= 0:
                logger.warning(f"Invalid alert data: {alert}")
                continue
            
            # Check if position already exists
            if os.path.exists(positions_file):
                logger.debug("Checking if position already exists")
                df = pd.read_csv(positions_file)
                existing = df[df["Symbol"].str.upper() == symbol.upper()]
                if not existing.empty and any(existing["Status"] == "ACTIVE"):
                    logger.warning(f"Active position already exists for {symbol}, skipping")
                    continue
            
            # Place buy order
            logger.info(f"Attempting BUY order for: {symbol}")
            success, result = place_buy_order(symbol, trigger_price)
            
            if success:
                logger.info(f"BUY ORDER SUCCESS: {symbol}")
                # Setup position with SL/Target
                setup_success = confirm_and_setup_position(result, trigger_price)
                if setup_success:
                    logger.info(f"POSITION SETUP COMPLETE: {symbol}")
                else:
                    logger.error(f"POSITION SETUP FAILED: {symbol}")
            else:
                logger.error(f"BUY ORDER FAILED: {symbol} - {result}")
            
            # Log alert to CSV
            logger.debug("Logging alert to CSV file")
            with open(alerts_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    get_ist_timestamp(), symbol, trigger_price, 
                    "SUCCESS" if success else "FAILED", result
                ])
            
            logger.info(f"Alert processing completed for: {symbol}")
            
        except queue.Empty:
            logger.debug("Alert queue empty, continuing...")
            continue
        except Exception as e:
            logger.error(f"Alert processing error: {e}")


# ================== FLASK ENDPOINTS ==================
# ================== FLASK ENDPOINTS ==================
@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive webhook alerts from Chartink"""
    logger.info("=== WEBHOOK REQUEST RECEIVED ===")

    try:
        # Try JSON first
        data = request.get_json(silent=True)

        # If no JSON, try form data
        if not data:
            data = request.form.to_dict()

        if not data:
            logger.warning("Empty webhook data received")
            return jsonify({"status": "error", "message": "No data received"}), 400

        logger.info(f"Webhook payload: {data}")

        # Handle both formats: single or multiple
        symbols_raw = (
            data.get("symbol", "").strip()
            or data.get("stocks", "").strip()
        )
        prices_raw = (
            data.get("trigger_price", "").strip()
            or data.get("trigger_prices", "").strip()
        )

        if not symbols_raw:
            logger.error("No symbol in webhook data")
            return jsonify({"status": "error", "message": "No symbol provided"}), 400

        # Split if multiple (comma separated)
        symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        prices = [p.strip() for p in prices_raw.split(",") if p.strip()] if prices_raw else []

        # If number of prices doesn't match, just reuse the first or fallback
        if prices and len(prices) != len(symbols):
            logger.warning(f"Mismatch symbols vs prices count: {symbols_raw} | {prices_raw}")
            if len(prices) == 1:
                prices = prices * len(symbols)  # replicate one price
            else:
                # pad with "0" for missing
                while len(prices) < len(symbols):
                    prices.append("0")

        alerts = []
        for i, sym in enumerate(symbols):
            try:
                trig_price = float(prices[i]) if i < len(prices) else 0.0
            except ValueError:
                trig_price = 0.0
            if trig_price <= 0:
                logger.warning(f"Invalid or missing trigger price for {sym}, skipping")
                continue

            alert_queue.put({"symbol": sym, "trigger_price": trig_price})
            alerts.append({"symbol": sym, "trigger_price": trig_price})
            logger.info(f"Alert queued: {sym} @ {trig_price}")

        if not alerts:
            return jsonify({"status": "error", "message": "No valid alerts queued"}), 400

        return jsonify({"status": "success", "alerts": alerts}), 200

    except Exception as e:
        logger.error(f"Webhook processing error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500



# Alias endpoint so both /webhook and /chartink-webhook work
@app.route('/chartink-webhook', methods=['POST'])
def chartink_webhook():
    """Alias for Chartink webhook"""
    return webhook()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.debug("Health check request received")
    status = {
        "status": "running",
        "timestamp": get_ist_timestamp(),
        "client_loaded": client_loaded,
        "queue_size": alert_queue.qsize(),
        "shutdown_flag": shutdown_flag
    }
    logger.debug(f"Health check response: {status}")
    return jsonify(status)

# ================== MAIN APPLICATION ==================
def main():
    global broker_client, client_loaded, position_monitor_running, shutdown_flag
    
    logger.info("=== TRADING BOT STARTING ===")
    logger.info(f"Start time: {get_ist_timestamp()}")
    
    # Login to broker
    if not login_broker():
        logger.error("Failed to login to broker. Exiting.")
        return

    # Ensure positions file exists
    initialize_positions_file()
    
    # Start alert processor thread
    logger.info("Starting alert processor thread...")
    alert_thread = threading.Thread(target=process_alerts, daemon=True)
    alert_thread.start()
    logger.info("Alert processor thread started")
    
    # Start position monitor thread
    logger.info("Starting position monitor thread...")
    position_monitor_running = True
    monitor_thread = threading.Thread(target=monitor_positions, daemon=True)
    monitor_thread.start()
    logger.info("Position monitor thread started")
    
    # Start Flask app
    logger.info("Starting Flask web server...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        logger.info("Flask server started successfully")
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}")
    
    logger.info("=== TRADING BOT SHUTDOWN ===")
    position_monitor_running = False
    shutdown_flag = True

if __name__ == "__main__":
    logger.info("Application starting from main entry point")
    main()