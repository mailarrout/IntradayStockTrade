import os
import pandas as pd
import logging
from datetime import datetime, timedelta, date
import pytz
from PyQt5.QtCore import QTimer
from client_loader import load_clients
from historical_loader import load_token_map
from api_helper import ShoonyaApiPy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OneMinuteDataFetcher:
    def __init__(self):
        self.ui_reference = type('Obj', (object,), {})()  # Mock UI reference
        self.clients_loaded = False
        self.token_map = None
        
    def load_clients_and_tokens(self):
        """Load clients and token mapping"""
        try:
            # Load clients
            logger.info("Loading clients...")
            self.ui_reference.clients = []
            success = load_clients(self.ui_reference, auto_load=True)
            
            if not success or not hasattr(self.ui_reference, 'clients') or not self.ui_reference.clients:
                logger.error("Failed to load clients")
                return False
                
            self.clients_loaded = True
            logger.info(f"Loaded {len(self.ui_reference.clients)} clients")
            
            # Load token mapping
            logger.info("Loading token mapping...")
            self.token_map = load_token_map()
            logger.info(f"Loaded {len(self.token_map)} tokens")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading clients and tokens: {str(e)}")
            return False
    
    def read_stock_list(self, file_path="app/stocklist.csv"):
        """Read stock symbols from CSV file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Stock list file not found: {os.path.abspath(file_path)}")
                return None
                
            df = pd.read_csv(file_path)
            if df.empty:
                logger.error("Stock list file is empty")
                return None
                
            # Assuming the CSV has a column named 'Symbol' or similar
            symbol_column = None
            for col in df.columns:
                if 'symbol' in col.lower():
                    symbol_column = col
                    break
            
            if symbol_column is None:
                # Try to use first column
                symbol_column = df.columns[0]
                
            symbols = df[symbol_column].dropna().str.strip().str.upper().tolist()
            logger.info(f"Read {len(symbols)} symbols from {file_path}")
            return symbols
            
        except Exception as e:
            logger.error(f"Error reading stock list: {str(e)}")
            return None
    
    def get_previous_trading_day_data(self, symbol, client):
        """Get data for the previous trading day with proper error handling"""
        try:
            ist = pytz.timezone("Asia/Kolkata")
            today = datetime.now(ist).date()
            
            # Try up to 7 days back to find a trading day
            for days_back in range(1, 8):
                target_date = today - timedelta(days=days_back)
                logger.info(f"Trying date {target_date} for {symbol}")
                
                # Get daily data for this date
                start_date = int(datetime.combine(target_date, datetime.min.time()).timestamp())
                end_date = int(datetime.combine(target_date, datetime.max.time()).timestamp())
                
                ret = client.get_daily_price_series(
                    exchange="NSE",
                    tradingsymbol=f"{symbol}-EQ",
                    startdate=str(start_date),
                    enddate=str(end_date)
                )
                
                # Check if response is valid
                if ret is None:
                    logger.debug(f"No response for {symbol} on {target_date}")
                    continue
                    
                if isinstance(ret, str):
                    logger.debug(f"String response for {symbol} on {target_date}: {ret}")
                    continue
                    
                if isinstance(ret, dict):
                    # Check if it's an error response
                    if ret.get('stat') == 'Not_Ok':
                        logger.debug(f"Error response for {symbol} on {target_date}: {ret.get('emsg', 'Unknown error')}")
                        continue
                    # If it's a single dict, treat it as data
                    high = float(ret.get('h', 0))
                    low = float(ret.get('l', 0))
                    if high > 0 and low > 0:
                        logger.info(f"Found trading data for {symbol} on {target_date}: High={high}, Low={low}")
                        return high, low, target_date
                
                elif isinstance(ret, list) and len(ret) > 0:
                    # Multiple days data, get the first one
                    day_data = ret[0]
                    if isinstance(day_data, dict):
                        high = float(day_data.get('h', 0))
                        low = float(day_data.get('l', 0))
                        if high > 0 and low > 0:
                            logger.info(f"Found trading data for {symbol} on {target_date}: High={high}, Low={low}")
                            return high, low, target_date
                    else:
                        logger.debug(f"Unexpected data format in list for {symbol}")
                        continue
                else:
                    logger.debug(f"Unexpected response format for {symbol}: {type(ret)}")
                    continue
                    
            logger.warning(f"No trading data found for {symbol} in the past 7 days")
            return None, None, None
            
        except Exception as e:
            logger.error(f"Error getting previous trading day data for {symbol}: {str(e)}")
            return None, None, None
    
    def fetch_1min_data_for_symbol(self, symbol, client, target_date):
        """Fetch 1-minute candle data for a single symbol for a specific date"""
        try:
            if symbol not in self.token_map:
                logger.warning(f"Token not found for {symbol}")
                return None
            
            token = self.token_map[symbol]
            logger.info(f"Fetching 1-min data for {symbol} (token={token}) for date {target_date}")
            
            # Get time in IST
            ist = pytz.timezone("Asia/Kolkata")
            
            # Set market hours for the target date
            market_open = ist.localize(datetime.combine(target_date, datetime.min.time().replace(hour=9, minute=15)))
            market_close = ist.localize(datetime.combine(target_date, datetime.min.time().replace(hour=15, minute=30)))
            
            # Convert to UTC timestamps
            starttime = int(market_open.astimezone(pytz.UTC).timestamp())
            endtime = int(market_close.astimezone(pytz.UTC).timestamp())
            
            # Fetch 1-minute data
            data = client.get_time_price_series(
                exchange="NSE",
                token=token,
                starttime=starttime,
                endtime=endtime,
                interval=1  # 1-minute interval
            )
            
            # Handle different response formats
            if data is None:
                logger.warning(f"No data returned for {symbol}")
                return None
                
            if isinstance(data, str):
                logger.warning(f"String response for {symbol}: {data}")
                return None
                
            if isinstance(data, dict) and data.get('stat') == 'Not_Ok':
                logger.warning(f"Error response for {symbol}: {data.get('emsg', 'Unknown error')}")
                return None
                
            if isinstance(data, list):
                df = pd.DataFrame(data)
                if not df.empty:
                    logger.info(f"Fetched {len(df)} 1-minute candles for {symbol}")
                    return df
                else:
                    logger.warning(f"No 1-minute data returned for {symbol}")
                    return None
            else:
                logger.warning(f"Unexpected data format for {symbol}: {type(data)}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching 1-min data for {symbol}: {str(e)}")
            return None
    
    def process_all_symbols(self):
        """Main method to process all symbols"""
        # Step 1: Load clients and tokens
        if not self.load_clients_and_tokens():
            return False
        
        # Step 2: Read stock list
        symbols = self.read_stock_list("app/stocklist.csv")
        if not symbols:
            return False
        
        # Get the first client
        client_name, client_id, client = self.ui_reference.clients[0]
        
        results = []
        processed_count = 0
        skipped_count = 0
        
        for symbol in symbols:
            try:
                logger.info(f"Processing symbol: {symbol}")
                
                # Step 3: Get previous trading day's high/low
                high, low, trading_date = self.get_previous_trading_day_data(symbol, client)
                
                if high is None or low is None or trading_date is None:
                    logger.warning(f"Skipping {symbol} - no trading data found")
                    skipped_count += 1
                    continue
                
                # Step 4: Get 1-minute data for the same trading day
                one_min_data = self.fetch_1min_data_for_symbol(symbol, client, trading_date)
                
                if one_min_data is not None:
                    # Save to file with date in filename
                    output_dir = os.path.join(os.getcwd(), "OneMinuteData")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    filename = f"{symbol}_{trading_date.strftime('%Y-%m-%d')}_1min.csv"
                    filepath = os.path.join(output_dir, filename)
                    one_min_data.to_csv(filepath, index=False)
                    
                    logger.info(f"Saved 1-minute data for {symbol} to {filepath}")
                
                # Store results
                results.append({
                    'symbol': symbol,
                    'trading_date': trading_date.strftime('%Y-%m-%d'),
                    'high': high,
                    'low': low,
                    'data_file': filepath if one_min_data is not None else 'Failed'
                })
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {str(e)}")
                skipped_count += 1
                continue
        
        # Save summary results
        if results:
            summary_df = pd.DataFrame(results)
            summary_file = os.path.join(os.getcwd(), "OneMinuteData", "summary_results.csv")
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Saved summary results to {summary_file}")
        
        logger.info(f"Processing completed. Processed: {processed_count}, Skipped: {skipped_count}")
        return processed_count > 0

# Usage
if __name__ == "__main__":
    fetcher = OneMinuteDataFetcher()
    
    # Process all symbols from app/stocklist.csv
    success = fetcher.process_all_symbols()
    
    if success:
        logger.info("Data fetching completed successfully")
    else:
        logger.error("Data fetching failed - no symbols were processed")