# FiveMinInvertedHammer.py
import pandas as pd
import os
import glob
from datetime import datetime
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

# Global variables for trade management - SYNCED with previous programs
MAX_TRADES_PER_DAY = 3
TRADE_QUANTITY = 10
active_trades = {}

"""
Intraday short-only strategy built around the 9:15 AM opening candle. 
Confirms setup using two consecutive higher-volume candles 
and a breakdown below reference lows. 
Uses partial profit booking (75%) and moves stop loss to cost.
"""
def run_strategy(ui_reference):
    """Main strategy function to be called from historical_loader.py"""
    try:
        logger.info("Starting strategy execution")
        
        # Get current date directory
        current_date = datetime.now().strftime('%Y-%m-%d')
        historical_data_dir = os.path.join('HistoricalData', current_date)
        
        if not os.path.exists(historical_data_dir):
            logger.info(f"No historical data directory found for today: {historical_data_dir}")
            return
        
        logger.info(f"Historical data directory found: {historical_data_dir}")
        
        # Find all CSV files in today's directory
        csv_files = glob.glob(os.path.join(historical_data_dir, '*.csv'))
        
        if not csv_files:
            logger.info(f"No CSV files found in {historical_data_dir}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to analyze for {current_date}")
        
        # Load already invalidated stocks from previous runs
        invalid_stocks = load_invalid_stocks(current_date)
        logger.info(f"Loaded {len(invalid_stocks)} invalid stocks from previous runs")
        
        # Load already triggered stocks to avoid duplicate signals
        triggered_stocks = load_triggered_stocks(current_date)
        logger.info(f"Loaded {len(triggered_stocks)} triggered stocks from previous runs")
        
        # Load active trades for monitoring
        load_active_trades(current_date)
        logger.info(f"Loaded {len(active_trades)} active trades")
        
        # Filter out invalid and already triggered stocks
        valid_csv_files = []
        for f in csv_files:
            stock_name = os.path.basename(f).replace('.csv', '')
            if stock_name not in invalid_stocks and stock_name not in triggered_stocks:
                valid_csv_files.append(f)
        
        if invalid_stocks:
            logger.info(f"Skipping {len(invalid_stocks)} already invalidated stocks: {', '.join(invalid_stocks)}")
        
        if triggered_stocks:
            logger.info(f"Skipping {len(triggered_stocks)} already triggered stocks: {', '.join(triggered_stocks)}")
        
        if not valid_csv_files:
            logger.info("No valid stocks to analyze (all have been invalidated or triggered)")
            # Still monitor active trades even if no new stocks to analyze
            monitor_active_trades(ui_reference, current_date)
            return
        
        logger.info(f"Analyzing {len(valid_csv_files)} valid stocks")
        
        # Create results list
        results = []
        newly_invalid_stocks = []
        newly_triggered_stocks = []
        
        # Analyze each file
        for file_path in valid_csv_files:
            stock_name = os.path.basename(file_path).replace('.csv', '')
            logger.debug(f"Analyzing stock: {stock_name}")
            
            result, is_invalid, is_triggered = analyze_stock_file(file_path, ui_reference)
            
            if result:
                results.append(result)
                newly_triggered_stocks.append(stock_name)
                logger.info(f"Trade signal found for {stock_name}")
            elif is_invalid:
                newly_invalid_stocks.append(stock_name)
                logger.info(f"Stock {stock_name} marked as invalid")
            elif is_triggered:
                newly_triggered_stocks.append(stock_name)
                logger.info(f"Stock {stock_name} already triggered")
        
        # Save newly invalidated stocks
        if newly_invalid_stocks:
            save_invalid_stocks(newly_invalid_stocks, current_date)
            logger.info(f"Marked {len(newly_invalid_stocks)} stocks as invalid: {', '.join(newly_invalid_stocks)}")
        
        # Save newly triggered stocks
        if newly_triggered_stocks:
            save_triggered_stocks(newly_triggered_stocks, current_date)
            logger.info(f"Marked {len(newly_triggered_stocks)} stocks as triggered: {', '.join(newly_triggered_stocks)}")
        
        # Execute trades for the first 2 signals (if we have capacity)
        executed_trades = execute_trades(results, ui_reference, current_date)
        
        # Save results to CSV
        if results:
            save_results_to_csv(results, current_date, ui_reference)
            
        logger.info(f"Strategy analysis completed. Found {len(results)} potential trades, executed {executed_trades} trades")
        
        # Monitor active trades for SL breaches and target achievements
        monitor_active_trades(ui_reference, current_date)
        
    except Exception as e:
        logger.error(f"Error in strategy execution: {str(e)}")

def load_triggered_stocks(current_date):
    """Load list of stocks that have already triggered trades - SYNCED"""
    path = os.path.join('app', f'{current_date}_triggered_stocks.csv')
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"load_triggered_stocks error: {e}")
        return []

def save_triggered_stocks(stocks, current_date):
    """Save list of triggered stocks - SYNCED"""
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
    """Load list of stocks that have already been invalidated - SYNCED"""
    path = os.path.join('app', f'{current_date}_invalid_stocks.csv')
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"load_invalid_stocks error: {e}")
        return []

def save_invalid_stocks(stocks, current_date):
    """Save list of invalidated stocks - SYNCED"""
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
    """Load active trades from CSV file - UPDATED with profit booking fields"""
    global active_trades
    active_trades = {}
    path = os.path.join('app', f'{current_date}_active_trades.csv')
    if not os.path.exists(path):
        return
    try:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            tid = row['trade_id']
            active_trades[tid] = {
                'stock': row['stock'],
                'entry_price': float(row['entry_price']),
                'stop_loss': float(row['stop_loss']),
                'target_price': float(row['target_price']) if 'target_price' in row else None,
                'quantity': int(row['quantity']),
                'entry_time': row['entry_time'],
                'status': row['status'],
                'order_id': str(row.get('order_id', '')),
                'sl_order_id': str(row.get('sl_order_id', '')),
                'profit_booked': bool(row.get('profit_booked', False)) if 'profit_booked' in row else False,
                'remaining_qty': int(row.get('remaining_qty', row['quantity'])) if 'remaining_qty' in row else int(row['quantity']),
                'original_sl': float(row.get('original_sl', row['stop_loss'])) if 'original_sl' in row else float(row['stop_loss'])
            }
        logger.info(f"Loaded {len(active_trades)} active trades")
    except Exception as e:
        logger.error(f"load_active_trades error: {e}")
        active_trades = {}

def save_active_trades(current_date):
    """Save active trades to CSV file - UPDATED with profit booking fields"""
    global active_trades
    app_dir = 'app'
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
    path = os.path.join(app_dir, f'{current_date}_active_trades.csv')
    rows = []
    for tid, info in active_trades.items():
        rows.append({
            'trade_id': tid,
            'stock': info['stock'],
            'entry_price': info['entry_price'],
            'stop_loss': info['stop_loss'],
            'target_price': info.get('target_price', ''),
            'quantity': info['quantity'],
            'remaining_qty': info.get('remaining_qty', info['quantity']),
            'entry_time': info['entry_time'],
            'status': info['status'],
            'order_id': info.get('order_id', ''),
            'sl_order_id': info.get('sl_order_id', ''),
            'profit_booked': info.get('profit_booked', False),
            'original_sl': info.get('original_sl', info['stop_loss'])
        })
    try:
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info(f"Saved {len(rows)} active trades with profit booking info")
    except Exception as e:
        logger.error(f"save_active_trades error: {e}")

def execute_trades(results, ui_reference, current_date):
    """Execute trades for the first 2 signals if we have capacity - UPDATED with profit booking fields"""
    global active_trades
    
    executed_count = 0
    
    logger.info(f"Attempting to execute trades from {len(results)} potential signals")
    
    # Get ALL positions from broker to count total trades (active + closed)
    total_broker_positions_count = 0
    if hasattr(ui_reference, "clients") and ui_reference.clients:
        try:
            client_name, client_id, client = ui_reference.clients[0]
            positions = client.get_positions()
            if positions:
                for pos in positions:
                    netqty = int(float(pos.get('netqty', 0)))
                    symbol = pos.get('tsym', '')
                    if symbol:
                        total_broker_positions_count += 1
        except Exception as e:
            logger.error(f"Error checking broker positions: {e}")
    
    if total_broker_positions_count >= MAX_TRADES_PER_DAY:
        logger.info(f"Already reached max {MAX_TRADES_PER_DAY} trades for the day")
        return 0
    
    remaining_slots = MAX_TRADES_PER_DAY - total_broker_positions_count
    logger.info(f"Remaining trade slots: {remaining_slots}")
    
    for result in results:
        if remaining_slots <= 0:
            break
            
        stock_name = result['Stock']
        
        # Check if we already have a trade for this stock
        if any(trade['stock'] == stock_name for trade in active_trades.values()):
            logger.info(f"Skipping {stock_name}: Already have active trade for this stock")
            continue
            
        entry_price = float(result['Entry_Price'])
        stop_loss = float(result['Stop_Loss'])
        target_price = float(result.get('Target_Price', 0))
        qty = TRADE_QUANTITY

        # Place the actual order
        try:
            if not hasattr(ui_reference, "clients") or not ui_reference.clients:
                logger.error("No clients available for order placement")
                continue
                
            client_name, client_id, client = ui_reference.clients[0]
            
            trading_symbol = f"{stock_name}-EQ" if not stock_name.endswith("-EQ") else stock_name
            
            # Place SELL order for short trade
            logger.info(f"Placing SELL order for {trading_symbol}, Quantity: {qty}")
            order_result = client.place_order(
                buy_or_sell="S",
                product_type="I",
                exchange="NSE",
                tradingsymbol=trading_symbol,
                quantity=qty,
                discloseqty=0,
                price_type="MKT",
                price=0,
                trigger_price=0,
                remarks=f"Strategy_SELL_{stock_name}"
            )
            
            if order_result and order_result.get('stat') == 'Ok':
                order_id = order_result.get('norenordno', '')
                logger.info(f"SELL order placed successfully for {stock_name}, Order ID: {order_id}")
                
                # Place SL order
                logger.info(f"Placing SL order for {stock_name}")
                sl_order_result = client.place_order(
                    buy_or_sell="B",
                    product_type="I",
                    exchange="NSE",
                    tradingsymbol=trading_symbol,
                    quantity=qty,
                    discloseqty=0,
                    price_type="MKT",
                    price=0,
                    trigger_price=0,
                    remarks=f"Strategy_SL_{stock_name}"
                )
                
                sl_order_id = sl_order_result.get('norenordno', '') if sl_order_result and sl_order_result.get('stat') == 'Ok' else ''
                
                # Create trade record with profit booking fields
                trade_id = f"{stock_name}_{datetime.now().strftime('%H%M%S')}"
                active_trades[trade_id] = {
                    'stock': stock_name,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target_price': target_price,
                    'quantity': qty,
                    'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'ACTIVE',
                    'order_id': order_id,
                    'sl_order_id': sl_order_id,
                    'profit_booked': False,
                    'remaining_qty': qty,
                    'original_sl': stop_loss
                }
                
                logger.info(f"TRADE EXECUTED: {stock_name} - Qty: {qty}, Entry: {entry_price:.2f}, SL: {stop_loss:.2f}, Target: {target_price:.2f}")
                
                # Show trade alert
                if hasattr(ui_reference, 'show_trade_alert'):
                    ui_reference.show_trade_alert(
                        stock_name,
                        result['Entry_Point'],
                        entry_price,
                        stop_loss
                    )
                
                executed_count += 1
                remaining_slots -= 1
                logger.info(f"Remaining trade slots: {remaining_slots}")
            else:
                error_msg = order_result.get('emsg', 'Unknown error') if order_result else 'No response from broker'
                logger.error(f"Failed to place order for {stock_name}: {error_msg}")
            
        except Exception as e:
            logger.error(f"Failed to execute trade for {stock_name}: {str(e)}")
    
    # Save active trades
    if executed_count > 0:
        save_active_trades(current_date)
        logger.info(f"Saved {executed_count} new trades to active trades file")
    
    return executed_count

def monitor_active_trades(ui_reference, current_date):
    """Monitor active trades for SL breaches and target achievements - UPDATED with profit booking"""
    global active_trades
    
    if not active_trades:
        logger.debug("No active trades to monitor")
        return
    
    logger.info(f"Monitoring {len(active_trades)} active trades")
    
    # Check actual positions to see if trades are still active
    check_actual_positions(ui_reference, current_date)
    
    historical_data_dir = os.path.join('HistoricalData', current_date)
    if not os.path.exists(historical_data_dir):
        logger.warning(f"Historical data directory not found: {historical_data_dir}")
        return
    
    trades_to_remove = []
    
    for trade_id, trade in list(active_trades.items()):
        if trade['status'] != 'ACTIVE':
            continue
            
        stock_name = trade['stock']
        file_path = os.path.join(historical_data_dir, f"{stock_name}.csv")
        if not os.path.exists(file_path):
            logger.warning(f"Stock data file not found: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            df = df.rename(columns={
                "time": "Date",
                "into": "Open",
                "inth": "High",
                "intl": "Low",
                "intc": "Close",
                "intv": "Volume"
            })
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S')
            df = df.sort_values('Date').reset_index(drop=True)

            if df.empty:
                continue
                
            latest_close = float(df.iloc[-1]['Close'])
            entry_price = float(trade['entry_price'])
            stop_loss = float(trade['stop_loss'])
            target_price = float(trade.get('target_price', 0))
            direction = 'SELL'  # This strategy only does short trades
            remaining_qty = trade.get('remaining_qty', TRADE_QUANTITY)
            profit_booked = trade.get('profit_booked', False)

            # Check for target hit (1:1 profit) - only if not already booked
            if not profit_booked and target_price > 0:
                target_hit = False
                if direction == 'SELL' and latest_close <= target_price:
                    target_hit = True

                if target_hit:
                    logger.info(f"{stock_name}: Target hit at {latest_close:.2f}")
                    # Book 75% profit
                    qty_to_book = int(remaining_qty * 0.75)
                    if exit_partial_trade(ui_reference, stock_name, qty_to_book, "TARGET"):
                        # Update trade info
                        trade['remaining_qty'] = remaining_qty - qty_to_book
                        trade['profit_booked'] = True
                        # Move SL to cost (entry price)
                        trade['stop_loss'] = entry_price
                        logger.info(f"{stock_name}: Booked 75% profit ({qty_to_book} shares), moved SL to cost {entry_price:.2f}")
                        
                        # Log profit booking
                        log_profit_booking(stock_name, entry_price, latest_close, qty_to_book, 
                                         remaining_qty - qty_to_book, current_date)

            # Check SL breach for remaining position
            sl_breached = False
            if direction == 'SELL' and latest_close > stop_loss:
                sl_breached = True

            if sl_breached:
                logger.warning(f"{stock_name}: SL breached at {latest_close:.2f}")
                if exit_trade_by_position(ui_reference, stock_name, "SL"):
                    trade['exit_price'] = latest_close
                    trade['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    trade['status'] = 'EXITED'
                    trade['exit_reason'] = 'SL_BREACH'
                    trade['remaining_qty'] = 0
                    pnl = (entry_price - latest_close) * trade['quantity']
                    trade['pnl'] = pnl
                    trades_to_remove.append(trade_id)
                    
                    # Log SL exit
                    log_trade_exit(stock_name, entry_price, latest_close, trade['quantity'], 
                                 "SL_BREACH", pnl, current_date)

        except Exception as e:
            logger.error(f"Error monitoring trade {trade_id}: {str(e)}")

    for tid in trades_to_remove:
        if tid in active_trades:
            info = active_trades[tid]
            pnl = info.get('pnl', 0)
            logger.info(f"Trade exited: {info['stock']} PnL: {pnl:+.2f}")
            del active_trades[tid]

    if trades_to_remove:
        save_active_trades(current_date)

def check_actual_positions(ui_reference, current_date):
    """Check actual positions to see if trades are still active and handle SL breaches"""
    global active_trades
    
    if not hasattr(ui_reference, "clients") or not ui_reference.clients:
        logger.warning("No clients available for position check")
        return
        
    client_name, client_id, client = ui_reference.clients[0]
    
    try:
        positions = client.get_positions()
        if not positions:
            logger.debug("No positions returned from broker")
            return

        held_positions = {}
        for pos in positions:
            sym = pos.get("tsym", "")
            net_qty = int(float(pos.get("netqty", 0)))
            if sym and net_qty != 0:
                held_positions[sym] = net_qty

        # If any active trade is no longer in held_positions, mark as exited externally
        to_remove = []
        for tid, t in list(active_trades.items()):
            if t['status'] == 'ACTIVE' and t['stock'] not in held_positions:
                logger.warning(f"{t['stock']} appears to have been exited externally")
                t['status'] = 'EXITED'
                t['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                t['exit_reason'] = 'EXTERNAL_EXIT'
                to_remove.append(tid)

        for tid in to_remove:
            del active_trades[tid]

        if to_remove:
            save_active_trades(current_date)

    except Exception as e:
        logger.error(f"Error checking actual positions: {str(e)}")

def analyze_stock_file(file_path, ui_reference):
    """Analyze a single stock CSV file - UPDATED to include target price"""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return None, False, False

    # Get the stock name from filename
    stock_name = os.path.basename(file_path).replace('.csv', '')

    # Rename columns
    plot_df = df.rename(columns={
        "time": "Date",
        "into": "Open",
        "inth": "High",
        "intl": "Low",
        "intc": "Close",
        "intv": "Volume"
    })

    # Convert Date column
    plot_df['Date'] = pd.to_datetime(plot_df['Date'], format='%d-%m-%Y %H:%M:%S')

    # Sort oldest first
    plot_df = plot_df.sort_values('Date').reset_index(drop=True)

    # --- Find the 9:15 candle ---
    nine_fifteen_candle = None
    for _, row in plot_df.iterrows():
        if row['Date'].hour == 9 and row['Date'].minute == 15:
            nine_fifteen_candle = {
                'time': row['Date'],
                'low': float(row['Low']),
                'high': float(row['High'])
            }
            break

    if nine_fifteen_candle is None:
        logger.warning(f"{stock_name}: 9:15 AM candle not found")
        return None, False, False

    # --- Ignore if any candle closed above 9:15 high ---
    for _, row in plot_df.iterrows():
        if float(row['Close']) > nine_fifteen_candle['high']:
            logger.info(f"{stock_name}: IGNORED - Candle closed above 9:15 high")
            return None, True, False

    # --- Check if 9:15 low was broken ---
    day_low_broken = any(float(row['Close']) < nine_fifteen_candle['low'] for _, row in plot_df.iterrows())
    if not day_low_broken:
        logger.debug(f"{stock_name}: 9:15 low not broken")
        return None, False, False

    # --- Find reference candles and entry point ---
    high_volume_count = 0
    reference_candle = None
    entry_candle = None

    for idx in range(1, len(plot_df)):
        current_candle = plot_df.iloc[idx]
        previous_candle = plot_df.iloc[idx - 1]

        # High volume check
        if float(current_candle['Volume']) > float(previous_candle['Volume']):
            high_volume_count += 1

            if high_volume_count == 2:
                reference_candle = {
                    'time': current_candle['Date'],
                    'low': float(current_candle['Low']),
                    'high': float(current_candle['High']),
                    'volume': float(current_candle['Volume']),
                    'index': int(idx)
                }

        # Entry condition
        if reference_candle is not None and float(current_candle['Close']) < reference_candle['low']:
            entry_candle = current_candle
            break

    if entry_candle is None:
        logger.debug(f"{stock_name}: No entry condition met")
        return None, False, False

    # --- Stop loss and target price ---
    entry_high = float(entry_candle['High'])
    prev_high = float(plot_df.iloc[entry_candle.name - 1]['High']) if entry_candle.name > 0 else entry_high
    stop_loss_level = max(entry_high, prev_high)
    
    # Calculate 1:1 target price for short trade
    entry_price = float(entry_candle['Close'])
    risk = abs(entry_price - stop_loss_level)
    target_price = entry_price - risk

    result = {
        'Stock': stock_name,
        'Close_Below_9_15': True,
        'Entry_Point': entry_candle['Date'],
        'Entry_Price': entry_price,
        'Stop_Loss': stop_loss_level,
        'Target_Price': target_price  # Added target price
    }

    logger.info(f"TRADE SIGNAL: {stock_name} - Entry: {entry_price:.2f}, SL: {stop_loss_level:.2f}, Target: {target_price:.2f}")

    return result, False, True

def save_results_to_csv(results, current_date, ui_reference):
    """Save analysis results to CSV file - UPDATED to include Target_Price"""
    try:
        app_dir = 'app'
        if not os.path.exists(app_dir):
            os.makedirs(app_dir)
            
        csv_filepath = os.path.join(app_dir, f'{current_date}_trade_signals.csv')
        
        # Convert results to DataFrame - INCLUDING TARGET PRICE
        df = pd.DataFrame(results)
        
        # Make sure Target_Price column exists (for backward compatibility)
        if 'Target_Price' not in df.columns:
            df['Target_Price'] = None
            for idx, result in enumerate(results):
                if 'Target_Price' in result:
                    df.at[idx, 'Target_Price'] = result['Target_Price']
        
        df.to_csv(csv_filepath, index=False)
        logger.info(f"Trade signals saved to: {csv_filepath}")
        
        # Log all signals with target prices
        for result in results:
            target_price = result.get('Target_Price', 'N/A')
            logger.info(
                f"{result['Stock']}: "
                f"Entry at {result['Entry_Point']} "
                f"(Price: {result['Entry_Price']:.2f}, "
                f"SL: {result['Stop_Loss']:.2f}, "
                f"Target: {target_price if target_price != 'N/A' else 'N/A':.2f})"
            )
            
    except Exception as e:
        logger.error(f"Error saving results to CSV: {str(e)}")

def exit_trade_by_position(ui_reference, symbol, reason=""):
    """Exit trade using actual position data from broker"""
    try:
        if not hasattr(ui_reference, "clients") or not ui_reference.clients:
            logger.error("No clients available for order placement")
            return False
            
        client_name, client_id, client = ui_reference.clients[0]
        
        # Get current positions
        positions = client.get_positions()
        if not positions:
            logger.warning(f"No positions found for {symbol}")
            return False
        
        # Find the specific position
        position_to_exit = None
        for pos in positions:
            pos_symbol = pos.get("tsym", "")
            net_qty = int(float(pos.get("netqty", 0)))
            if pos_symbol == symbol and net_qty != 0:
                position_to_exit = pos
                break
        
        if not position_to_exit:
            logger.warning(f"Position not found for {symbol}")
            return False
        
        # Extract position details
        net_qty = int(float(position_to_exit.get("netqty", 0)))
        exchange = position_to_exit.get("exch", "NSE")
        product_alias = position_to_exit.get("s_prdt_ali", "").upper()
        
        # Map product type
        if product_alias == "CNC":
            product = "C"
        elif product_alias == "NRML":
            product = "M"
        elif product_alias == "MIS":
            product = "I"
        elif product_alias in ("BO", "BRACKET ORDER"):
            product = "B"
        elif product_alias in ("CO", "COVER ORDER"):
            product = "H"
        else:
            product = product_alias
        
        # Determine order parameters based on position type
        if net_qty < 0:  # Short position - need to buy to exit
            buy_or_sell = "B"
            qty = abs(net_qty)
        elif net_qty > 0:  # Long position - need to sell to exit
            buy_or_sell = "S"
            qty = net_qty
        else:
            logger.info(f"No exit needed for {symbol} - position is flat")
            return False

        # Place market order to exit
        order_result = client.place_order(
            buy_or_sell=buy_or_sell,
            product_type=product,
            exchange=exchange,
            tradingsymbol=symbol,
            quantity=qty,
            discloseqty=0,
            price_type="MKT",
            price=0,
            trigger_price=0,
            remarks=f"Strategy_{reason}_{symbol}"
        )
        
        if order_result and order_result.get('stat') == 'Ok':
            logger.info(f"Exit order placed for {symbol}")
            return True
        else:
            error_msg = order_result.get('emsg', 'Unknown error') if order_result else 'No response from broker'
            logger.error(f"Failed to place exit order for {symbol}: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"Error exiting trade for {symbol}: {str(e)}")
        return False

def exit_partial_trade(ui_reference, symbol, quantity, reason=""):
    """Exit partial quantity of a trade"""
    if not hasattr(ui_reference, 'clients') or not ui_reference.clients:
        logger.error("No clients available for partial exit")
        return False
        
    client_name, client_id, client = ui_reference.clients[0]
    try:
        positions = client.get_positions()
        if not positions:
            return False
            
        for pos in positions:
            if pos.get('tsym', '') == symbol:
                net_qty = int(float(pos.get('netqty', 0)))
                if net_qty == 0:
                    continue
                    
                if net_qty > 0:
                    buy_or_sell = "S"  # Selling to reduce long position
                else:
                    buy_or_sell = "B"  # Buying to reduce short position
                    quantity = min(quantity, abs(net_qty))  # Don't exceed position size
                    
                product_alias = pos.get('s_prdt_ali', '').upper()
                if product_alias == "CNC":
                    product = "C"
                elif product_alias == "NRML":
                    product = "M"
                elif product_alias == "MIS":
                    product = "I"
                elif product_alias in ("BO", "BRACKET ORDER"):
                    product = "B"
                elif product_alias in ("CO", "COVER ORDER"):
                    product = "H"
                else:
                    product = "I"

                logger.info(f"Placing partial exit order for {symbol}: {buy_or_sell} {quantity}")
                order_result = client.place_order(
                    buy_or_sell=buy_or_sell,
                    product_type=product,
                    exchange=pos.get("exch", "NSE"),
                    tradingsymbol=symbol,
                    quantity=quantity,
                    discloseqty=0,
                    price_type="MKT",
                    price=0,
                    trigger_price=0,
                    remarks=f"Strategy_Partial_{reason}_{symbol}"
                )
                return order_result and order_result.get('stat') == 'Ok'
                
        return False  # Position not found
                
    except Exception as e:
        logger.exception(f"exit_partial_trade error for {symbol}: {e}")
        return False

def log_profit_booking(stock, entry_price, exit_price, qty_exited, remaining_qty, current_date):
    """Log profit booking to CSV file"""
    app_dir = 'app'
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
        
    log_file = os.path.join(app_dir, f'{current_date}_profit_booking_log.csv')
    
    # Correct PnL calculation for both long and short trades
    pnl = (exit_price - entry_price) * qty_exited
    
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock': stock,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'qty_exited': qty_exited,
        'remaining_qty': remaining_qty,
        'pnl': pnl,
        'action': 'PROFIT_BOOKING'
    }
    
    try:
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([log_entry])
        df.to_csv(log_file, index=False)
        logger.info(f"Logged profit booking for {stock}: PnL {pnl:+.2f}")
    except Exception as e:
        logger.error(f"Error logging profit booking: {e}")

def log_trade_exit(stock, entry_price, exit_price, quantity, reason, pnl, current_date):
    """Log trade exit to CSV file"""
    app_dir = 'app'
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
        
    log_file = os.path.join(app_dir, f'{current_date}_trade_exit_log.csv')
    
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock': stock,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'quantity': quantity,
        'exit_reason': reason,
        'pnl': pnl
    }
    
    try:
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([log_entry])
        df.to_csv(log_file, index=False)
        logger.info(f"Logged trade exit for {stock}")
    except Exception as e:
        logger.error(f"Error logging trade exit: {e}")