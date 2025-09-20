# FiveMinVolumeReference.py
import pandas as pd
import os
import glob
from datetime import datetime
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

# Trade management globals - SYNCED with previous program
MAX_TRADES_PER_DAY = 3
TRADE_QUANTITY = 10
active_trades = {}  # trade_id -> trade info dict
"""
Intraday short-selling strategy based on low-volume reference candles. 
Identifies breakdowns below a green low-volume candle within 2 bars. 
Trades are executed with stop loss and partial profit booking (75%), 
then stop loss is moved to cost for remaining position.
"""

def run_strategy(ui_reference):
    """Main entry point for the strategy (used by your test harness)."""
    try:
        logger.info("Starting Volume Reference strategy execution")
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        historical_data_dir = os.path.join('HistoricalData', current_date)

        if not os.path.exists(historical_data_dir):
            logger.info(f"No historical data directory found for today: {historical_data_dir}")
            return

        logger.info(f"Historical data directory found: {historical_data_dir}")
        
        csv_files = glob.glob(os.path.join(historical_data_dir, '*.csv'))
        if not csv_files:
            logger.info(f"No CSV files found in {historical_data_dir}")
            return

        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Load bookkeeping (so strategy can skip already processed stocks if desired)
        invalid_stocks = load_invalid_stocks(current_date)
        triggered_stocks = load_triggered_stocks(current_date)
        load_active_trades(current_date)

        logger.info(f"Loaded {len(invalid_stocks)} invalid stocks, {len(triggered_stocks)} triggered stocks, {len(active_trades)} active trades")

        valid_files = []
        for f in csv_files:
            stock_name = os.path.basename(f).replace('.csv', '')
            if stock_name not in invalid_stocks and stock_name not in triggered_stocks:
                valid_files.append(f)

        logger.info(f"Found {len(valid_files)} valid CSV files to analyze after filtering")

        results = []
        newly_invalid = []
        newly_triggered = []

        for file_path in valid_files:
            stock_name = os.path.basename(file_path).replace('.csv', '')
            logger.debug(f"Analyzing stock: {stock_name}")
            
            result, is_invalid, is_triggered = analyze_stock_file(file_path, ui_reference)
            if result:
                results.append(result)
                newly_triggered.append(stock_name)
                logger.info(f"Trade signal found for {stock_name}")
            elif is_invalid:
                newly_invalid.append(stock_name)
                logger.info(f"Stock {stock_name} marked as invalid")
            elif is_triggered:
                newly_triggered.append(stock_name)
                logger.info(f"Stock {stock_name} already triggered")

        if newly_invalid:
            save_invalid_stocks(newly_invalid, current_date)
            logger.info(f"Marked {len(newly_invalid)} stocks as invalid: {', '.join(newly_invalid)}")

        if newly_triggered:
            save_triggered_stocks(newly_triggered, current_date)
            logger.info(f"Marked {len(newly_triggered)} stocks as triggered: {', '.join(newly_triggered)}")

        # Execute trades if clients available
        executed = execute_trades(results, ui_reference, current_date)

        if results:
            save_results_to_csv(results, current_date, ui_reference)

        logger.info(f"Strategy analysis completed. Signals found: {len(results)}. Executed: {executed}.")

        # Monitor active trades (SL/Target)
        monitor_active_trades(ui_reference, current_date)

    except Exception as e:
        logger.error(f"Error in strategy execution: {str(e)}")


# -------------------------
# Bookkeeping helpers - UPDATED to match previous program
# -------------------------
def load_triggered_stocks(current_date):
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
    """Load active trades from CSV into global active_trades dict - UPDATED"""
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
    """Persist active_trades to CSV - UPDATED"""
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


# -------------------------
# Order placement / Execution - UPDATED with profit booking logic
# -------------------------
def execute_trades(results, ui_reference, current_date):
    """Execute trades with profit booking logic"""
    global active_trades
    
    executed_count = 0
    
    logger.info(f"Attempting to execute trades from {len(results)} potential signals")
    
    # Get ALL positions from broker to count total trades
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


# -------------------------
# Monitoring & Exit - UPDATED with profit booking logic
# -------------------------
def monitor_active_trades(ui_reference, current_date):
    """Check active trades for target hit and exit if needed with profit booking"""
    global active_trades

    if not active_trades:
        logger.debug("No active trades to monitor")
        return

    logger.info(f"Monitoring {len(active_trades)} active trades")

    # Sync with broker positions
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
    """Check broker positions and update active_trades or exit if necessary."""
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
            if t['stock'] not in held_positions:
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


def exit_trade_by_position(ui_reference, symbol, reason=""):
    """Attempt to exit a trade using broker positions"""
    try:
        if not hasattr(ui_reference, "clients") or not ui_reference.clients:
            logger.error("No clients available for exit order")
            return False
            
        client_name, client_id, client = ui_reference.clients[0]

        positions = client.get_positions()
        if not positions:
            logger.warning(f"No positions available for exit: {symbol}")
            return False
        
        position_to_exit = None
        for pos in positions:
            if pos.get("tsym", "") == symbol and int(float(pos.get("netqty", 0))) != 0:
                position_to_exit = pos
                break

        if not position_to_exit:
            logger.warning(f"Position not found for {symbol}")
            return False

        net_qty = int(float(position_to_exit.get("netqty", 0)))
        exchange = position_to_exit.get("exch", "NSE")
        product_alias = position_to_exit.get("s_prdt_ali", "").upper()

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

        if net_qty < 0:
            buy_or_sell = "B"
            qty = abs(net_qty)
        elif net_qty > 0:
            buy_or_sell = "S"
            qty = net_qty
        else:
            logger.info(f"No exit needed for {symbol} - position flat")
            return False

        logger.info(f"Exiting {symbol}: {buy_or_sell} {qty} shares")

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
            remarks=f"VolRefExit_{reason}_{symbol}"
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
                    remarks=f"VolRef_Partial_{reason}_{symbol}"
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


# -------------------------
# Strategy core: analyze one file - UPDATED to include target price
# -------------------------
def analyze_stock_file(file_path, ui_reference):
    """
    Implements the volume-reference candle strategy with target price calculation.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return None, False, False

    stock_name = os.path.basename(file_path).replace('.csv', '')

    # Rename columns to consistent names
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
        df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values('Date').reset_index(drop=True)

    # --- Step 1: First 3 candles (9:15, 9:20, 9:25) ---
    first_three = df[(df['Date'].dt.hour == 9) & (df['Date'].dt.minute.isin([15, 20, 25]))].sort_values('Date')
    if len(first_three) < 3:
        logger.info(f"{stock_name}: Not enough first-3 candles (9:15-9:25). Skipping.")
        return None, False, False

    # initial reference = lowest volume among first three
    ref_pos = first_three['Volume'].idxmin()
    current_ref = {
        'index': int(ref_pos),
        'low': float(df.loc[ref_pos, 'Low']),
        'high': float(df.loc[ref_pos, 'High']),
        'volume': float(df.loc[ref_pos, 'Volume']),
        'time': df.loc[ref_pos, 'Date']
    }

    # Scan forward for lower-volume opposite candles
    i = current_ref['index'] + 1
    while i < len(df):
        row = df.iloc[i]
        current_volume = float(row['Volume'])
        current_close = float(row['Close'])
        current_open = float(row['Open'])

        # If this candle has lower volume than current reference AND is an 'opposite' candle (close > open)
        if current_volume < current_ref['volume'] and current_close > current_open:
            
            # Mark this as new reference
            current_ref = {
                'index': int(i),
                'low': float(row['Low']),
                'high': float(row['High']),
                'volume': float(row['Volume']),
                'time': row['Date']
            }

            # Check next two candles (i+1 and i+2) for breakout
            for j in (i + 1, i + 2):
                if j < len(df):
                    chk = df.iloc[j]
                    chk_low = float(chk['Low'])
                    chk_close = float(chk['Close'])
                    
                    low_broken = chk_low < current_ref['low']
                    close_below = chk_close < current_ref['low']
                    
                    if low_broken and close_below:
                        # Entry confirmed
                        entry_candle = chk
                        entry_price = float(entry_candle['Close'])
                        stop_loss = float(current_ref['high'])
                        
                        # Calculate 1:1 target price for short trade
                        risk = abs(entry_price - stop_loss)
                        target_price = entry_price - risk

                        result = {
                            'Stock': stock_name,
                            'Entry_Point': entry_candle['Date'],
                            'Entry_Price': float(entry_price),
                            'Stop_Loss': float(stop_loss),
                            'Target_Price': float(target_price)  # Added target price
                        }

                        logger.info(f"TRADE SIGNAL: {stock_name} - Entry: {entry_price:.2f}, SL: {stop_loss:.2f}, Target: {target_price:.2f}")
                        return result, False, True

            # If next two candles did not confirm, continue scanning
        i += 1

    # No entry found
    return None, False, False


# -------------------------
# Output
# -------------------------
def save_results_to_csv(results, current_date, ui_reference):
    try:
        logger.debug(f"Saving {len(results)} trade signals to CSV")
        
        app_dir = 'app'
        if not os.path.exists(app_dir):
            os.makedirs(app_dir)
            logger.debug("Created app directory")
            
        csv_filepath = os.path.join(app_dir, f'{current_date}_volume_reference_signals.csv')
        
        # Convert results to DataFrame - INCLUDING TARGET PRICE
        df = pd.DataFrame(results)
        
        # Make sure all required columns are present
        if 'Target_Price' not in df.columns:
            # Add target price if missing (backward compatibility)
            df['Target_Price'] = None
            for idx, result in enumerate(results):
                if 'Target_Price' in result:
                    df.at[idx, 'Target_Price'] = result['Target_Price']
        
        # Save to CSV with all columns including Target_Price
        df.to_csv(csv_filepath, index=False)
        logger.info(f"Trade signals saved to: {csv_filepath}")
        
        for result in results:
            target_price = result.get('Target_Price', 'N/A')
            logger.info(f"{result['Stock']}: Entry at {result['Entry_Point']} (Price: {float(result['Entry_Price']):.2f}, SL: {float(result['Stop_Loss']):.2f}, Target: {target_price if target_price != 'N/A' else 'N/A'})")
            
    except Exception as e:
        logger.error(f"Error saving results to CSV: {str(e)}")