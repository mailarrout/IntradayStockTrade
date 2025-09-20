import os
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt, QTimer
from datetime import datetime, time
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox, QPushButton, QHBoxLayout, QWidget
from pytz import timezone
IST = timezone('Asia/Kolkata')
import logging

# Set up logger
logger = logging.getLogger(__name__)

class PositionLoader:
    # Configuration constants
    AUTO_REFRESH_INTERVAL = 10000  # 10 seconds
    INITIAL_DELAY = 2000  # 2 seconds delay after client load
    EXIT_CHECK_INTERVAL = 30000  # 30 seconds to check for exit time
    EXIT_TIME = time(15, 15)  # 3:15 PM IST
    
    def __init__(self, ui, client_manager):
        logger.info("Initializing PositionLoader")
        self.ui = ui
        self.client_manager = client_manager
        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_refresh)
        self._last_update = None
        self.exit_timer = QTimer()
        self.exit_timer.timeout.connect(self.check_exit_time)
        self.auto_exit_triggered = False
        
        # Setup tables with headers
        self._setup_tables()
        
        # Connect UI signals
        self.ui.PositionRefreshPushButton.clicked.connect(self.update_positions)
        self.ui.AllClientsRefreshPushButton.clicked.connect(self.update_all_clients_mtm)
        
        # Start auto-refresh after client load
        self.start_auto_refresh()
        self.start_exit_check()

        # Connect SL/Target update button
        self.ui.UpdateSLTgtPushButton.clicked.connect(self.update_sl_target)
        
        # Set default SL/Target values
        self.ui.SLQLine.setText("3000")
        self.ui.TargetQEdit.setText("3000")
        
        # Store current SL/Target values
        self.current_sl = 3000.0
        self.current_target = 3000.0
        logger.info("PositionLoader initialization completed")

    
    def _setup_tables(self):
        """Initialize tables with headers"""
        try:
            logger.info("Setting up tables with headers")
            start_time = datetime.now(IST)
            
            # Position Table - Updated headers
            pos_headers = ["Symbol", "Buy Qty", "Sell Qty", "Net Qty", "Buy Price", "Sell Price", "LTP", "MTM", "PnL", "Action"]
            pos_widths = [150, 80, 80, 80, 80, 80, 80, 80, 80, 100]
            self.ui.PositionTable.setColumnCount(len(pos_headers))
            self.ui.PositionTable.setHorizontalHeaderLabels(pos_headers)
            for i, width in enumerate(pos_widths):
                self.ui.PositionTable.setColumnWidth(i, width)
            
            # Client Position Table
            client_headers = ["Client Name", "Client ID", "MTM", "PnL", "Total"]
            client_widths = [200, 100, 100, 100, 100]
            self.ui.ClientPositionTable.setColumnCount(len(client_headers))
            self.ui.ClientPositionTable.setHorizontalHeaderLabels(client_headers)
            for i, width in enumerate(client_widths):
                self.ui.ClientPositionTable.setColumnWidth(i, width)
                
            setup_time = (datetime.now(IST) - start_time).total_seconds()
            logger.debug(f"Tables initialized in {setup_time:.2f} seconds")
            logger.info("Table setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up tables at {datetime.now(IST).strftime('%H:%M:%S IST')}: {str(e)}")
            raise
    
    def start_auto_refresh(self):
        """Start auto-refresh after initial delay"""
        logger.info(f"Starting auto-refresh with {self.INITIAL_DELAY}ms initial delay")
        QTimer.singleShot(self.INITIAL_DELAY, self.start_timer)
    
    def start_timer(self):
        """Start the auto-refresh timer"""
        if not self.timer.isActive():
            self.timer.start(self.AUTO_REFRESH_INTERVAL)
            self.auto_refresh()
            logger.info(f"Auto-refresh started ({self.AUTO_REFRESH_INTERVAL//1000}s interval)")
    
    def start_exit_check(self):
        """Start the exit time check timer"""
        if not self.exit_timer.isActive():
            self.exit_timer.start(self.EXIT_CHECK_INTERVAL)
            logger.info(f"Exit time check started ({self.EXIT_CHECK_INTERVAL//1000}s interval)")
    
    def stop_updates(self):
        """Stop the auto-refresh timer"""
        if self.timer.isActive():
            self.timer.stop()
            logger.info("Auto-refresh stopped")
        if self.exit_timer.isActive():
            self.exit_timer.stop()
            logger.info("Exit time check stopped")
    
    def check_exit_time(self):
        """Check if it's time to exit all positions"""
        try:
            current_time = datetime.now(IST).time()
            logger.debug(f"Checking exit time. Current time: {current_time.strftime('%H:%M:%S')}, Exit time: {self.EXIT_TIME.strftime('%H:%M:%S')}")
            
            # Check if current time is after or equal to exit time
            if current_time >= self.EXIT_TIME and not self.auto_exit_triggered:
                logger.info(f"Exit time reached ({current_time.strftime('%H:%M:%S')} IST). Exiting all positions.")
                self.auto_exit_triggered = True
                self.exit_all_positions()
            
            # Reset the flag after market hours (after 4:00 PM)
            if current_time >= time(16, 0) and self.auto_exit_triggered:
                self.auto_exit_triggered = False
                logger.info("Auto-exit flag reset for next trading day")
                
        except Exception as e:
            logger.error(f"Error checking exit time: {str(e)}")
    
    def exit_all_positions(self, sl_target_exit=False, reason=""):
        """Exit all open positions"""
        try:
            logger.info(f"Starting exit_all_positions. SL/Target exit: {sl_target_exit}, Reason: {reason}")
            
            if not self._validate_dependencies():
                logger.warning("Cannot exit positions - dependencies not validated")
                return
            
            client_name, client_id, primary_client = self.client_manager.clients[0]
            logger.debug(f"Using primary client: {client_name} ({client_id})")
            
            # Get positions
            positions = primary_client.get_positions()
            if not positions:
                logger.info("No positions found to exit")
                return
            
            logger.info(f"Found {len(positions)} positions to process for exit")
            exit_count = 0
            error_count = 0
            
            for pos in positions:
                try:
                    symbol = pos.get("tsym", "")
                    if not symbol:
                        logger.debug("Skipping position with empty symbol")
                        continue
                    
                    net_qty = int(float(pos.get("netqty", 0)))
                    if net_qty == 0:  # Skip flat positions
                        logger.debug(f"Skipping flat position for {symbol}")
                        continue
                    
                    avg_price = float(pos.get("netavgprc", 0))
                    exchange = pos.get("exch", "")
                    product_alias = pos.get("s_prdt_ali", "").upper()
                    
                    # Map product alias to product code
                    if product_alias == "CNC":
                        product = "C"
                    elif product_alias == "NRML":
                        product = "M"
                    elif product_alias == "MIS":
                        product = "I"
                    elif product_alias == "BO" or product_alias == "BRACKET ORDER":
                        product = "B"
                    elif product_alias == "CO" or product_alias == "COVER ORDER":
                        product = "H"
                    else:
                        product = product_alias  # Fallback to original value if unknown
                    
                    logger.debug(f"Processing position: {symbol}, Net Qty: {net_qty}, Product: {product}")
                    
                    # FIX: Determine if this is an auto-exit (time-based) or SL/Target exit
                    is_auto_exit = self.auto_exit_triggered and not sl_target_exit
                    
                    success = self._place_exit_order(symbol, net_qty, avg_price, exchange, product, 
                                                reason=reason, sl_target_exit=sl_target_exit, auto_exit=is_auto_exit)
                    
                    if success:
                        exit_count += 1
                        logger.info(f"Successfully placed exit order for {symbol}")
                    else:
                        error_count += 1
                        logger.error(f"Failed to place exit order for {symbol}")
                        
                    # Small delay between orders to avoid rate limiting
                    QTimer.singleShot(500, lambda: None)
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing position {symbol} for auto-exit: {str(e)}")
                    continue
            
            # Show appropriate message based on exit type
            if sl_target_exit:
                exit_type = f"{reason} exit"
                message_title = f"{reason} Exit"
            else:
                exit_type = "Auto exit"
                message_title = "Auto Exit"
            
            logger.info(f"{exit_type} completed: {exit_count} successful, {error_count} failed")
            QMessageBox.information(self.ui, message_title, 
                                f"{exit_type} completed: {exit_count} successful, {error_count} failed")
            
        except Exception as e:
            error_msg = f"Auto-exit failed: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self.ui, "Auto Exit Error", error_msg)
    
    def auto_refresh(self):
        """Automatically refresh positions"""
        try:
            update_time = datetime.now(IST)
            self._last_update = update_time.strftime("%H:%M:%S")
            logger.debug(f"Starting auto-refresh at {self._last_update}")
            
            self.update_positions()
            self.update_all_clients_mtm()
            
            self.ui.statusBar().showMessage(f"Last update: {self._last_update}", 5000)
            logger.debug("Auto-refresh completed successfully")
            
        except Exception as e:
            logger.error(f"Auto-refresh failed: {str(e)}")
    
    def _place_exit_order(self, symbol, net_qty, avg_price, exchange, product, 
                        reason="", sl_target_exit=False, auto_exit=False):
        """Place exit order for a position"""
        try:
            logger.info(f"Placing exit order for {symbol}. Net Qty: {net_qty}, Reason: {reason}")
            
            if not self._validate_dependencies():
                logger.warning("Cannot place exit order - dependencies not validated")
                return False
            
            client_name, client_id, primary_client = self.client_manager.clients[0]
            logger.debug(f"Using primary client: {client_name} ({client_id})")
                        
            if sl_target_exit and reason == "Target":

                qty = abs(net_qty)
                if net_qty < 0:  
                    buy_or_sell = "B"
                    transaction_type = "BUY"
                else:  
                    buy_or_sell = "S" 
                    transaction_type = "SELL"
                logger.debug(f"Target hit - reversing position for {symbol}. Qty: {qty}, Type: {transaction_type}")
            else:
                # Normal exit (for SL or manual exit)
                if net_qty < 0:  # Short position - need to buy to exit
                    buy_or_sell = "B"
                    qty = abs(net_qty)
                    transaction_type = "BUY"
                elif net_qty > 0:  # Long position - need to sell to exit
                    buy_or_sell = "S"
                    qty = net_qty
                    transaction_type = "SELL"
                else:  # Flat position - no action needed
                    logger.info(f"No exit needed for {symbol} - position is flat")
                    return False

            # Generate remarks based on the reason
            if reason:
                remarks = f"{reason}_{symbol}"
            else:
                remarks = f"Exit_{symbol}"
                
            if auto_exit:
                remarks += "_Auto"
                
            logger.debug(f"Order parameters: buy_or_sell={buy_or_sell}, product_type={product}, exchange={exchange}, tradingsymbol={symbol}, quantity={qty}, price_type='MKT', remarks='{remarks}'")
            
            # Place the order with correct keyword arguments
            order_result = primary_client.place_order(
                buy_or_sell=buy_or_sell,
                product_type=product,
                exchange=exchange,
                tradingsymbol=symbol, 
                quantity=qty,
                discloseqty=0,
                price_type="MKT",  
                price=0,
                trigger_price=0,
                remarks=remarks
            )
            
            if order_result and order_result.get('stat') == 'Ok':
                log_msg = f"{'Auto-' if auto_exit else ''}{'Target-' if (sl_target_exit and reason == 'Target') else 'SL-' if (sl_target_exit and reason == 'SL') else ''}Exit order placed for {symbol}: {transaction_type} {qty} @ MARKET"
                logger.info(log_msg)
                
                if not auto_exit and not sl_target_exit:
                    QMessageBox.information(self.ui, "Exit Order", log_msg)
                return True
            else:
                error_msg = order_result.get('emsg', 'Unknown error') if order_result else 'No response from broker'
                log_msg = f"Failed to place {'auto-' if auto_exit else ''}{'Target-' if (sl_target_exit and reason == 'Target') else 'SL-' if (sl_target_exit and reason == 'SL') else ''}exit order for {symbol}: {error_msg}"
                logger.error(log_msg)
                
                if not auto_exit and not sl_target_exit:
                    QMessageBox.critical(self.ui, "Exit Order Failed", log_msg)
                return False
                
        except Exception as e:
            error_msg = f"Error placing {'auto-' if auto_exit else ''}{'Target-' if (sl_target_exit and reason == 'Target') else 'SL-' if (sl_target_exit and reason == 'SL') else ''}exit order for {symbol}: {str(e)}"
            logger.error(error_msg)
            
            if not auto_exit and not sl_target_exit:
                QMessageBox.critical(self.ui, "Error", error_msg)
            return False
    
    def _create_exit_button(self, symbol, net_qty, buy_price, sell_price, exchange, product):
        """Create an exit button for a position row"""
        button = QPushButton("Exit")
        button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        
        # Disable button if position is flat
        if net_qty == 0:
            button.setEnabled(False)
        
        # Connect button click to exit function
        button.clicked.connect(lambda: self._place_exit_order(symbol, net_qty, buy_price if net_qty > 0 else sell_price, exchange, product))
        return button
    
    def update_positions(self):
        """Update positions table for the primary client"""
        try:
            logger.info("Starting position update")
            
            if not self._validate_dependencies():
                logger.warning("Cannot update positions - dependencies not validated")
                return
            
            client_name, client_id, primary_client = self.client_manager.clients[0]
            logger.debug(f"Updating positions for client: {client_name} ({client_id})")
            
            # Get positions
            try:
                positions = primary_client.get_positions()
                if positions is None:
                    logger.warning(f"{client_name}: Failed to get positions (None returned)")
                    return
            except Exception as e:
                logger.error(f"{client_name}: Error fetching positions: {str(e)}")
                return
            
            self.ui.PositionTable.setRowCount(0)
            
            if not positions:
                logger.info(f"{client_id}: No positions found")
                self.update_mtm_display(0)
                return
            
            logger.info(f"Found {len(positions)} positions to process")
            total_mtm = 0
            total_pnl = 0
            rows_data = []
            
            for pos in positions:
                try:
                    symbol = pos.get("tsym", "")
                    if not symbol:
                        logger.debug("Skipping position with empty symbol")
                        continue
                    
                    # Get quantity data
                    buy_qty = int(float(pos.get("daybuyqty", 0)))
                    sell_qty = int(float(pos.get("daysellqty", 0)))
                    net_qty = int(float(pos.get("netqty", 0)))
                    
                    # Get price data - updated to include buy and sell prices
                    buy_price = float(pos.get("daybuyavgprc", 0)) if buy_qty > 0 else 0.0
                    sell_price = float(pos.get("daysellavgprc", 0)) if sell_qty > 0 else 0.0
                    ltp = float(pos.get("lp", 0))
                    mtm = float(pos.get("urmtom", 0))
                    pnl = float(pos.get("rpnl", 0))
                    
                    # Get additional info for exit order
                    exchange = pos.get("exch", "")
                    product_alias = pos.get("s_prdt_ali", "").upper()
                    if product_alias == "CNC":
                        product = "C"
                    elif product_alias == "NRML":
                        product = "M"
                    elif product_alias == "MIS":
                        product = "I"
                    elif product_alias == "BO" or product_alias == "BRACKET ORDER":
                        product = "B"
                    elif product_alias == "CO" or product_alias == "COVER ORDER":
                        product = "H"
                    else:
                        product = product_alias  # Fallback to original value if unknown
                    
                    total_mtm += mtm
                    total_pnl += pnl
                    
                    row_data = {
                        "symbol": symbol,
                        "buy_qty": buy_qty,
                        "sell_qty": sell_qty,
                        "net_qty": net_qty,
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "ltp": ltp,
                        "mtm": mtm,
                        "pnl": pnl,
                        "exchange": exchange,
                        "product": product
                    }
                    rows_data.append(row_data)
                    
                    logger.debug(f"Processed position: {symbol}, Net Qty: {net_qty}, MTM: {mtm}, PnL: {pnl}")
                    
                except Exception as e:
                    logger.error(f"Error processing position: {str(e)}")
                    continue
            
            # Sort: Shorts → Longs → Flats
            rows_data.sort(key=lambda x: (0 if x["net_qty"] < 0 else 1 if x["net_qty"] > 0 else 2))
            logger.debug(f"Sorted {len(rows_data)} positions for display")
            
            # Update table
            for row_idx, row_data in enumerate(rows_data):
                self.ui.PositionTable.insertRow(row_idx)
                
                items = [
                    QTableWidgetItem(row_data["symbol"]),
                    QTableWidgetItem(str(row_data["buy_qty"])),
                    QTableWidgetItem(str(row_data["sell_qty"])),
                    QTableWidgetItem(str(row_data["net_qty"])),
                    QTableWidgetItem(f"{row_data['buy_price']:.2f}"),
                    QTableWidgetItem(f"{row_data['sell_price']:.2f}"),
                    QTableWidgetItem(f"{row_data['ltp']:.2f}"),
                    QTableWidgetItem(f"{row_data['mtm']:.2f}"),
                    QTableWidgetItem(f"{row_data['pnl']:.2f}"),
                ]
                
                for col, item in enumerate(items):
                    item.setTextAlignment(Qt.AlignCenter)
                    
                    # Color coding
                    if col == 1 and row_data["buy_qty"] > 0:  # Buy Qty
                        item.setForeground(QColor("green"))
                    elif col == 2 and row_data["sell_qty"] > 0:  # Sell Qty
                        item.setForeground(QColor("red"))
                    elif col == 3:  # Net Qty
                        item.setForeground(QColor("green") if row_data["net_qty"] > 0 else 
                                         QColor("red") if row_data["net_qty"] < 0 else 
                                         QColor("black"))
                    elif col == 4:  # Buy Price
                        if row_data["buy_qty"] > 0:
                            item.setForeground(QColor("green"))
                    elif col == 5:  # Sell Price
                        if row_data["sell_qty"] > 0:
                            item.setForeground(QColor("red"))
                    elif col in [7, 8]:  # MTM and PnL
                        value = float(item.text())
                        item.setForeground(QColor("green") if value > 0 else 
                                         QColor("red") if value < 0 else 
                                         QColor("black"))
                    
                    self.ui.PositionTable.setItem(row_idx, col, item)
                
                # Add exit button in the last column
                exit_button = self._create_exit_button(
                    row_data["symbol"], 
                    row_data["net_qty"], 
                    row_data["buy_price"],
                    row_data["sell_price"],
                    row_data["exchange"],
                    row_data["product"]
                )
                
                # Create a widget to hold the button (for proper centering)
                button_widget = QWidget()
                button_layout = QHBoxLayout(button_widget)
                button_layout.addWidget(exit_button)
                button_layout.setAlignment(Qt.AlignCenter)
                button_layout.setContentsMargins(0, 0, 0, 0)
                button_widget.setLayout(button_layout)
                
                self.ui.PositionTable.setCellWidget(row_idx, 9, button_widget)  # Changed to column 9
            
            # Update MTM display
            current_mtm = total_mtm + total_pnl
            self.check_sl_target_conditions(current_mtm, rows_data)
            self.update_mtm_display(current_mtm)
            
            logger.info(
                f"{client_name}: Positions updated - Valid: {len(rows_data)}, "
                f"MTM: {total_mtm:.2f}, PnL: {total_pnl:.2f}, Total: {current_mtm:.2f}"
            )
            
            self.ui.statusBar().showMessage(f"Positions updated for {client_id} ({len(rows_data)} positions)")
            
        except Exception as e:
            error_msg = f"Position update failed: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self.ui, "Error", error_msg)
    
    def update_mtm_display(self, mtm_value):
        """Update the MTM display label"""
        try:
            if mtm_value is None:
                self.ui.MTMShowQLabel.setText("N/A")
                logger.debug("MTM value is None, setting display to N/A")
                return
                
            mtm_text = f"MTM: {mtm_value:+,.2f}"
            self.ui.MTMShowQLabel.setText(mtm_text)
            
            if mtm_value > 0:
                style = "color: green; font-weight: bold;"
            elif mtm_value < 0:
                style = "color: red; font-weight: bold;"
            else:
                style = "color: white; font-weight: bold;"
                
            self.ui.MTMShowQLabel.setStyleSheet(style)
            logger.debug(f"Updated MTM display to: {mtm_text}")
            
        except Exception as e:
            logger.error(f"Failed to update MTM display: {str(e)}")
            self.ui.MTMShowQLabel.setText("Error")
    
    def update_all_clients_mtm(self):   
        """Update MTM for all clients in the clients table"""
        try:
            logger.info("Starting update of all clients MTM")
            
            if not self._validate_dependencies():
                logger.warning("Cannot update client MTM - dependencies not validated")
                return
            
            table = self.ui.ClientPositionTable  
            table.setRowCount(0)
            
            successful_updates = 0
            total_clients = len(self.client_manager.clients)
            logger.debug(f"Processing MTM for {total_clients} clients")
            
            for row, (name, client_id, client) in enumerate(self.client_manager.clients):
                try:
                    # Get positions
                    positions = client.get_positions() or []
                    
                    # Calculate MTM and PnL
                    mtm = sum(float(p.get("urmtom", 0)) for p in positions)
                    pnl = sum(float(p.get("rpnl", 0)) for p in positions)
                    total = mtm + pnl
                    
                    # Add row to table
                    table.insertRow(row)
                    
                    # Create table items
                    name_item = QTableWidgetItem(name)
                    name_item.setTextAlignment(Qt.AlignCenter)
                    
                    id_item = QTableWidgetItem(client_id)
                    id_item.setTextAlignment(Qt.AlignCenter)
                    
                    mtm_item = QTableWidgetItem(f"{mtm:+,.2f}")
                    mtm_item.setTextAlignment(Qt.AlignCenter)
                    mtm_item.setForeground(QColor('green' if mtm >= 0 else 'red'))
                    
                    pnl_item = QTableWidgetItem(f"{pnl:+,.2f}")
                    pnl_item.setTextAlignment(Qt.AlignCenter)
                    pnl_item.setForeground(QColor('green' if pnl >= 0 else 'red'))
                    
                    total_item = QTableWidgetItem(f"{total:+,.2f}")
                    total_item.setTextAlignment(Qt.AlignCenter)
                    total_item.setForeground(QColor('green' if total >= 0 else 'red'))
                    
                    # Set items in table
                    table.setItem(row, 0, name_item)
                    table.setItem(row, 1, id_item)
                    table.setItem(row, 2, mtm_item)
                    table.setItem(row, 3, pnl_item)
                    table.setItem(row, 4, total_item)
                    
                    successful_updates += 1
                    logger.debug(f"Updated MTM for client {name}: MTM={mtm:.2f}, PnL={pnl:.2f}, Total={total:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error processing client {name}: {str(e)}")
                    continue
                    
            logger.info(f"Successfully updated {successful_updates}/{total_clients} client MTMs")
            
        except Exception as e:
            logger.error(f"Client MTM update failed: {str(e)}")
    
    def _validate_dependencies(self):
        """Validate required dependencies are available"""
        if not self.client_manager or not self.client_manager.clients:
            logger.warning("No clients available")
            return False
        return True
    

    def update_sl_target(self):
        """Update SL and Target values from UI"""
        try:
            logger.info("Updating SL and Target values")
            sl_text = self.ui.SLQLine.text().strip()
            target_text = self.ui.TargetQEdit.text().strip()
            
            if sl_text:
                self.current_sl = float(sl_text)
                logger.debug(f"SL updated to: {self.current_sl}")
            if target_text:
                self.current_target = float(target_text)
                logger.debug(f"Target updated to: {self.current_target}")
                
            logger.info(f"SL/Target updated - SL: {self.current_sl}, Target: {self.current_target}")
            QMessageBox.information(self.ui, "SL/Target Updated", 
                                f"SL: {self.current_sl}\nTarget: {self.current_target}")
            
        except ValueError:
            logger.error("Invalid SL/Target values - must be numbers")
            QMessageBox.critical(self.ui, "Error", "SL and Target must be valid numbers")
        except Exception as e:
            logger.error(f"Error updating SL/Target: {str(e)}")
            QMessageBox.critical(self.ui, "Error", f"Failed to update SL/Target: {str(e)}")    


    def check_sl_target_conditions(self, current_mtm, positions_data):
        """Check if MTM hits SL or Target and exit positions accordingly"""
        try:
            logger.debug(f"Checking SL/Target conditions. Current MTM: {current_mtm:.2f}, SL: {-abs(self.current_sl):.2f}, Target: {abs(self.current_target):.2f}")
            
            if current_mtm <= -abs(self.current_sl):
                logger.warning(f"SL hit! Current MTM: {current_mtm:.2f}, SL: {-abs(self.current_sl):.2f}")
                self.exit_all_positions(sl_target_exit=True, reason="SL")
                
            elif current_mtm >= abs(self.current_target):
                logger.warning(f"Target hit! Current MTM: {current_mtm:.2f}, Target: {abs(self.current_target):.2f}")
                self.exit_all_positions(sl_target_exit=True, reason="Target")
                
        except Exception as e:
            logger.error(f"Error checking SL/Target conditions: {str(e)}")