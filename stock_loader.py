import pandas as pd
from PyQt5.QtWidgets import QMessageBox, QTableView
from PyQt5.QtCore import QAbstractTableModel, Qt, QTimer
import pytz
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import logging
from historical_loader import fetch_historical_for_symbols, start_continuous_fetching, stop_continuous_fetching
from datetime import datetime, date
import os
from PyQt5.QtGui import QColor

# Set up logger
logger = logging.getLogger(__name__)

# Timezone setup for IST
IST = pytz.timezone('Asia/Kolkata')

# ---------------------
# QAbstractTableModel wrapper for Pandas DataFrame
# ---------------------
class PandasModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self._data = data
        self._highlighted_rows = set()
        logger.debug("PandasModel initialized with data shape: %s", data.shape)

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
            
        # Background color for highlighted rows
        if index.isValid() and role == Qt.BackgroundRole:
            if index.row() in self._highlighted_rows:
                return QColor(128, 0, 128)  # Purple color (RGB: 128, 0, 128)
                
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None
        
    def highlightRow(self, row, highlight=True):
        """Highlight or unhighlight a row"""
        if highlight:
            self._highlighted_rows.add(row)
        elif row in self._highlighted_rows:
            self._highlighted_rows.remove(row)
        self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount()-1))

# ---------------------
# Main Stock Loader
# ---------------------
class StockLoader:
    def __init__(self):
        logger.info("Initializing StockLoader")
        self.load_count = 0
        self.max_loads = 3
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_scheduled_load)

        # ✅ NEW: Retry timer (5-min loop until 5 stocks found)
        self.retry_timer = QTimer()
        self.retry_timer.timeout.connect(self.retry_loading)

        self.driver = None
        self.is_loading = False
        self.ui_reference = None
        self.setup_driver()
        logger.info("StockLoader initialization completed")

    # ---------------------
    # Retry Logic (Every 5 min until ≥ 5 stocks)
    # ---------------------
    def start_retry_loading(self):
        logger.info("Starting retry loading every 5 minutes until 5 stocks are found")
        if not self.retry_timer.isActive():
            self.retry_timer.start(5 * 60 * 1000)  # 5 minutes
            self.retry_loading()  # Try immediately once

    def retry_loading(self):
        try:
            logger.info("Retrying stock fetch (5-min loop)")
            success = self.get_stock_data(self.ui_reference, scheduled=True)
            if success:
                table_view = self.ui_reference.StockListQFrame.findChild(QTableView, "StockNameTableView")
                if table_view and table_view.model() and hasattr(table_view.model(), "_data"):
                    df = table_view.model()._data
                    stock_count = len(df)
                    logger.info("Retry load result: %d stocks", stock_count)

                    if stock_count >= 5:
                        logger.info("Minimum 5 stocks found, stopping retry loop")
                        self.retry_timer.stop()
        except Exception as e:
            logger.error("Retry loading failed: %s", str(e))


    def setup_driver(self):
        try:
            logger.info("Setting up Chrome driver")
            chrome_options = Options()
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--start-maximized")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Chrome driver setup completed successfully")
        except Exception as e:
            logger.error("Driver setup failed: %s", str(e))
            raise

    # ---------------------
    # Scheduling
    # ---------------------
    def start_scheduled_loading(self, ui_reference):
        logger.info("Starting scheduled loading")
        self.ui_reference = ui_reference
        self.timer.start(60000)  # every 1 minute
        logger.info("Scheduled stock loading started")
        self.show_next_schedule_time()

    def show_next_schedule_time(self):
        if self.load_count == 0:
            next_time = "9:25 AM"
        elif self.load_count == 1:
            next_time = "9:30 AM"
        elif self.load_count == 2:
            next_time = "9:35 AM"
        else:
            next_time = "No more scheduled loads today"
        logger.info("Next scheduled load: %s", next_time)

    def check_scheduled_load(self):
        try:
            if self.is_loading:
                logger.debug("Already loading, skipping scheduled check")
                return

            current_time = datetime.now(IST).time()
            logger.debug("Checking scheduled load. Current time: %s, Load count: %d",
                         current_time.strftime("%H:%M:%S"), self.load_count)

            scheduled_times = [(9, 25), (9, 30), (9, 35)]
            if self.load_count < self.max_loads:
                target_hour, target_minute = scheduled_times[self.load_count]
                if current_time.hour == target_hour and current_time.minute == target_minute:
                    self.is_loading = True
                    self.load_count += 1
                    logger.info("Scheduled load %d triggered - restarting browser", self.load_count)

                    if self.driver:
                        self.driver.quit()
                        logger.debug("Old driver quit")
                    self.setup_driver()
                    self.get_stock_data(self.ui_reference, scheduled=True)
                    self.is_loading = False

            if self.load_count >= self.max_loads:
                self.timer.stop()
                logger.info("Scheduled loading completed. Timer stopped.")
                
                # ✅ Start retrying every 5 min if < 5 stocks
                table_view = self.ui_reference.StockListQFrame.findChild(QTableView, "StockNameTableView")
                if table_view and table_view.model() and hasattr(table_view.model(), "_data"):
                    df = table_view.model()._data
                    if len(df) < 5:
                        logger.info("Fewer than 5 stocks found (%d). Starting 5-min retry loop.", len(df))
                        self.start_retry_loading()

        except Exception as e:
            logger.error("Error in scheduled load check: %s", str(e))
            self.is_loading = False

    # ---------------------
    # Login
    # ---------------------
    def login_to_chartink(self):
        try:
            logger.info("Attempting to login to Chartink")
            self.driver.get("https://chartink.com/login")

            WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.NAME, "email")))

            email_field = self.driver.find_element(By.NAME, "email")
            password_field = self.driver.find_element(By.NAME, "password")
            submit_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")

            email_field.clear()
            email_field.send_keys("amiya000@gmail.com")
            password_field.clear()
            password_field.send_keys("@679893rmRM")
            submit_button.click()

            WebDriverWait(self.driver, 15).until(
                EC.any_of(
                    EC.url_contains("dashboard"),
                    EC.presence_of_element_located((By.CLASS_NAME, "user-profile"))
                )
            )

            logger.info("Successfully logged into Chartink")
            return True

        except TimeoutException:
            logger.warning("Timeout during Chartink login - may already be logged in")
            return False
        except Exception as e:
            logger.error("Login failed: %s", str(e))
            return False

    # ---------------------
    # Fetch Stock Data
    # ---------------------
    def get_stock_data(self, ui_reference, scheduled=False):
        try:
            logger.info("Starting stock data fetch. Scheduled: %s", scheduled)
            screener_url = "https://chartink.com/screener/5-min-inverted-hammer-day-sell-3"
            # screener_url = "http://chartink.com/screener/top-losser-nifty"
            # screener_url = "https://chartink.com/screener/rsi-with-yesterday-high-break"
            logger.debug("Navigating to URL: %s", screener_url)
            self.driver.get(screener_url)

            # If redirected to login
            if "login" in self.driver.current_url.lower():
                logger.info("Login required. Attempting login.")
                if not self.login_to_chartink():
                    logger.error("Chartink login failed")
                    QMessageBox.critical(ui_reference, "Error", "Chartink login failed")
                    return False
                logger.debug("Login successful, reloading screener URL")
                self.driver.get(screener_url)

            # Wait for the page to load completely - increased timeout
            logger.debug("Waiting for page to load")
            WebDriverWait(self.driver, 45).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Give some extra time for JavaScript to render content
            import time
            logger.debug("Waiting 5 seconds for JavaScript rendering")
            time.sleep(5)

            # Try multiple selectors for the table
            table_selectors = [
                "table.w-full",
                "table.table",
                "table.data-table",
                "table.screener-table",
                "table"
            ]

            df = pd.DataFrame()
            logger.debug("Attempting to find table with %d selectors", len(table_selectors))
            
            for selector in table_selectors:
                try:
                    logger.debug("Trying selector: %s", selector)
                    table_element = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    table_html = table_element.get_attribute("outerHTML")
                    df = self.parse_table_html(table_html)
                    if not df.empty:
                        logger.debug("Successfully found table with selector: %s", selector)
                        break
                except Exception as e:
                    logger.debug("Selector %s failed: %s", selector, str(e))
                    continue

            # If still empty, try to find any table with data
            if df.empty:
                logger.debug("No table found with standard selectors, trying all tables")
                try:
                    all_tables = self.driver.find_elements(By.TAG_NAME, "table")
                    logger.debug("Found %d tables on page", len(all_tables))
                    for i, table in enumerate(all_tables):
                        try:
                            table_html = table.get_attribute("outerHTML")
                            df = self.parse_table_html(table_html)
                            if not df.empty:
                                logger.debug("Found valid table at index %d", i)
                                break
                        except Exception as e:
                            logger.debug("Table %d parsing failed: %s", i, str(e))
                            continue
                except Exception as e:
                    logger.debug("Finding all tables failed: %s", str(e))
                    pass

            if df.empty:
                # Last resort: take screenshot to debug
                logger.error("No stock data found after multiple attempts")
                self.driver.save_screenshot("chartink_debug.png")
                logger.error("Screenshot saved as chartink_debug.png")
                QMessageBox.critical(ui_reference, "Error", "No stock data found. Check debug screenshot.")
                return False

            logger.info("Successfully parsed %d rows of stock data", len(df))
            self.populate_stock_table(ui_reference, df)

            msg = f"{'Scheduled' if scheduled else 'Manual'} load completed with {len(df)} stocks"
            logger.info(msg)
            if not scheduled:
                QMessageBox.information(ui_reference, "Success", msg)

            return True

        except TimeoutException:
            error_msg = "Timeout waiting for page to load. Chartink may be slow."
            logger.error(error_msg)
            QMessageBox.critical(ui_reference, "Error", error_msg)
            return False
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(ui_reference, "Error", error_msg)
            return False

    def parse_table_html(self, table_html: str) -> pd.DataFrame:
        try:
            logger.debug("Parsing table HTML with BeautifulSoup")
            soup = BeautifulSoup(table_html, "html.parser")
            headers = [th.get_text(strip=True) for th in soup.select("thead th")]
            logger.debug("Found %d table headers: %s", len(headers), headers)
            
            rows = []
            for tr in soup.select("tbody tr"):
                row = [td.get_text(strip=True) for td in tr.find_all("td")]
                rows.append(row)
            
            logger.debug("Found %d table rows", len(rows))
            df = pd.DataFrame(rows, columns=headers)
            return self.clean_dataframe(df)
        except Exception as e:
            logger.warning("BeautifulSoup parsing failed: %s. Trying pandas.read_html.", str(e))
            try:
                dfs = pd.read_html(table_html)
                if dfs:
                    logger.debug("pandas.read_html found %d tables", len(dfs))
                    return self.clean_dataframe(dfs[0])
            except Exception as e2:
                logger.error("Both parsing methods failed: %s", str(e2))
            return pd.DataFrame()

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Cleaning dataframe. Initial shape: %s", df.shape)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df = df.reset_index(drop=True)
        df.columns = [str(col).strip() for col in df.columns]
        logger.debug("Cleaned dataframe shape: %s", df.shape)
        return df

    # ---------------------
    # Populate PyQt Table
    # ---------------------
    
    def populate_stock_table(self, ui_reference, df: pd.DataFrame):
        try:
            logger.debug("Populating stock table with %d rows", len(df))
            table_view = ui_reference.StockListQFrame.findChild(QTableView, "StockNameTableView")
            if not table_view:
                logger.error("StockNameTableView not found")
                QMessageBox.warning(ui_reference, "Warning", "Stock table view not found")
                return

            # Sort the DataFrame by Symbol column alphabetically
            if "Symbol" in df.columns:
                df = df.sort_values(by="Symbol", ascending=True)
                logger.debug("Sorted DataFrame by Symbol column")

            # Get current model data if exists
            current_model = table_view.model()
            current_symbols = []
            
            if current_model and hasattr(current_model, '_data'):
                # Append to existing data
                current_df = current_model._data
                current_symbols = current_df["Symbol"].tolist() if "Symbol" in current_df.columns else []
                
                # Merge with new data, avoiding duplicates
                combined_df = pd.concat([current_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["Symbol"], keep="first")
                
                # Sort the combined DataFrame by Symbol column alphabetically
                if "Symbol" in combined_df.columns:
                    combined_df = combined_df.sort_values(by="Symbol", ascending=True)
                
                logger.info("Merged data: %d existing + %d new = %d total symbols", 
                        len(current_symbols), len(df), len(combined_df))
                
                # Load combined dataframe into table
                model = PandasModel(combined_df)
                table_view.setModel(model)
                
                # Get all symbols for historical fetching (existing + new)
                all_symbols = combined_df["Symbol"].tolist()
            else:
                # Load sorted dataframe into table
                model = PandasModel(df)
                table_view.setModel(model)
                all_symbols = df["Symbol"].tolist()
            
            table_view.resizeColumnsToContents()
            logger.info("Table updated with %d total symbols", len(all_symbols))

            # ✅ NEW: Save scanned stocks to file
            current_date = datetime.now().strftime('%Y-%m-%d')
            save_scanned_stocks(all_symbols, current_date)

            # ✅ NEW: Highlight symbols that match position book
            self.highlight_position_book_symbols(ui_reference, table_view, all_symbols)

            # Start/update continuous 5-minute interval fetching with ALL symbols
            print(f"🚀 STARTING CONTINUOUS FETCHING for {len(all_symbols)} symbols")
            print(f"   Symbols: {all_symbols}")
            logger.info("Starting/updating continuous fetching for %d symbols", len(all_symbols))
            start_continuous_fetching(ui_reference, all_symbols)

        except Exception as e:
            error_msg = f"Error populating table: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(ui_reference, "Error", error_msg)

    # ✅ NEW: Add this method to highlight position book symbols
    def highlight_position_book_symbols(self, ui_reference, table_view, symbols):
        """Highlight symbols that exist in position book with purple color"""
        try:
            # Check if we have clients and positions
            if not hasattr(ui_reference, 'clients') or not ui_reference.clients:
                logger.debug("No clients available for position book check")
                return
                
            client_name, client_id, client = ui_reference.clients[0]
            
            # Get positions from broker
            positions = client.get_positions()
            if not positions:
                logger.debug("No positions found in position book")
                return
                
            # Extract symbols from position book (remove -EQ suffix)
            position_symbols = set()
            for pos in positions:
                symbol = pos.get('tsym', '')
                if symbol:
                    # Remove -EQ suffix if present
                    clean_symbol = symbol.replace('-EQ', '')
                    position_symbols.add(clean_symbol)
            
            logger.debug(f"Position book symbols: {list(position_symbols)}")
            
            # Apply highlighting to matching symbols
            model = table_view.model()
            if not model:
                return
                
            # Get the symbol column index
            symbol_col_index = -1
            for col in range(model.columnCount()):
                header = model.headerData(col, Qt.Horizontal, Qt.DisplayRole)
                if header and "symbol" in header.lower():
                    symbol_col_index = col
                    break
            
            if symbol_col_index == -1:
                logger.debug("Symbol column not found in table")
                return
                
            # Custom PandasModel with highlighting support
            class HighlightedPandasModel(PandasModel):
                def __init__(self, data, position_symbols):
                    super().__init__(data)
                    self.position_symbols = position_symbols
                    
                def data(self, index, role=Qt.DisplayRole):
                    if not index.isValid():
                        return None
                        
                    if role == Qt.DisplayRole:
                        return str(self._data.iloc[index.row(), index.column()])
                        
                    # Highlight matching symbols with purple background
                    if role == Qt.BackgroundRole and index.column() == symbol_col_index:
                        symbol = str(self._data.iloc[index.row(), index.column()])
                        if symbol in self.position_symbols:
                            return QColor("#5208A2")   # Purple color (RGB: 128, 0, 128)
                            
                    return None
            
            # Replace the model with highlighted version
            highlighted_model = HighlightedPandasModel(model._data, position_symbols)
            table_view.setModel(highlighted_model)
            
            logger.info(f"Highlighted {len(position_symbols)} symbols from position book")
            
        except Exception as e:
            logger.error(f"Error highlighting position book symbols: {str(e)}")

    # ---------------------
    # Cleanup
    # ---------------------
    def cleanup(self):
        try:
            logger.info("Starting cleanup")
            if self.driver:
                self.driver.quit()
                logger.info("Chrome driver closed")
            if self.timer.isActive():
                self.timer.stop()
                logger.info("Scheduler timer stopped")
            
            # Stop the historical loader timer
            stop_continuous_fetching()
            logger.info("Cleanup completed successfully")
                
        except Exception as e:
            logger.error("Cleanup failed: %s", str(e))

# ---------------------
# Global instance & helper functions
# ---------------------
stock_loader = StockLoader()

def get_stock_name(ui_reference):
    try:
        logger.info("Manual stock data fetch requested")
        return stock_loader.get_stock_data(ui_reference, scheduled=False)
    except Exception as e:
        logger.error("Manual load failed: %s", str(e))
        QMessageBox.critical(ui_reference, "Error", f"Manual load failed: {str(e)}")
        return False

def start_scheduled_loading(ui_reference):
    try:
        logger.info("Starting scheduled loading via helper function")
        stock_loader.start_scheduled_loading(ui_reference)
        return True
    except Exception as e:
        logger.error("Failed to start scheduled loading: %s", str(e))
        QMessageBox.critical(ui_reference, "Error", f"Failed to start scheduled loading: {str(e)}")
        return False

def cleanup_resources():
    logger.info("Cleanup resources requested")
    stock_loader.cleanup()


def save_scanned_stocks(stock_names, current_date):
    """Save scanned stock names to CSV file"""
    try:
        app_dir = 'app'
        if not os.path.exists(app_dir):
            os.makedirs(app_dir)
            logger.debug("Created app directory")
        
        csv_file = os.path.join(app_dir, f'{current_date}_scanned_stocks.csv')
        
        # Create DataFrame with stock names and timestamp
        df = pd.DataFrame({
            'stock_name': stock_names,
            'scanned_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'chartink_screener'
        })
        
        # Append to existing file or create new one
        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            df = pd.concat([existing_df, df]).drop_duplicates(subset=['stock_name'])
        
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved {len(stock_names)} scanned stocks to {csv_file}")
        
    except Exception as e:
        logger.error(f"Error saving scanned stocks: {str(e)}")   