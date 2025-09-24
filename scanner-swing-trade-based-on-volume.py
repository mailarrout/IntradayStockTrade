# scanner-swing-trade-based-on-volume.py

import pandas as pd
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
from datetime import datetime, date
import os
import time

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SwingTradeVolumeScraper:
    def __init__(self):
        logger.info("Initializing SwingTradeVolumeScraper")
        self.driver = None
        self.setup_driver()
        
    def setup_driver(self):
        """Setup Chrome driver with options"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--headless")  # Run in background
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Chrome driver setup completed successfully")
        except Exception as e:
            logger.error("Driver setup failed: %s", str(e))
            raise

    def login_to_chartink(self):
        """Login to Chartink website"""
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

    def get_swing_trade_data(self):
        """Fetch swing trade data based on volume"""
        try:
            logger.info("Starting swing trade volume data fetch")
            screener_url = "https://chartink.com/screener/swing-trade-based-on-volume"
            logger.debug("Navigating to URL: %s", screener_url)
            self.driver.get(screener_url)

            # If redirected to login
            if "login" in self.driver.current_url.lower():
                logger.info("Login required. Attempting login.")
                if not self.login_to_chartink():
                    logger.error("Chartink login failed")
                    return None
                logger.debug("Login successful, reloading screener URL")
                self.driver.get(screener_url)

            # Wait for the page to load completely
            logger.debug("Waiting for page to load")
            WebDriverWait(self.driver, 45).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Give some extra time for JavaScript to render content
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
                self.driver.save_screenshot("swing_trade_debug.png")
                logger.error("Screenshot saved as swing_trade_debug.png")
                return None

            logger.info("Successfully parsed %d rows of swing trade data", len(df))
            return df

        except TimeoutException:
            error_msg = "Timeout waiting for page to load. Chartink may be slow."
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return None

    def parse_table_html(self, table_html: str) -> pd.DataFrame:
        """Parse HTML table using BeautifulSoup"""
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
        """Clean and process the dataframe"""
        logger.debug("Cleaning dataframe. Initial shape: %s", df.shape)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df = df.reset_index(drop=True)
        df.columns = [str(col).strip() for col in df.columns]
        
        # Ensure Symbol column exists (case-insensitive)
        symbol_columns = [col for col in df.columns if 'symbol' in col.lower()]
        if symbol_columns:
            df.rename(columns={symbol_columns[0]: 'symbol'}, inplace=True)
        
        logger.debug("Cleaned dataframe shape: %s", df.shape)
        return df

    def create_symbols_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a clean dataframe with ONLY symbol and date columns"""
        if df.empty or 'symbol' not in df.columns:
            return pd.DataFrame(columns=['symbol', 'date'])
            
        # Get today's date
        today_date = date.today().strftime('%Y-%m-%d')
        
        # Create new dataframe with ONLY symbol and date columns
        symbols_df = pd.DataFrame({
            'symbol': df['symbol'],
            'date': today_date
        })
        
        # Remove duplicates and sort
        symbols_df = symbols_df.drop_duplicates(subset=['symbol']).sort_values('symbol')
        symbols_df = symbols_df.reset_index(drop=True)
        
        logger.info("Created clean dataframe with %d unique symbols for date %s", len(symbols_df), today_date)
        return symbols_df

    def save_clean_symbols_data(self, df: pd.DataFrame):
        """Save ONLY symbol and date columns to CSV"""
        try:
            # Create SwingTrade folder if it doesn't exist
            folder_path = 'SwingTrade'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                logger.info("Created SwingTrade directory")

            # Fixed filename
            csv_file = os.path.join(folder_path, 'swing_trade_volume.csv')
            
            if df.empty:
                logger.warning("No data to save")
                return False

            # Create clean dataframe with ONLY symbol and date
            clean_df = self.create_symbols_dataframe(df)
            
            if clean_df.empty:
                logger.warning("No symbols to save")
                return False

            # Check if file already exists
            if os.path.exists(csv_file):
                # Load existing data
                existing_df = pd.read_csv(csv_file)
                
                # Remove existing entries with today's date to avoid duplicates
                today_date = date.today().strftime('%Y-%m-%d')
                existing_df = existing_df[existing_df['date'] != today_date]
                
                # Combine with new data
                combined_df = pd.concat([existing_df, clean_df], ignore_index=True)
                
                logger.info("Merged with existing data: %d total symbol entries", len(combined_df))
                df_to_save = combined_df
            else:
                # First save
                df_to_save = clean_df
                logger.info("Creating new file with %d symbols", len(df_to_save))

            # Save to CSV - ONLY symbol and date columns
            df_to_save.to_csv(csv_file, index=False)
            
            # Verify the saved file
            saved_df = pd.read_csv(csv_file)
            logger.info("Successfully saved %d symbols to %s", len(saved_df), csv_file)
            logger.info("Columns in saved file: %s", list(saved_df.columns))
            
            return True

        except Exception as e:
            logger.error("Error saving data: %s", str(e))
            return False

    def run_scraper(self):
        """Main method to run the scraper"""
        try:
            logger.info("Starting swing trade volume scraper")
            df = self.get_swing_trade_data()
            
            if df is not None and not df.empty:
                success = self.save_clean_symbols_data(df)
                if success:
                    logger.info("Scraper completed successfully")
                    return True
                else:
                    logger.error("Failed to save data")
                    return False
            else:
                logger.error("No data fetched from Chartink")
                return False
                
        except Exception as e:
            logger.error("Scraper failed: %s", str(e))
            return False
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.driver:
                self.driver.quit()
                logger.info("Chrome driver closed")
        except Exception as e:
            logger.error("Cleanup failed: %s", str(e))

# Main execution function
def main():
    """Main function to run the scraper"""
    scraper = SwingTradeVolumeScraper()
    success = scraper.run_scraper()
    
    if success:
        print("Swing trade volume data fetched and saved successfully!")
        print("File: SwingTrade/swing_trade_volume.csv")
        print("Columns: symbol, date")
    else:
        print("Failed to fetch swing trade volume data")
    
    return success

if __name__ == "__main__":
    main()