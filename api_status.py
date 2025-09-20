# api_status.py
from PyQt5.QtCore import QTimer
import logging
import requests
import time

# Get logger for this module
logger = logging.getLogger(__name__)

class ApiStatus:
    def __init__(self, main_window):
        self.main_window = main_window
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_status)
        self.is_online = False
        self.last_successful_check = 0
        self.consecutive_failures = 0
        self.check_interval = 30000  # 30 seconds normal
        self.fast_retry_interval = 5000  # 5 seconds when offline
        logger.info("ApiStatus initialized with normal interval: 30s, fast retry: 5s")
        
    def start(self, interval=30000):
        """Start status monitoring"""
        self.check_interval = interval
        self.timer.start(interval)
        logger.info(f"API status monitoring started with interval: {interval//1000} seconds")
        self.check_status()  # Initial check
        
    def check_status(self):
        """Check API status with smart logging"""
        try:
            logger.debug("Starting API status check")
            response = requests.get("https://api.shoonya.com/", timeout=10)
            is_online = response.status_code < 400
            
            logger.debug(f"API response status code: {response.status_code}")
            
            # Reset failure counter on success
            self.consecutive_failures = 0
            
            if is_online != self.is_online:
                self.is_online = is_online
                status = "ONLINE" if is_online else "OFFLINE"
                
                logger.info(f"API status changed to: {status}")
                
                # Update status bar
                self.main_window.statusbar.showMessage(f"API Status: {status}")
                
                if hasattr(self.main_window, 'ApiStatusLabel'):
                    self.main_window.ApiStatusLabel.setText("ONLINE" if is_online else "OFFLINE")
                
                # Adjust timer based on status
                if is_online:
                    self.timer.setInterval(self.check_interval)
                    logger.info("Timer interval set back to normal: 30 seconds")
                else:
                    self.timer.setInterval(self.fast_retry_interval)
                    logger.info("Timer interval set to fast retry: 5 seconds")
                    
            self.last_successful_check = time.time()
            logger.debug("API status check completed successfully")
            
        except requests.exceptions.Timeout:
            self.consecutive_failures += 1
            logger.warning(f"API connection timeout - consecutive failures: {self.consecutive_failures}")
            self.handle_connection_failure("Timeout")
            
        except requests.exceptions.ConnectionError:
            self.consecutive_failures += 1
            logger.warning(f"API connection error - consecutive failures: {self.consecutive_failures}")
            self.handle_connection_failure("Connection Error")
            
        except requests.exceptions.RequestException as e:
            self.consecutive_failures += 1
            logger.warning(f"API request exception: {e} - consecutive failures: {self.consecutive_failures}")
            self.handle_connection_failure(f"Request Exception: {e}")
            
        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"Unexpected error during API check: {e} - consecutive failures: {self.consecutive_failures}")
            self.handle_connection_failure(f"Unexpected Error: {e}")
    
    def handle_connection_failure(self, error_type):
        """Handle connection failure scenarios"""
        # Only log the first failure and occasional updates
        if self.consecutive_failures == 1:
            logger.error(f"First API connection failure: {error_type}")
        elif self.consecutive_failures % 6 == 0:  # Log every 30 seconds of failures
            logger.warning(f"API still offline - {self.consecutive_failures} consecutive failures, last error: {error_type}")
        
        if self.is_online:  # Status changed from online to offline
            self.is_online = False
            self.main_window.statusbar.showMessage("API Status: OFFLINE")
            if hasattr(self.main_window, 'ApiStatusLabel'):
                self.main_window.ApiStatusLabel.setText("OFFLINE")
            logger.warning("API status changed from ONLINE to OFFLINE")
            
        # Check more frequently when offline to detect recovery faster
        self.timer.setInterval(self.fast_retry_interval)
        logger.debug("Timer interval set to fast retry mode due to connection failure")