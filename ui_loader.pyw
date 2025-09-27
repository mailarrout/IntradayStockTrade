import sys
import logging
import os
from datetime import datetime
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
import pytz

# ===== CONFIGURATION FLAG =====
DEBUG_MODE = True
# ==============================

# Configure logging with IST timezone FIRST
class ISTFormatter(logging.Formatter):
    """Custom formatter that converts timestamps to IST"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, pytz.timezone('Asia/Kolkata'))
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S IST')

current_date = datetime.now().strftime('%Y-%m-%d')
log_file = os.path.join('logs', f'{current_date}_app.log')
os.makedirs('logs', exist_ok=True)

# Create formatter with IST time
ist_formatter = ISTFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if DEBUG_MODE:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a',
        encoding='utf-8'
    )
    log_level_name = "DEBUG"
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a',
        encoding='utf-8'
    )
    log_level_name = "INFO"

# Apply IST formatter to all handlers
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.setFormatter(ist_formatter)

# ALWAYS keep console output for real-time monitoring
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ist_formatter)  # Apply IST format to console too
logging.getLogger().addHandler(console_handler)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info(f"Application starting - Logging to {log_file} (Mode: {log_level_name})")


# Application modules
from stock_loader import start_scheduled_loading, get_stock_name
from client_loader import load_clients
from chart_loader import ChartLoader
from position_loader import PositionLoader
from api_helper import ShoonyaApiPy
from api_status import ApiStatus

class QTextEditHandler(logging.Handler):
    """Custom logging handler that sends messages to QTextEdit"""
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        # ✅ USE IST FORMatter for UI logs
        self.setFormatter(ISTFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        
    def emit(self, record):
        msg = self.format(record)
        self.text_edit.appendPlainText(msg)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        # Initialize logger for this class
        self.logger = logging.getLogger(__name__)
        
        try:
            ist = pytz.timezone("Asia/Kolkata")
            current_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')
            self.logger.info(f"UI Loader starting up at {current_time}")
            
            uic.loadUi("IntradayDashboard.ui", self)
            
            # Setup UI logging
            self.setup_ui_logging()
            
            # Connect buttons
            self.LoadClientButton.clicked.connect(lambda: load_clients(self))
            self.GetStockNameButton.clicked.connect(lambda: get_stock_name(self))
            self.StartMonitoringButton.clicked.connect(lambda: self.logger.info("Start Monitoring button clicked"))
            
            # Initialize clients list
            self.clients = []
            
            # Create API instance
            self.api_client = ShoonyaApiPy()
            
            # Initialize ChartLoader with API client
            self.chart_loader = ChartLoader(self.HLOCGraphView, self)
            
            # Connect stock table click event
            self.StockNameTableView.clicked.connect(lambda index: self.chart_loader.on_stock_clicked(index, self))
            
            # Connect Load Graph button
            self.LoadGraphButton.clicked.connect(self.chart_loader.on_load_graph_clicked)
            
            # Auto-load clients and then get stock names after 2 seconds
            load_clients(self, auto_load=True)
            QTimer.singleShot(2000, lambda: get_stock_name(self))
            
            # Initialize position loader after clients are loaded
            QTimer.singleShot(3000, lambda: setattr(self, 'position_loader', PositionLoader(self, self)))
            
            start_scheduled_loading(self)

            self.api_status = ApiStatus(self)
            self.api_status.start()
            
            self.logger.info("UI initialized successfully")
            
        except Exception as e:
            self.logger.error(f"UI initialization failed: {str(e)}")
            raise

    def setup_ui_logging(self):
        """Setup logging to UI TextEdit"""
        # Create custom handler for UI with IST formatting
        ui_handler = QTextEditHandler(self.LogTextEdit)
        
        # ✅ Set level based on DEBUG_MODE
        if DEBUG_MODE:
            ui_handler.setLevel(logging.DEBUG)
        else:
            ui_handler.setLevel(logging.INFO)
        
        # ✅ Use the same IST formatter as console
        ui_handler.setFormatter(ISTFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add to root logger
        logging.getLogger().addHandler(ui_handler)
        
        self.logger.info("UI logging initialized successfully with IST timezone")

if __name__ == "__main__":
    try:
        ist = pytz.timezone("Asia/Kolkata")
        logger.info(f"Application starting at {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')}")
        
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow()
        window.show()
        
        result = app.exec_()
        logger.info("Application exited normally")
        sys.exit(result)
        
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}")
        sys.exit(1)