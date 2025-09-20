import os
import pandas as pd
import pyotp
from datetime import datetime
from pytz import timezone
from PyQt5.QtWidgets import QMessageBox
import logging
from api_helper import ShoonyaApiPy

# Get logger for this module
logger = logging.getLogger(__name__)

# Timezone setup for IST
IST = timezone('Asia/Kolkata')

def load_clients(ui_reference, auto_load=False):
    """Load client credentials from file and initialize API connections
    auto_load: If True, this is an automatic load on startup (suppress some UI messages)
    """
    try:
        logger.info("Starting client loading process")
        
        # Check if file exists
        if not os.path.exists("ClientInfo.txt"):
            msg = "ClientInfo.txt not found"
            logger.error(msg)
            if not auto_load:  # Only show message box for manual loading
                QMessageBox.critical(ui_reference, "Error", msg)
            return False

        logger.info("ClientInfo.txt file found, reading contents")
        
        # Read the file
        df = pd.read_csv("ClientInfo.txt")
        if df.empty:
            msg = "ClientInfo.txt is empty"
            logger.warning(msg)
            if not auto_load:  # Only show message box for manual loading
                QMessageBox.warning(ui_reference, "Warning", msg)
            return False

        logger.info(f"Loaded {len(df)} client records from file")
        
        # Initialize clients list
        if not hasattr(ui_reference, 'clients'):
            ui_reference.clients = []
            logger.debug("Created new clients list in ui_reference")
        
        # Clear existing clients
        ui_reference.clients = []
        logger.debug("Cleared existing clients list")

        success_count = 0
        total_clients = len(df)
        logger.info(f"Processing {total_clients} clients")
        
        for index, row in df.iterrows():
            try:
                client_id = row["Client ID"]
                logger.info(f"Processing client {index + 1}/{total_clients}: {client_id}")
                
                login_time = datetime.now(IST)
                logger.debug(f"Generating 2FA token for client {client_id}")
                
                twoFA = pyotp.TOTP(row["token"]).now()
                logger.debug(f"2FA token generated for client {client_id}")

                logger.debug(f"Creating ShoonyaApiPy instance for client {client_id}")
                client = ShoonyaApiPy()  # Create new client instance
                
                logger.debug(f"Attempting login for client {client_id}")
                ret = client.login(
                    userid=row["Client ID"],
                    password=row["Password"],
                    twoFA=twoFA,
                    vendor_code=row["vc"],
                    api_secret=row["app_key"],
                    imei=row["imei"]
                )

                if ret and ret.get('stat') == 'Ok':
                    client_name = row.get("Client Name", row["Client ID"])
                    ui_reference.clients.append((client_name, row["Client ID"], client))
                    success_count += 1
                    logger.info(f"{client_name} logged in successfully at {login_time.strftime('%H:%M:%S IST')}")
                else:
                    error_msg = ret.get('emsg', 'Unknown error') if ret else 'No response'
                    logger.error(f"Login failed for client {client_id} at {login_time.strftime('%H:%M:%S IST')}: {error_msg}")
                    
            except KeyError as e:
                logger.error(f"Missing required field in ClientInfo.txt for client {index + 1}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing client {row['Client ID']}: {str(e)}")
        
        # Show result based on auto_load flag
        if success_count > 0:
            msg = f"Successfully loaded {success_count} out of {total_clients} client(s)"
            logger.info(msg)
            if not auto_load:  # Only show message box for manual loading
                QMessageBox.information(ui_reference, "Success", msg)
            return True
        else:
            msg = f"No clients were successfully loaded out of {total_clients} attempts"
            logger.error(msg)
            if not auto_load:  # Only show message box for manual loading
                QMessageBox.critical(ui_reference, "Error", msg)
            return False
            
    except pd.errors.EmptyDataError:
        error_msg = "ClientInfo.txt is empty or contains no valid data"
        logger.error(error_msg)
        if not auto_load:
            QMessageBox.critical(ui_reference, "Error", error_msg)
        return False
            
    except pd.errors.ParserError:
        error_msg = "ClientInfo.txt has invalid format or cannot be parsed as CSV"
        logger.error(error_msg)
        if not auto_load:
            QMessageBox.critical(ui_reference, "Error", error_msg)
        return False
            
    except Exception as e:
        error_msg = f"Unexpected error in load_clients: {str(e)}"
        logger.error(error_msg)
        if not auto_load:  # Only show message box for manual loading
            QMessageBox.critical(ui_reference, "Error", error_msg)
        return False