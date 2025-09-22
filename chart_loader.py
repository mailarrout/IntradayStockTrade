import logging
import mplfinance as mpf
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog, QVBoxLayout
from PyQt5.QtCore import Qt
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates
import mplcursors

# Get logger for this module
logger = logging.getLogger(__name__)

class ChartLoader:
    def __init__(self, graph_view, client_manager=None):
        self.graph_view = graph_view
        self.client_manager = client_manager
        logger.info("ChartLoader initialized")
    
    def on_stock_clicked(self, index, ui_reference):
        """Handle stock click event - plot latest historical data"""
        try:
            logger.info("Stock table clicked event triggered")
            
            # Get the model and data
            model = index.model()
            if not model:
                logger.warning("No model found in clicked index")
                return
            
            # Get the symbol from the first column (assuming Symbol is column 0)
            symbol = model.data(model.index(index.row(), 2), Qt.DisplayRole)
            
            if not symbol:
                logger.warning("No symbol found in clicked row")
                return
            
            logger.info(f"Stock clicked: {symbol}")
            
            # Find latest historical file
            latest_file = self.find_latest_historical_file(symbol)
            if not latest_file:
                error_msg = f"No historical data found for {symbol}"
                logger.error(error_msg)
                QMessageBox.warning(ui_reference, "Data Not Found", error_msg)
                return
            
            logger.info(f"Loading historical data from: {latest_file}")
            
            # Load historical data
            df = self.load_historical_data(latest_file)
            if df is None or df.empty:
                error_msg = f"Failed to load data for {symbol}"
                logger.error(error_msg)
                QMessageBox.warning(ui_reference, "Load Error", error_msg)
                return
            
            logger.info(f"Loaded {len(df)} records for {symbol}")
            
            # Get trade data for this symbol
            trade_data = self.get_trade_data(symbol)
            logger.info(f"Retrieved {len(trade_data)} trade records for {symbol}")
            
            # Plot the data
            success = self.plot_candlestick_mplfinance(df, symbol, trade_data)
            
            if success:
                logger.info(f"Candlestick chart displayed for {symbol}")
            else:
                logger.error(f"Failed to plot chart for {symbol}")
                QMessageBox.warning(ui_reference, "Chart Error", f"Failed to plot chart for {symbol}")
            
        except Exception as e:
            error_msg = f"Error processing stock click: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(ui_reference, "Error", error_msg)
    
    def get_trade_data(self, symbol):
        """Get trade data for a specific symbol from the client manager"""
        if not self.client_manager or not self.client_manager.clients:
            logger.warning("No clients available to fetch trade data")
            return []
        
        try:
            logger.info(f"Fetching trade data for symbol: {symbol}")
            
            # Use the first client to get trade book
            client_name, client_id, primary_client = self.client_manager.clients[0]
            logger.debug(f"Using client: {client_name} (ID: {client_id})")
            
            # Get trade book
            trade_book = primary_client.get_trade_book()
            
            if trade_book and isinstance(trade_book, list):
                # Filter trades for the specific symbol
                symbol_trades = [trade for trade in trade_book if trade.get('tsym', '').startswith(symbol)]
                logger.info(f"Found {len(symbol_trades)} trades for {symbol}")
                return symbol_trades
            else:
                logger.warning("No trade data returned from API or empty trade book")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching trade data: {str(e)}")
            return []
    
    def on_load_graph_clicked(self):
        """Handle Load Graph button click"""
        try:
            logger.info("Load Graph button clicked")
            
            # Open file dialog to select CSV file
            file_path, _ = QFileDialog.getOpenFileName(
                self.graph_view,
                "Select Candlestick Data File",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_path:
                logger.info("User cancelled file selection")
                return
            
            logger.info(f"Selected file: {file_path}")
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Loaded CSV with {len(df)} rows")
            
            # Validate the required columns (using ACTUAL CSV structure)
            required_columns = ['time', 'into', 'inth', 'intl', 'intc', 'intv']
            if not all(col in df.columns for col in required_columns):
                error_msg = f"Missing columns. Required: {required_columns}, Found: {list(df.columns)}"
                logger.error(error_msg)
                QMessageBox.warning(
                    self.graph_view,
                    "Invalid File Format",
                    "The selected file does not contain the required columns: "
                    "time, into, inth, intl, intc, intv"
                )
                return
            
            # Extract symbol from filename (format: SYMBOL.csv in date folders)
            symbol = self.extract_symbol_from_filename(file_path)
            logger.info(f"Using symbol: {symbol}")
            
            # Get trade data for this symbol
            trade_data = self.get_trade_data(symbol)
            logger.info(f"Retrieved {len(trade_data)} trade records for {symbol}")
            
            # Plot the data
            success = self.plot_candlestick_mplfinance(df, symbol, trade_data)
            
            if success:
                logger.info(f"Chart for {symbol} plotted successfully")
                QMessageBox.information(
                    self.graph_view,
                    "Success",
                    f"Chart for {symbol} loaded successfully!"
                )
            else:
                logger.error("Failed to plot chart")
                QMessageBox.warning(
                    self.graph_view,
                    "Error",
                    "Failed to plot the chart. Please check the file format."
                )
                
        except Exception as e:
            error_msg = f"Error loading graph file: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(
                self.graph_view,
                "Error",
                f"An error occurred while loading the file:\n{str(e)}"
            )
    
    def find_latest_historical_file(self, symbol):
        """Find today's historical data file for the given symbol"""
        try:
            logger.info(f"Searching for today's historical file for symbol: {symbol}")
            
            historical_dir = Path("HistoricalData")
            if not historical_dir.exists():
                logger.error("HistoricalData directory not found")
                return None
            
            # Get today's date folder
            today_str = datetime.now().strftime("%Y-%m-%d")
            today_folder = historical_dir / today_str
            
            if not today_folder.exists():
                logger.error(f"Today's folder {today_str} not found")
                return None
            
            # Look for symbol file in today's folder
            file_path = today_folder / f"{symbol}.csv"
            if file_path.exists():
                logger.info(f"Today's file for {symbol}: {file_path}")
                return file_path
            else:
                logger.error(f"No historical file found for {symbol} in today's folder")
                return None
                
        except Exception as e:
            logger.error(f"Error finding historical file for {symbol}: {str(e)}")
            return None

    def load_historical_data(self, file_path):
        """Load historical data from CSV file"""
        try:
            logger.info(f"Loading historical data from: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded historical data with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            return None

    def extract_symbol_from_filename(self, file_path):
        """Extract symbol from filename - for files named SYMBOL.csv in date folders"""
        try:
            path_obj = Path(file_path)
            # The symbol is just the filename without extension (e.g., "RELIANCE" from "RELIANCE.csv")
            symbol = path_obj.stem.upper()
            logger.debug(f"Extracted symbol '{symbol}' from filename: {file_path}")
            return symbol
            
        except Exception as e:
            logger.warning(f"Error extracting symbol from filename: {e}")
            return "UNKNOWN"

    def plot_candlestick_mplfinance(self, df, symbol, trade_data=None):
        """Plot candlestick chart using mplfinance with trade markers aligned to candles,
        and show hover tooltips for both candles and volume bars.
        """
        try:
            logger.info("Starting to plot candlestick chart (with tooltips)")

            # Clear existing widgets from graph view
            for widget in self.graph_view.findChildren(FigureCanvas):
                widget.deleteLater()
            logger.debug("Cleared existing graph widgets")

            # Convert time to datetime (day-first to match CSV like 08-09-2025)
            df['time'] = pd.to_datetime(
                df['time'],
                dayfirst=True,
                errors='coerce',
                format='%d-%m-%Y %H:%M:%S'
            )
            df = df.sort_values('time')
            logger.debug("Converted and sorted time column")

            # Rename columns for mplfinance
            plot_df = df.rename(columns={
                "time": "Date",
                "into": "Open",
                "inth": "High",
                "intl": "Low",
                "intc": "Close",
                "intv": "Volume"
            })

            # set index
            plot_df.set_index("Date", inplace=True)
            logger.debug("Renamed columns and set Date as index")

            logger.info(f"Plotting data with shape: {plot_df.shape}")

            # Prepare trade markers
            buy_markers = pd.Series(data=np.nan, index=plot_df.index, dtype=float)
            sell_markers = pd.Series(data=np.nan, index=plot_df.index, dtype=float)

            if trade_data and len(trade_data) > 0:
                logger.info(f"Preparing {len(trade_data)} trade markers")
                for trade in trade_data:
                    try:
                        time_str = trade.get("exch_tm") or trade.get("fltm") or trade.get("norentm")
                        if not time_str or time_str == "01-01-1980 00:00:00":
                            logger.debug("Skipping trade with invalid time")
                            continue

                        trade_time = pd.to_datetime(time_str, dayfirst=True, errors='coerce',
                                                    format='%d-%m-%Y %H:%M:%S')
                        if pd.isna(trade_time):
                            trade_time = pd.to_datetime(time_str, dayfirst=True, errors='coerce')
                        if pd.isna(trade_time):
                            logger.debug(f"Could not parse trade time: {time_str}")
                            continue

                        diffs = (plot_df.index - trade_time).total_seconds()
                        if len(diffs) == 0:
                            logger.debug("No data points for trade time comparison")
                            continue
                        nearest_pos = int(np.abs(diffs).argmin())
                        candle = plot_df.iloc[nearest_pos]

                        candle_high = float(candle["High"])
                        candle_low = float(candle["Low"])
                        candle_range = max(candle_high - candle_low, 0.0001)
                        offset = max(candle_range * 0.02, 0.01)

                        ttype = str(trade.get("trantype", "")).upper()
                        if ttype == "B":
                            buy_markers.iloc[nearest_pos] = candle_low - offset
                        elif ttype == "S":
                            sell_markers.iloc[nearest_pos] = candle_high + offset

                    except Exception as exc:
                        logger.error(f"Failed mapping trade -> candle: {exc} | trade={trade}")
                        continue

            # NEW: Prepare lowest volume markers
            low_vol_markers = pd.Series(data=np.nan, index=plot_df.index, dtype=float)
            low_vol_colors = pd.Series(data=None, index=plot_df.index, dtype=object)
            
            if len(plot_df) >= 4:
                try:
                    # Get the first 3 candles of the day
                    first_three = plot_df.iloc[:3]
                    lowest_volume_first_three = first_three['Volume'].min()
                    logger.debug(f"Lowest volume in first 3 candles: {lowest_volume_first_three}")

                    # Find candles after the first 3 with volume lower than the first 3's minimum
                    remaining_df = plot_df.iloc[3:]
                    low_volume_candles = remaining_df[remaining_df['Volume'] < lowest_volume_first_three]
                    
                    if not low_volume_candles.empty:
                        # Find the first green (close > open) and first red (close < open) candle
                        first_green_low_vol = None
                        first_red_low_vol = None
                        
                        for idx, row in low_volume_candles.iterrows():
                            if first_green_low_vol is None and row['Close'] > row['Open']:
                                first_green_low_vol = (idx, row)
                            if first_red_low_vol is None and row['Close'] < row['Open']:
                                first_red_low_vol = (idx, row)
                            
                            # Break early if we found both
                            if first_green_low_vol and first_red_low_vol:
                                break

                        # Add markers to the series
                        if first_green_low_vol:
                            idx, row = first_green_low_vol
                            # Place red triangle (downward) above the high
                            low_vol_markers.loc[idx] = float(row['High']) + (float(row['High']) - float(row['Low'])) * 0.1
                            low_vol_colors.loc[idx] = 'red'
                            logger.debug(f"Added red triangle marker for low volume green candle at {idx}")

                        if first_red_low_vol:
                            idx, row = first_red_low_vol
                            # Place green triangle (upward) below the low
                            low_vol_markers.loc[idx] = float(row['Low']) - (float(row['High']) - float(row['Low'])) * 0.1
                            low_vol_colors.loc[idx] = 'green'
                            logger.debug(f"Added green triangle marker for low volume red candle at {idx}")
                            
                except Exception as e:
                    logger.error(f"Error preparing low volume markers: {e}")

            # Build addplots list
            addplots = []
            if not buy_markers.isna().all():
                addplots.append(
                    mpf.make_addplot(buy_markers, type='scatter', marker='^',
                                    markersize=50, color='white', secondary_y=False)
                )
                logger.debug("Added buy markers to plot")
            if not sell_markers.isna().all():
                addplots.append(
                    mpf.make_addplot(sell_markers, type='scatter', marker='v',
                                    markersize=50, color='white', secondary_y=False)
                )
                logger.debug("Added sell markers to plot")
                
            # NEW: Add low volume markers to addplots
            if not low_vol_markers.isna().all():
                # Get the indices where we have markers
                marker_indices = low_vol_markers[low_vol_markers.notna()].index
                for idx in marker_indices:
                    marker_color = low_vol_colors.loc[idx]
                    marker_series = pd.Series(data=np.nan, index=plot_df.index, dtype=float)
                    marker_series.loc[idx] = low_vol_markers.loc[idx]
                    
                    # Choose marker based on color
                    marker_shape = 'v' if marker_color == 'red' else '^'
                    
                    addplots.append(
                        mpf.make_addplot(marker_series, type='scatter', marker=marker_shape,
                                        markersize=100, color=marker_color, secondary_y=False)
                    )
                logger.debug("Added low volume markers to plot")

            # Custom style
            mc = mpf.make_marketcolors(
                up='#00ff00', down='#ff0000',
                edge='inherit', wick='inherit',
                volume='in', ohlc='i'
            )
            s = mpf.make_mpf_style(
                marketcolors=mc,
                facecolor='#002B36',
                edgecolor='#586e75',
                figcolor='#002B36',
                gridcolor='#073642',
                gridstyle='--',
                y_on_right=False
            )
            logger.debug("Created custom market style")

            # Plot with mplfinance
            fig, axes = mpf.plot(
                plot_df[['Open', 'High', 'Low', 'Close', 'Volume']],
                type="candle",
                style=s,
                title=f"{symbol}",
                ylabel="Price",
                volume=True,
                returnfig=True,
                show_nontrading=False,
                addplot=addplots,
                figsize=(12, 8)
            )
            logger.debug("Created mplfinance plot")

            # --- FIX 1: restore white axis labels/ticks ---
            axes[0].tick_params(axis='x', colors='white')
            axes[0].tick_params(axis='y', colors='white')
            axes[0].yaxis.label.set_color('white')
            if fig._suptitle:
                fig._suptitle.set_color('white')
            if len(axes) > 2:
                axes[2].tick_params(axis='x', colors='white')
                axes[2].tick_params(axis='y', colors='white')
                axes[2].yaxis.label.set_color('white')
            logger.debug("Applied axis styling")

            # --- FIX 2: add hover tooltips on price + volume ---
            cursor = mplcursors.cursor(
                [*axes[0].get_children(), *axes[2].get_children()],
                hover=True
            )
            logger.debug("Created hover cursor")

            # Keep a reference list of extra arrows
            self._volume_arrows = []

            @cursor.connect("add")
            def on_hover(sel):
                if hasattr(sel.target, "__len__") and len(sel.target) >= 1:
                    x = int(round(sel.target[0]))
                    if 0 <= x < len(plot_df):
                        row = plot_df.iloc[x]

                        # Tooltip text
                        sel.annotation.set(
                            text=(
                                f"Date: {row.name.strftime('%Y-%m-%d %H:%M')}\n"
                                f"O: {row['Open']:.2f}  H: {row['High']:.2f}\n"
                                f"L: {row['Low']:.2f}  C: {row['Close']:.2f}\n"
                                f"Vol: {row['Volume']:,}"
                            ),
                            position=(0, 20),
                            anncoords="offset points"
                        )

                        # Style tooltip
                        sel.annotation.set_color("white")
                        sel.annotation.get_bbox_patch().set(fc="black", alpha=0.7, ec="white", lw=1)
                        sel.annotation.arrow_patch.set_arrowstyle("->")
                        sel.annotation.arrow_patch.set_color("white")

                        # --- Clear old arrows ---
                        for arrow in self._volume_arrows:
                            try:
                                arrow.remove()
                            except Exception:
                                pass
                        self._volume_arrows.clear()

                        # --- Add fresh arrow to volume bar ---
                        vol_ax = sel.annotation.axes.figure.axes[2]  # volume subplot
                        vol_y = row["Volume"]

                        arrow = vol_ax.annotate(
                            "",
                            xy=(x, vol_y), xycoords=("data", "data"),
                            xytext=(x, vol_y + max(plot_df["Volume"]) * 0.1),
                            textcoords="data",
                            arrowprops=dict(arrowstyle="->", color="white", lw=1),
                            annotation_clip=False
                        )
                        self._volume_arrows.append(arrow)

            # Opening range lines
            self.add_opening_range_lines(axes, plot_df)
            logger.debug("Added opening range lines")

            # Attach canvas
            canvas = FigureCanvas(fig)
            layout = self.graph_view.layout()
            if layout:
                while layout.count():
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
            else:
                layout = QVBoxLayout()
                self.graph_view.setLayout(layout)
            layout.addWidget(canvas)
            canvas.draw()
            logger.debug("Attached canvas to layout")

            logger.info("Chart plotted successfully with hover tooltips")
            return True

        except Exception as e:
            error_msg = f"Error plotting with mplfinance: {str(e)}"
            logger.error(error_msg)
            return False

    def add_trade_markers(self, fig, axes, plot_df, trade_data):
        """Add buy/sell markers at the candle that matches trade time (exch_tm/fltm)."""
        try:
            logger.info("Adding trade markers to chart")
            
            price_axis = axes[0] if isinstance(axes, (list, np.ndarray)) else axes

            # Ensure datetime index
            if not isinstance(plot_df.index, pd.DatetimeIndex):
                plot_df.index = pd.to_datetime(plot_df.index)

            for trade in trade_data:
                try:
                    # Pick exch_tm if available, else fltm
                    trade_time_str = trade.get("exch_tm") or trade.get("fltm")
                    if not trade_time_str or trade_time_str == "01-01-1980 00:00:00":
                        logger.debug("Skipping trade with invalid time")
                        continue

                    trade_time = pd.to_datetime(trade_time_str, format="%d-%m-%Y %H:%M:%S")

                    # Find nearest candle
                    diffs = (plot_df.index - trade_time).total_seconds()
                    nearest_idx = abs(diffs).argmin()
                    candle_time = plot_df.index[nearest_idx]
                    candle = plot_df.iloc[nearest_idx]

                    # Offset (small gap so arrow not touching candle)
                    candle_range = float(candle["High"]) - float(candle["Low"])
                    offset = max(candle_range * 0.02, 0.05)

                    # Convert x to mpl date
                    x_val = mdates.date2num(candle_time.to_pydatetime())

                    if trade.get("trantype") == "B":
                        # BUY → below Low
                        y_val = float(candle["Low"]) - offset
                        price_axis.scatter([x_val], [y_val], marker="^", s=80,
                                           facecolors="green", edgecolors="black",
                                           linewidths=0.7, zorder=10)
                        logger.debug(f"BUY marker at {candle_time} (Low={candle['Low']})")

                    elif trade.get("trantype") == "S":
                        # SELL → above High
                        y_val = float(candle["High"]) + offset
                        price_axis.scatter([x_val], [y_val], marker="v", s=80,
                                           facecolors="red", edgecolors="black",
                                           linewidths=0.7, zorder=10)
                        logger.debug(f"SELL marker at {candle_time} (High={candle['High']})")

                except Exception as e:
                    logger.error(f"Trade marker error: {e} | Trade: {trade}")
                    continue

        except Exception as e:
            logger.error(f"add_trade_markers failed: {e}")

    def add_opening_range_lines(self, axes, df):
        """Draw horizontal lines at 9:15 high/low across the whole session."""
        try:
            logger.debug("Adding opening range lines (9:15 high/low)")
            
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("DataFrame index is not DatetimeIndex, skipping opening range lines")
                return

            # Find the 9:15 AM candle
            candle = df[df.index.strftime("%H:%M") == "09:15"]
            if candle.empty:
                logger.warning("No 9:15 AM candle found in data")
                return

            high = float(candle["High"].iloc[0])
            low = float(candle["Low"].iloc[0])
            logger.debug(f"9:15 levels - High: {high}, Low: {low}")

            price_ax = axes[0] if isinstance(axes, (list, np.ndarray)) else axes

            # Use x-axis limits from chart itself
            xmin, xmax = price_ax.get_xlim()

            # Plot dashed horizontal lines across the chart
            price_ax.hlines(high, xmin, xmax, colors="#cc7d7d", linestyles="--", linewidth=1)
            price_ax.hlines(low, xmin, xmax, colors="#cc7d7d", linestyles="--", linewidth=1)
            logger.debug("Added 9:15 high/low lines to chart")

        except Exception as e:
            logger.error(f"Failed to add 9:15 lines: {e}")


