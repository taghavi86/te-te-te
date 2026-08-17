"""
TTana - AI Table Tennis Coach
Main Application Entry Point
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from app.ui.main_window import MainWindow
from app.core.config import Config


def main():
    """Main application entry point."""
    
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("TTana")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("TTana")
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Load configuration
    config = Config.load()
    
    # Apply theme
    if config.ui.theme == "dark":
        app.setStyle("Fusion")
        # TODO: Add dark palette
    
    # Create and show main window
    window = MainWindow(config)
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
