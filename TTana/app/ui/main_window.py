"""
Main Application Window
PyQt6-based main window with tabbed interface
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QMenuBar, QMenu,
    QStatusBar, QLabel, QPushButton, QFileDialog,
    QSplitter, QFrame, QToolBar, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QIcon, QFont, QAction, QActionGroup

from app.core.config import ConfigModel


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, config: ConfigModel, parent=None):
        super().__init__(parent)
        
        self.config = config
        self.current_session = None
        
        self._init_ui()
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_status_bar()
        self._connect_signals()
        
        self.setWindowTitle("TTana - AI Table Tennis Coach")
        self.setMinimumSize(1280, 720)
        self.resize(1400, 900)
    
    def _init_ui(self):
        """Initialize user interface."""
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabsClosable(False)
        
        # Add tabs (placeholders for now)
        self._setup_tabs()
        
        main_layout.addWidget(self.tab_widget)
    
    def _setup_tabs(self):
        """Setup application tabs."""
        
        # Dashboard Tab
        from app.ui.dashboard import DashboardWidget
        self.dashboard_tab = DashboardWidget(self.config)
        self.tab_widget.addTab(self.dashboard_tab, "Dashboard")
        
        # Video Comparison Tab
        from app.ui.comparison import ComparisonWidget
        self.comparison_tab = ComparisonWidget(self.config)
        self.tab_widget.addTab(self.comparison_tab, "Video Comparison")
        
        # Biomechanics Tab
        from app.ui.biomechanics import BiomechanicsWidget
        self.biomechanics_tab = BiomechanicsWidget(self.config)
        self.tab_widget.addTab(self.biomechanics_tab, "Biomechanics")
        
        # Coach Report Tab
        from app.ui.coach import CoachWidget
        self.coach_tab = CoachWidget(self.config)
        self.tab_widget.addTab(self.coach_tab, "Coach Report")
        
        # Settings Tab (optional)
        # self.settings_tab = SettingsWidget(self.config)
        # self.tab_widget.addTab(self.settings_tab, "Settings")
    
    def _create_menu_bar(self):
        """Create menu bar."""
        
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        # New Session
        new_session_action = QAction("&New Session", self)
        new_session_action.setShortcut("Ctrl+N")
        new_session_action.triggered.connect(self._new_session)
        file_menu.addAction(new_session_action)
        
        # Open Session
        open_session_action = QAction("&Open Session", self)
        open_session_action.setShortcut("Ctrl+O")
        open_session_action.triggered.connect(self._open_session)
        file_menu.addAction(open_session_action)
        
        file_menu.addSeparator()
        
        # Load User Video
        load_user_video_action = QAction("Load &User Video", self)
        load_user_video_action.setShortcut("Ctrl+U")
        load_user_video_action.triggered.connect(self._load_user_video)
        file_menu.addAction(load_user_video_action)
        
        # Load Reference Video
        load_ref_video_action = QAction("Load &Reference Video", self)
        load_ref_video_action.setShortcut("Ctrl+R")
        load_ref_video_action.triggered.connect(self._load_reference_video)
        file_menu.addAction(load_ref_video_action)
        
        file_menu.addSeparator()
        
        # Save Session
        save_session_action = QAction("&Save Session", self)
        save_session_action.setShortcut("Ctrl+S")
        save_session_action.triggered.connect(self._save_session)
        file_menu.addAction(save_session_action)
        
        # Export Results
        export_action = QAction("&Export Results", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis Menu
        analysis_menu = menubar.addMenu("&Analysis")
        
        # Start Analysis
        start_analysis_action = QAction("&Start Analysis", self)
        start_analysis_action.setShortcut("F5")
        start_analysis_action.triggered.connect(self._start_analysis)
        analysis_menu.addAction(start_analysis_action)
        
        # Stop Analysis
        stop_analysis_action = QAction("&Stop Analysis", self)
        stop_analysis_action.setShortcut("F6")
        stop_analysis_action.triggered.connect(self._stop_analysis)
        analysis_menu.addAction(stop_analysis_action)
        
        analysis_menu.addSeparator()
        
        # Clear Cache
        clear_cache_action = QAction("&Clear Cache", self)
        clear_cache_action.triggered.connect(self._clear_cache)
        analysis_menu.addAction(clear_cache_action)
        
        # LLM Menu
        llm_menu = menubar.addMenu("&LLM")
        
        # Generate Report
        generate_report_action = QAction("&Generate Coach Report", self)
        generate_report_action.setShortcut("F7")
        generate_report_action.triggered.connect(self._generate_report)
        llm_menu.addAction(generate_report_action)
        
        # Chat with Coach
        chat_action = QAction("&Chat with Coach", self)
        chat_action.setShortcut("F8")
        chat_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(3))
        llm_menu.addAction(chat_action)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        
        # Documentation
        docs_action = QAction("&Documentation", self)
        docs_action.setShortcut("F1")
        docs_action.triggered.connect(self._open_docs)
        help_menu.addAction(docs_action)
        
        # About
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_tool_bar(self):
        """Create toolbar."""
        
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # New Session
        new_action = QAction("📄", self)
        new_action.setToolTip("New Session")
        new_action.triggered.connect(self._new_session)
        toolbar.addAction(new_action)
        
        # Open
        open_action = QAction("📂", self)
        open_action.setToolTip("Open Session")
        open_action.triggered.connect(self._open_session)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # User Video
        user_video_action = QAction("🎬", self)
        user_video_action.setToolTip("Load User Video")
        user_video_action.triggered.connect(self._load_user_video)
        toolbar.addAction(user_video_action)
        
        # Reference Video
        ref_video_action = QAction("🏆", self)
        ref_video_action.setToolTip("Load Reference Video")
        ref_video_action.triggered.connect(self._load_reference_video)
        toolbar.addAction(ref_video_action)
        
        toolbar.addSeparator()
        
        # Analyze
        analyze_action = QAction("▶️", self)
        analyze_action.setToolTip("Start Analysis")
        analyze_action.triggered.connect(self._start_analysis)
        toolbar.addAction(analyze_action)
        
        # Stop
        stop_action = QAction("⏹️", self)
        stop_action.setToolTip("Stop Analysis")
        stop_action.triggered.connect(self._stop_analysis)
        toolbar.addAction(stop_action)
    
    def _create_status_bar(self):
        """Create status bar."""
        
        statusbar = QStatusBar()
        self.setStatusBar(statusbar)
        
        # Status label
        self.status_label = QLabel("Ready")
        statusbar.addPermanentWidget(self.status_label)
        
        # Progress bar (placeholder)
        # self.progress_bar = QProgressBar()
        # self.progress_bar.setMaximumWidth(200)
        # statusbar.addPermanentWidget(self.progress_bar)
    
    def _connect_signals(self):
        """Connect signals and slots."""
        # Placeholder for signal connections
        pass
    
    # Slot methods
    
    def _new_session(self):
        """Create new session."""
        self.status_label.setText("Creating new session...")
        # TODO: Implement session creation
    
    def _open_session(self):
        """Open existing session."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Session",
            "sessions/",
            "Session Files (*.json);;All Files (*)"
        )
        if file_path:
            self.status_label.setText(f"Opening session: {file_path}")
            # TODO: Implement session loading
    
    def _load_user_video(self):
        """Load user video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select User Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.status_label.setText(f"Loaded user video: {Path(file_path).name}")
            self.dashboard_tab.load_user_video(file_path)
    
    def _load_reference_video(self):
        """Load reference video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.status_label.setText(f"Loaded reference video: {Path(file_path).name}")
            self.dashboard_tab.load_reference_video(file_path)
    
    def _save_session(self):
        """Save current session."""
        self.status_label.setText("Saving session...")
        # TODO: Implement session saving
    
    def _export_results(self):
        """Export analysis results."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "JSON Files (*.json);;PDF Files (*.pdf);;All Files (*)"
        )
        if file_path:
            self.status_label.setText(f"Exporting results to: {file_path}")
            # TODO: Implement export
    
    def _start_analysis(self):
        """Start analysis pipeline."""
        self.status_label.setText("Starting analysis...")
        # Switch to dashboard tab
        self.tab_widget.setCurrentIndex(0)
        # TODO: Implement analysis start
    
    def _stop_analysis(self):
        """Stop analysis pipeline."""
        self.status_label.setText("Stopping analysis...")
        # TODO: Implement analysis stop
    
    def _clear_cache(self):
        """Clear analysis cache."""
        self.status_label.setText("Clearing cache...")
        # TODO: Implement cache clearing
    
    def _generate_report(self):
        """Generate coach report."""
        self.status_label.setText("Generating coach report...")
        # Switch to coach tab
        self.tab_widget.setCurrentIndex(3)
        # TODO: Implement report generation
    
    def _open_docs(self):
        """Open documentation."""
        # TODO: Open documentation in browser
        print("Opening documentation...")
    
    def _show_about(self):
        """Show about dialog."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About TTana",
            "<h2>TTana - AI Table Tennis Coach</h2>"
            "<p>Version 1.0.0</p>"
            "<p>Single-camera video analysis system for table tennis technique improvement.</p>"
            "<p>Powered by:</p>"
            "<ul>"
            "<li>RTMPose for pose estimation</li>"
            "<li>ByteTrack for tracking</li>"
            "<li>Dynamic Time Warping for comparison</li>"
            "<li>Qwen3.5-9B for AI coaching</li>"
            "</ul>"
            "<p>© 2024 TTana Project</p>"
        )
    
    def closeEvent(self, event):
        """Handle application close."""
        # TODO: Save state and cleanup
        event.accept()
