"""
Dashboard Widget - Main control panel for video loading and analysis
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGroupBox, QGridLayout, QProgressBar, QFileDialog,
    QSplitter, QTextEdit, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class DashboardWidget(QWidget):
    """Dashboard widget for main application control."""
    
    video_loaded = pyqtSignal(str, str)  # user_video_path, ref_video_path
    analysis_started = pyqtSignal()
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        
        self.config = config
        self.user_video_path = None
        self.reference_video_path = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize dashboard UI."""
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("🏓 TTana - AI Table Tennis Coach")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Single-Camera Video Analysis System")
        subtitle_label.setFont(QFont("Segoe UI", 11))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #888;")
        main_layout.addWidget(subtitle_label)
        
        # Video Selection Section
        video_group = QGroupBox("📹 Video Selection")
        video_layout = QGridLayout(video_group)
        video_layout.setSpacing(15)
        
        # User Video
        user_video_label = QLabel("🎬 Your Video:")
        user_video_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        video_layout.addWidget(user_video_label, 0, 0)
        
        self.user_video_btn = QPushButton("Select User Video")
        self.user_video_btn.setMinimumHeight(40)
        self.user_video_btn.clicked.connect(self._select_user_video)
        video_layout.addWidget(self.user_video_btn, 0, 1)
        
        self.user_video_path_label = QLabel("No video selected")
        self.user_video_path_label.setStyleSheet("color: #666; font-style: italic;")
        video_layout.addWidget(self.user_video_path_label, 0, 2)
        
        # Reference Video
        ref_video_label = QLabel("🏆 Professional Reference:")
        ref_video_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        video_layout.addWidget(ref_video_label, 1, 0)
        
        self.ref_video_btn = QPushButton("Select Reference Video")
        self.ref_video_btn.setMinimumHeight(40)
        self.ref_video_btn.clicked.connect(self._select_reference_video)
        video_layout.addWidget(self.ref_video_btn, 1, 1)
        
        self.ref_video_path_label = QLabel("No video selected")
        self.ref_video_path_label.setStyleSheet("color: #666; font-style: italic;")
        video_layout.addWidget(self.ref_video_path_label, 1, 2)
        
        main_layout.addWidget(video_group)
        
        # Analysis Status Section
        status_group = QGroupBox("📊 Analysis Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setPlaceholderText("Analysis status will appear here...")
        status_layout.addWidget(self.status_text)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% - Ready")
        status_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(status_group)
        
        # Action Buttons
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(20)
        
        # Analyze Button
        self.analyze_btn = QPushButton("▶️ Start Analysis")
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.analyze_btn.clicked.connect(self._start_analysis)
        self.analyze_btn.setEnabled(False)
        actions_layout.addWidget(self.analyze_btn)
        
        # Stop Button
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.stop_btn.clicked.connect(self._stop_analysis)
        self.stop_btn.setEnabled(False)
        actions_layout.addWidget(self.stop_btn)
        
        main_layout.addLayout(actions_layout)
        
        # Quick Info Section
        info_group = QGroupBox("ℹ️ Quick Guide")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(200)
        info_text.setHtml("""
        <h3>How to Use:</h3>
        <ol>
            <li><b>Select Your Video:</b> Load a video of your table tennis stroke</li>
            <li><b>Select Reference Video:</b> Load a professional player's video for comparison</li>
            <li><b>Start Analysis:</b> Click the analyze button to begin processing</li>
            <li><b>View Results:</b> Check the Video Comparison, Biomechanics, and Coach Report tabs</li>
            <li><b>Chat with Coach:</b> Ask questions about your technique in the Coach Report tab</li>
        </ol>
        
        <h3>Tips:</h3>
        <ul>
            <li>Use videos with good lighting and clear player visibility</li>
            <li>Side-view angles work best for stroke analysis</li>
            <li>Ensure the player is visible throughout the stroke</li>
            <li>30 FPS or higher is recommended</li>
        </ul>
        """)
        info_layout.addWidget(info_text)
        
        main_layout.addWidget(info_group)
        
        # Spacer
        main_layout.addStretch()
    
    def _select_user_video(self):
        """Select user video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select User Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.user_video_path = file_path
            self.user_video_path_label.setText(file_path)
            self.user_video_path_label.setStyleSheet("color: #4CAF50;")
            self._update_analyze_button()
            self.video_loaded.emit(file_path, self.reference_video_path)
    
    def _select_reference_video(self):
        """Select reference video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.reference_video_path = file_path
            self.ref_video_path_label.setText(file_path)
            self.ref_video_path_label.setStyleSheet("color: #4CAF50;")
            self._update_analyze_button()
            self.video_loaded.emit(self.user_video_path, file_path)
    
    def _update_analyze_button(self):
        """Update analyze button state based on video selection."""
        both_selected = self.user_video_path and self.reference_video_path
        self.analyze_btn.setEnabled(both_selected)
        
        if both_selected:
            self.status_text.append("✓ Both videos loaded. Ready to analyze.")
    
    def _start_analysis(self):
        """Start analysis pipeline."""
        if not (self.user_video_path and self.reference_video_path):
            return
        
        self.analyze_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% - Initializing...")
        
        self.status_text.append("\n" + "="*50)
        self.status_text.append("Starting analysis pipeline...")
        self.status_text.append(f"User Video: {self.user_video_path}")
        self.status_text.append(f"Reference Video: {self.reference_video_path}")
        self.status_text.append("="*50 + "\n")
        
        self.analysis_started.emit()
    
    def _stop_analysis(self):
        """Stop analysis pipeline."""
        self.stop_btn.setEnabled(False)
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setFormat("%p% - Stopped")
        self.status_text.append("\n⚠️ Analysis stopped by user.\n")
    
    def load_user_video(self, file_path: str):
        """Load user video from external call."""
        if file_path:
            self.user_video_path = file_path
            self.user_video_path_label.setText(file_path)
            self.user_video_path_label.setStyleSheet("color: #4CAF50;")
            self._update_analyze_button()
    
    def load_reference_video(self, file_path: str):
        """Load reference video from external call."""
        if file_path:
            self.reference_video_path = file_path
            self.ref_video_path_label.setText(file_path)
            self.ref_video_path_label.setStyleSheet("color: #4CAF50;")
            self._update_analyze_button()
    
    def update_progress(self, value: int, message: str = ""):
        """Update progress bar and status."""
        self.progress_bar.setValue(value)
        if message:
            self.progress_bar.setFormat(f"%p% - {message}")
            self.status_text.append(f"[{value}%] {message}")
    
    def add_status_message(self, message: str):
        """Add message to status text."""
        self.status_text.append(message)
