"""
Coach Widget - AI Coach report and chat interface
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox, QScrollArea, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal

from app.core.config import ConfigModel


class CoachWidget(QWidget):
    """AI Coach report and chat widget."""
    
    message_sent = pyqtSignal(str)
    
    def __init__(self, config: ConfigModel, parent=None):
        super().__init__(parent)
        
        self.config = config
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("🤖 AI Coach - Qwen3.5-9B")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title_label)
        
        # Report Section
        report_group = QGroupBox("📋 Coach Report")
        report_layout = QVBoxLayout(report_group)
        
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText(
            "Coach report will appear here after analysis.\n\n"
            "The report includes:\n"
            "- Summary of your technique\n"
            "- Primary issues identified\n"
            "- Root cause analysis\n"
            "- Evidence from video\n"
            "- Secondary issues\n"
            "- Your strengths\n"
            "- Recommended corrections\n"
            "- Training plan\n"
            "- Next session goals"
        )
        report_layout.addWidget(self.report_text)
        
        main_layout.addWidget(report_group)
        
        # Chat Section
        chat_group = QGroupBox("💬 Chat with Coach")
        chat_layout = QVBoxLayout(chat_group)
        
        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setPlaceholderText(
            "Ask questions about your technique...\n\n"
            "Examples:\n"
            "- What's wrong with my backswing?\n"
            "- How can I improve my timing?\n"
            "- Show me my strongest stroke\n"
            "- Why is my follow-through late?"
        )
        chat_layout.addWidget(self.chat_history)
        
        # Chat input
        input_layout = QHBoxLayout()
        
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(80)
        self.chat_input.setPlaceholderText("Type your question here...")
        input_layout.addWidget(self.chat_input)
        
        self.send_btn = QPushButton("Send ➤")
        self.send_btn.setMinimumWidth(100)
        self.send_btn.clicked.connect(self._send_message)
        input_layout.addWidget(self.send_btn)
        
        chat_layout.addLayout(input_layout)
        
        main_layout.addWidget(chat_group)
        
        # Generate Report Button
        self.generate_btn = QPushButton("📄 Generate New Report")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.generate_btn.clicked.connect(self._generate_report)
        main_layout.addWidget(self.generate_btn)
    
    def _send_message(self):
        """Send message to AI coach."""
        message = self.chat_input.toPlainText().strip()
        
        if not message:
            return
        
        # Add user message to chat
        self.chat_history.append(f"<b>You:</b> {message}")
        self.chat_history.append("")
        
        # Clear input
        self.chat_input.clear()
        
        # Emit signal for processing
        self.message_sent.emit(message)
    
    def _generate_report(self):
        """Generate new coach report."""
        self.report_text.setPlaceholderText("Generating report... Please wait.")
        # TODO: Implement report generation
    
    def add_coach_response(self, response: str):
        """Add coach response to chat."""
        self.chat_history.append(f"<b>Coach:</b> {response}")
        self.chat_history.append("")
    
    def set_report(self, report: str):
        """Set coach report text."""
        self.report_text.setText(report)
    
    def append_to_report(self, text: str):
        """Append text to report."""
        self.report_text.append(text)
