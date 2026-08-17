"""
Comparison Widget - Side-by-side video comparison with skeleton overlay
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from app.core.config import ConfigModel


class ComparisonWidget(QWidget):
    """Video comparison widget with skeleton overlay."""
    
    def __init__(self, config: ConfigModel, parent=None):
        super().__init__(parent)
        
        self.config = config
        
        layout = QVBoxLayout(self)
        
        label = QLabel("📹 Video Comparison View\n\n[Placeholder: Side-by-side video player with skeleton overlay will be implemented here]")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 14px; color: #666; padding: 50px;")
        
        layout.addWidget(label)
