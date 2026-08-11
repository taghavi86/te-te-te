import sys
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QSlider, QWidget, 
                             QGroupBox, QGridLayout, QScrollArea, QFrame)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette
import pyqtgraph as pg

# --- تنظیمات اولیه مدل‌های هوش مصنوعی ---
# بارگذاری مدل‌ها (در اولین اجرا دانلود می‌شوند)
try:
    yolo_model = YOLO('yolov8n.pt')  # مدل سبک برای تشخیص سریع
except Exception as e:
    print(f"Error loading YOLO: {e}")
    yolo_model = None

mp_pose = mp.solutions.pose
pose_solution = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True)
mp_drawing = mp.solutions.drawing_utils

class VideoAnalysisThread(QThread):
    """رشته پردازش ویدئو برای جلوگیری از فریز شدن رابط کاربری"""
    frame_ready = pyqtSignal(np.ndarray, dict)
    analysis_ready = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, video_path, reference_data=None):
        super().__init__()
        self.video_path = video_path
        self.reference_data = reference_data
        self.is_running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        frame_count = 0
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            analysis_result = self.analyze_frame(frame)
            
            # ارسال فریم پردازش شده و داده‌های تحلیل
            self.frame_ready.emit(frame, analysis_result)
            if frame_count % 5 == 0: # ارسال داده‌های آماری هر 5 فریم برای کاهش تاخیر
                self.analysis_ready.emit(analysis_result)
            
            frame_count += 1
        
        cap.release()
        self.finished.emit()

    def analyze_frame(self, frame):
        results = {
            'angles': {},
            'objects': [],
            'feedback': [],
            'pose_landmarks': None
        }

        # 1. تشخیص اشیاء با YOLO (توپ، راکت، شخص)
        if yolo_model:
            yolo_results = yolo_model(frame, verbose=False)[0]
            for box in yolo_results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:
                    label = yolo_results.names[cls]
                    results['objects'].append({'label': label, 'conf': conf})

        # 2. تحلیل اسکلت بدن با MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_solution.process(rgb_frame)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            results['pose_landmarks'] = landmarks
            
            # محاسبه زوایای کلیدی (مثلاً زاویه آرنج برای فورهند)
            # نقاط: شانه (11), آرنج (13), مچ (15) برای دست راست
            if len(landmarks) > 15:
                angle_elbow = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                angle_shoulder = self.calculate_angle(landmarks[13], landmarks[11], landmarks[23]) # شانه به لگن
                
                results['angles']['elbow'] = angle_elbow
                results['angles']['shoulder'] = angle_shoulder

                # 3. منطق مربی هوشمند (مقایسه با الگو یا استانداردها)
                # فرض: زاویه آرنج ایده‌آل در ضربه بین 90 تا 110 درجه است
                if 90 <= angle_elbow <= 110:
                    results['feedback'].append("✅ تکنیک آرنج عالی است.")
                elif angle_elbow < 90:
                    results['feedback'].append("⚠️ آرنج خیلی خم شده است. دست را بازتر کنید.")
                else:
                    results['feedback'].append("⚠️ آرنج خیلی صاف است. کمی خمیده نگه دارید.")

                # مقایسه با ویدئوی مرجع (اگر موجود باشد)
                if self.reference_data and 'elbow' in self.reference_data:
                    ref_angle = self.reference_data['elbow']
                    diff = abs(angle_elbow - ref_angle)
                    if diff > 15:
                        results['feedback'].append(f"❌ تفاوت با پرو: {diff:.1f} درجه انحراف در آرنج.")
                    else:
                        results['feedback'].append("✅ شباهت زیاد به سبک حرفه‌ای.")

        return results

    def calculate_angle(self, a, b, c):
        """محاسبه زاویه بین سه نقطه"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def stop(self):
        self.is_running = False
        self.wait()

class TennisCoachApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎾 هوشمند تنیس پرو | AI Tennis Coach")
        self.setGeometry(100, 100, 1400, 900)
        
        # استایل دهی مدرن (Dark Theme)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QGroupBox { 
                border: 1px solid #333; 
                border-radius: 5px; 
                margin-top: 10px; 
                font-weight: bold;
                color: #00d2ff;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #005f9e; }
            QLabel { font-size: 14px; color: #e0e0e0; }
            QSlider::groove:horizontal { border: 1px solid #999; height: 8px; background: #333; }
            QSlider::handle:horizontal { background: #00d2ff; width: 18px; margin: -2px 0; border-radius: 9px; }
        """)

        self.current_video_path = None
        self.reference_video_path = None
        self.reference_data = None # ذخیره داده‌های ویدئوی حرفه‌ای
        self.worker = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- بخش چپ: ویدئو و کنترل‌ها ---
        left_panel = QVBoxLayout()
        
        # نمایشگر ویدئو
        self.video_label = QLabel("لطفاً ویدئو را بارگذاری کنید")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000; border: 2px solid #333;")
        left_panel.addWidget(self.video_label)

        # کنترل‌ها
        control_layout = QHBoxLayout()
        self.btn_load_user = QPushButton("📂 بارگذاری ویدئوی من")
        self.btn_load_pro = QPushButton("🏆 بارگذاری ویدئوی حرفه‌ای (الگو)")
        self.btn_start = QPushButton("▶ شروع تحلیل")
        self.btn_stop = QPushButton("⏹ توقف")
        
        self.btn_load_user.clicked.connect(self.load_user_video)
        self.btn_load_pro.clicked.connect(self.load_pro_video)
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_stop.clicked.connect(self.stop_analysis)

        control_layout.addWidget(self.btn_load_user)
        control_layout.addWidget(self.btn_load_pro)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        left_panel.addLayout(control_layout)

        # اسلایدر زمان
        self.slider = QSlider(Qt.Orientation.Horizontal)
        left_panel.addWidget(self.slider)

        main_layout.addLayout(left_panel, 3)

        # --- بخش راست: داشبورد و تحلیل ---
        right_panel = QVBoxLayout()
        
        # گروه آمار زنده
        stats_group = QGroupBox("📊 آمار زنده و زوایا")
        stats_layout = QGridLayout()
        
        self.lbl_elbow_angle = QLabel("زاویه آرنج: --")
        self.lbl_shoulder_angle = QLabel("زاویه شانه: --")
        self.lbl_detected_obj = QLabel("اشیاء detected: --")
        
        stats_layout.addWidget(self.lbl_elbow_angle, 0, 0)
        stats_layout.addWidget(self.lbl_shoulder_angle, 0, 1)
        stats_layout.addWidget(self.lbl_detected_obj, 1, 0, 1, 2)
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)

        # گروه بازخورد مربی
        feedback_group = QGroupBox("🧠 بازخورد مربی هوشمند")
        feedback_layout = QVBoxLayout()
        self.feedback_text = QLabel("در حال انتظار برای تحلیل...")
        self.feedback_text.setWordWrap(True)
        self.feedback_text.setStyleSheet("color: #ffcc00; font-weight: bold;")
        feedback_layout.addWidget(self.feedback_text)
        feedback_group.setLayout(feedback_layout)
        right_panel.addWidget(feedback_group)

        # نمودار تغییرات زاویه
        graph_group = QGroupBox("📈 نمودار تغییرات زاویه آرنج")
        graph_layout = QVBoxLayout()
        self.angle_plot = pg.PlotWidget()
        self.angle_plot.setBackground('#1e1e1e')
        self.angle_plot.getAxis('left').setPen(color='#ffffff')
        self.angle_plot.getAxis('bottom').setPen(color='#ffffff')
        self.angle_plot.showGrid(x=True, y=True, alpha=0.3)
        self.angle_data = []
        graph_layout.addWidget(self.angle_plot)
        graph_group.setLayout(graph_layout)
        right_panel.addWidget(graph_group)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, 1)

    def load_user_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "انتخاب ویدئوی تمرین", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.current_video_path = path
            self.feedback_text.setText(f"ویدئوی شما بارگذاری شد: {path.split('/')[-1]}")
    
    def load_pro_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "انتخاب ویدئوی بازیکن حرفه‌ای", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.reference_video_path = path
            self.feedback_text.setText("در حال پردازش ویدئوی حرفه‌ای به عنوان الگو... (لطفاً صبر کنید)")
            self.process_reference_video(path)

    def process_reference_video(self, path):
        # پردازش ساده برای استخراج میانگین زوایای ویدئوی مرجع
        cap = cv2.VideoCapture(path)
        angles = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose_solution.process(rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                if len(lm) > 15:
                    ang = self.calculate_angle_static(lm[11], lm[13], lm[15])
                    angles.append(ang)
        cap.release()
        
        if angles:
            avg_angle = sum(angles) / len(angles)
            self.reference_data = {'elbow': avg_angle}
            self.feedback_text.setText(f"الگو ثبت شد! میانگین زاویه آرنج پرو: {avg_angle:.1f}°")
        else:
            self.feedback_text.setText("خطا در تشخیص چهره/بدن در ویدئوی الگو.")

    def calculate_angle_static(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    def start_analysis(self):
        if not self.current_video_path:
            self.feedback_text.setText("❌ ابتدا ویدئوی خود را بارگذاری کنید!")
            return
        
        self.btn_start.setEnabled(False)
        self.worker = VideoAnalysisThread(self.current_video_path, self.reference_data)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.analysis_ready.connect(self.update_dashboard)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.start()

    def stop_analysis(self):
        if self.worker:
            self.worker.stop()
            self.btn_start.setEnabled(True)

    def analysis_finished(self):
        self.btn_start.setEnabled(True)
        self.feedback_text.setText("✅ تحلیل کامل شد.")

    def update_frame(self, frame, data):
        # رسم نتایج روی فریم
        if data['pose_landmarks']:
            mp_drawing.draw_landmarks(frame, data['pose_landmarks'], mp_pose.POSE_CONNECTIONS)
        
        # تبدیل به فرمت QT
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def update_dashboard(self, data):
        if 'elbow' in data['angles']:
            angle = data['angles']['elbow']
            self.lbl_elbow_angle.setText(f"زاویه آرنج: {angle:.1f}°")
            
            # آپدیت نمودار
            self.angle_data.append(angle)
            if len(self.angle_data) > 100:
                self.angle_data.pop(0)
            self.angle_plot.clear()
            self.angle_plot.plot(self.angle_data, pen='y')
        
        if 'shoulder' in data['angles']:
            self.lbl_shoulder_angle.setText(f"زاویه شانه: {data['angles']['shoulder']:.1f}°")

        objs = ", ".join([f"{o['label']}({o['conf']:.2f})" for o in data['objects'][:3]])
        self.lbl_detected_obj.setText(f"اشیاء: {objs if objs else 'هیچ'}")

        if data['feedback']:
            msg = "<br>".join(data['feedback'])
            self.feedback_text.setText(msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # فونت فارسی/انگلیسی مناسب (در صورت نصب بودن)
    font = QFont("Segoe UI", 12)
    app.setFont(font)

    window = TennisCoachApp()
    window.show()
    sys.exit(app.exec())
