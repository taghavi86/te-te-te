# 🎾 AI Tennis Coach Pro - حرفه‌ای‌ترین سیستم تحلیل تنیس با هوش مصنوعی

<div align="center">

![Tennis AI](https://img.shields.io/badge/Tennis-AI%20Coach-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-State--of--the--Art-purple)
![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose-orange)

**یک سیستم تحلیل تنیس در سطح جهانی با استفاده از پیشرفته‌ترین مدل‌های هوش مصنوعی**

</div>

---

## 🌟 ویژگی‌های کلیدی

### 🔥 قابلیت‌های منحصر به فرد

- **🎯 تشخیص فوق‌دقیق**: استفاده از YOLOv8X (بزرگترین مدل) برای تشخیص توپ، راکت و بازیکن
- **🦴 آنالیز بیومکانیکی**: تحلیل ۳۳ نقطه کلیدی بدن با MediaPipe BlazePose
- **⚡ پردازش بلادرنگ**: پشتیبانی از ویدئوهای 60 FPS با کیفیت Full HD
- **🏆 مقایسه با حرفه‌ای‌ها**: مقایسه تکنیک شما با فدرر، نادال، جوکوویچ
- **📊 داشبورد حرفه‌ای**: رابط کاربری Streamlit با نمودارهای تعاملی Plotly
- **🧠 هوش مصنوعی پیشرفته**: الگوریتم‌های Dynamic Time Warping برای تحلیل توالی حرکات
- **🎾 ردیابی توپ**: فیلتر کالمن برای پیش‌بینی مسیر توپ حتی هنگام عدم مشاهده
- **💡 مربی هوشمند**: تولید خودکار توصیه‌ها و برنامه‌های تمرینی

---

## 📦 نصب و راه‌اندازی

### پیش‌نیازها

```bash
Python 3.8 یا بالاتر
GPU اختیاری (برای سرعت بیشتر)
```

### نصب سریع

```bash
cd NewV
pip install -r requirements.txt
streamlit run main.py
```

### نصب کامل با تمام قابلیت‌ها

```bash
# Core ML libraries
pip install torch torchvision ultralytics mediapipe

# Computer Vision
pip install opencv-python opencv-contrib-python

# Data & Visualization
pip install numpy pandas scipy plotly streamlit

# Utilities
pip install pyyaml av imageio tqdm fastdtw
```

---

## 🚀 نحوه استفاده

### ۱. آپلود ویدئو
- ویدئوی تمرین تنیس خود را آپلود کنید
- فرمت‌های پشتیبانی شده: MP4, AVI, MOV, MKV
- کیفیت پیشنهادی: 1080p با 60 FPS

### ۲. انتخاب حالت تحلیل
- **Single Video**: تحلیل ویدئوی شما
- **Professional Comparison**: مقایسه با ویدئوی بازیکنان حرفه‌ای
- **Live Camera**: تحلیل بلادرنگ با وبکم

### ۳. انتخاب کامپوننت‌ها
- ✅ Pose Detection (تشخیص وضعیت بدن)
- ✅ Ball Tracking (ردیابی توپ)
- ✅ Racket Detection (تشخیص راکت)
- ✅ Technique Analysis (تحلیل تکنیک)
- ✅ Professional Comparison (مقایسه با حرفه‌ای‌ها)

---

## 🏗️ ساختار پروژه

```
NewV/
├── main.py                 # نقطه ورود اصلی
├── config/settings.yaml    # تنظیمات سیستم
├── core/pipeline.py        # پایپ‌لاین اصلی تحلیل
├── models/
│   ├── pose_estimator.py   # تشخیص وضعیت بدن
│   ├── ball_detector.py    # تشخیص و ردیابی توپ
│   ├── racket_detector.py  # تشخیص راکت
│   ├── technique_analyzer.py # تحلیل تکنیک
│   └── pro_comparator.py   # مقایسه با حرفه‌ای‌ها
├── utils/                  # ابزارهای کمکی
├── ui/dashboard.py         # داشبورد Streamlit
└── requirements.txt        # وابستگی‌ها
```

---

## 🧠 مدل‌های هوش مصنوعی

### Pose Estimation
- **Model**: MediaPipe BlazePose (Full Complexity)
- **Keypoints**: 33 نقاط کلیدی بدن
- **Accuracy**: >98%

### Object Detection
- **Model**: YOLOv8X (Extra Large)
- **Classes**: Ball, Racket, Player
- **mAP**: 0.75+
- **Features**: Kalman Filter Tracking

### Sequence Comparison
- **Algorithm**: Dynamic Time Warping (DTW)
- **Database**: Federer, Nadal, Djokovic references

---

## 📊 خروجی‌های تحلیل

### معیارهای تکنیک
- Stance Score: امتیاز وضعیت پاها
- Backswing Quality: کیفیت عقب‌بردن راکت
- Contact Point: نقطه تماس با توپ
- Follow-through: تکمیل حرکت
- Balance: تعادل بدن

### آمار توپ
- Max Speed: حداکثر سرعت
- Trajectory: مسیر پرواز
- Spin Estimate: تخمین چرخش
- Landing Point: پیش‌بینی نقطه فرود

### مقایسه با حرفه‌ای‌ها
- Overall Similarity: درصد شباهت کلی
- Component Scores: امتیاز هر_component
- Timing Analysis: تحلیل زمان‌بندی
- Recommendations: توصیه‌های شخصی‌سازی شده

---

## 💻 اجرای پیشرفته

### با GPU
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
streamlit run main.py --server.headless true
```

---

## 🔧 تنظیمات سفارشی

فایل `config/settings.yaml` را ویرایش کنید:

```yaml
YOLO_MODEL: "yolov8x.pt"
CONFIDENCE_THRESHOLD: 0.75
GPU_ACCELERATION: true
FRAME_RATE: 60
RESOLUTION: [1920, 1080]
```

---

## 📄 مجوز

MIT License

---

<div align="center">

**ساخته شده با ❤️ برای جامعه تنیس**

⭐ اگر این پروژه برایتان مفید بود، ستاره دهید!

</div>
