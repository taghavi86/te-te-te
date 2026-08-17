# راهنمای نصب و راه‌اندازی TTana

## AI Table Tennis Coach - مربی هوشمند تنیس روی میز

---

## 📋 فهرست مطالب

1. [پیش‌نیازهای سخت‌افزاری](#پیش‌نیازهای-سخت‌افزاری)
2. [پیش‌نیازهای نرم‌افزاری](#پیش‌نیازهای-نرم‌افزاری)
3. [مراحل نصب گام‌به‌گام](#مراحل-نصب-گام‌به‌گام)
4. [پیکربندی LM Studio و Qwen](#پیکربندی-lm-studio-و-qwen)
5. [اجرای برنامه](#اجرای-برنامه)
6. [گردش کار تحلیل](#گردش-کار-تحلیل)
7. [عیب‌یابی](#عیب‌یابی)

---

## پیش‌نیازهای سخت‌افزاری

### حداقل مشخصات سیستم

| کامپوننت | حداقل | پیشنهادی |
|----------|-------|----------|
| **سیستم‌عامل** | Windows 10 (64-bit) | Windows 11 (64-bit) |
| **پردازنده** | Intel i5-8400 / AMD Ryzen 5 2600 | Intel i7-10700K / AMD Ryzen 7 5800X |
| **حافظه RAM** | 16 GB | 32 GB |
| **کارت گرافیک** | NVIDIA RTX 2060 Super (8GB) | NVIDIA RTX 3080 (10GB+) |
| **فضای ذخیره‌سازی** | 50 GB SSD | 100 GB NVMe SSD |
| **VRAM** | 8 GB | 12+ GB |

### نکات مهم GPU

- **فقط NVIDIA**: سیستم از CUDA استفاده می‌کند
- **حداقل VRAM**: 8 گیگابایت برای اجرای همزمان Pose Model و LLM
- **CUDA Support**: نسخه CUDA 11.8 یا 12.x

---

## پیش‌نیازهای نرم‌افزاری

### 1. Python 3.11+

```bash
# دانلود از python.org
https://www.python.org/downloads/

# بررسی نسخه
python --version
# باید باشد: Python 3.11.x یا بالاتر
```

### 2. CUDA Toolkit

```bash
# دانلود CUDA Toolkit 11.8
https://developer.nvidia.com/cuda-11-8-0-download-archive

# یا CUDA 12.x
https://developer.nvidia.com/cuda-downloads

# بررسی نصب
nvcc --version
nvidia-smi
```

### 3. FFmpeg

```bash
# دانلود از ffmpeg.org
https://ffmpeg.org/download.html

# یا با Chocolatey (Windows)
choco install ffmpeg

# بررسی نصب
ffmpeg -version
```

### 4. Git (اختیاری)

```bash
# دانلود از git-scm.com
https://git-scm.com/download/win
```

---

## مراحل نصب گام‌به‌گام

### مرحله ۱: آماده‌سازی محیط

```bash
# رفتن به پوشه پروژه
cd TTana

# ساخت محیط مجازی
python -m venv venv

# فعال‌سازی محیط مجازی
# در Windows:
venv\Scripts\activate

# در Linux/Mac:
source venv/bin/activate
```

### مرحله ۲: نصب PyTorch با CUDA

```bash
# برای CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# برای CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# بررسی نصب PyTorch با CUDA
python -c "import torch; print(torch.cuda.is_available())"
# باید چاپ کند: True
```

### مرحله ۳: نصب MMCV و MMPose

```bash
# نصب MMEngine
pip install mmengine

# نصب MMCV با نسخه سازگار
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# برای CUDA 12.x
# pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# نصب MMPose
pip install mmpose>=1.0.0

# نصب RTMPose pre-trained models
mim download mmpose --config rtmpose-m_8xb32-210e_coco-256x192 --dest .
```

### مرحله ۴: نصب سایر وابستگی‌ها

```bash
# نصب تمام dependencies
pip install -r requirements.txt
```

### مرحله ۵: بررسی نصب

```bash
# اجرای تست سلامت
python tests/test_installation.py

# باید تمام تست‌ها Pass شوند
```

---

## پیکربندی LM Studio و Qwen

### مرحله ۱: نصب LM Studio

1. دانلود از سایت رسمی:
   ```
   https://lmstudio.ai/
   ```

2. نصب و اجرای LM Studio

### مرحله ۲: دانلود مدل Qwen3.5-9B

1. باز کردن LM Studio
2. رفتن به تب **"Discover"** یا **"Search"**
3. جستجوی `Qwen3.5-9B` یا `Qwen2.5-7B` (نسخه سبک‌تر)
4. انتخاب نسخه Quantized مناسب:
   - **Q4_K_M**: تعادل خوب بین کیفیت و سرعت (توصیه شده)
   - **Q5_K_M**: کیفیت بالاتر، مصرف VRAM بیشتر
   - **Q3_K_S**: سبک‌ترین، کیفیت پایین‌تر

5. کلیک روی **Download**

### مرحله ۳: راه‌اندازی Local Server

1. رفتن به تب **"Local Server"** در LM Studio
2. انتخاب مدل دانلود شده از منوی کشویی
3. تنظیمات پیشنهادی:
   ```
   Context Length: 4096
   GPU Offload: Max (تمام لایه‌ها روی GPU)
   ```
4. کلیک روی **"Start Server"**
5. بررسی آدرس سرور (پیش‌فرض: `http://localhost:1234`)

### مرحله ۴: تست اتصال

```bash
# با curl
curl http://localhost:1234/v1/models

# باید لیست مدل‌ها را برگرداند
```

---

## اجرای برنامه

### روش ۱: اجرای مستقیم

```bash
# فعال‌سازی محیط مجازی
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# اجرای برنامه
python app/main.py
```

### روش ۲: ساخت فایل اجرایی (اختیاری)

```bash
# نصب PyInstaller
pip install pyinstaller

# ساخت executable
pyinstaller --onefile --windowed --name=TTana app/main.py

# فایل اجرایی در dist/TTana.exe ساخته می‌شود
```

### روش ۳: ایجاد میانبر

1. کلیک راست روی دسکتاپ → New → Shortcut
2. مسیر را وارد کنید:
   ```
   C:\path\to\TTana\venv\Scripts\python.exe C:\path\to\TTana\app\main.py
   ```
3. نام را `TTana Coach` بگذارید

---

## گردش کار تحلیل

### ۱. شروع جلسه جدید

1. اجرای برنامه TTana
2. کلیک روی **"New Session"**
3. وارد کردن نام بازیکن (اختیاری)

### ۲. بارگذاری ویدئوها

#### ویدئوی کاربر:
- کلیک روی **"Select User Video"**
- انتخاب فایل ویدئویی از سیستم
- فرمت‌های پشتیبانی شده: MP4, AVI, MOV, MKV

#### ویدئوی مرجع:
- کلیک روی **"Select Reference Video"**
- انتخاب ویدئوی بازیکن حرفه‌ای
- می‌تواند از مسابقات جهانی یا آموزش‌های معتبر باشد

### ۳. تنظیمات تحلیل

- **Handedness**: راست‌دست / چپ‌دست / خودکار
- **Confidence Threshold**: حداقل اطمینان تشخیص (پیش‌فرض: 0.5)
- **Stroke Types**: انواع ضربات برای تحلیل

### ۴. اجرای تحلیل

1. کلیک روی **"Start Analysis"**
2. مشاهده پیشرفت مراحل:
   - ✅ Decode Video
   - ✅ Person Detection
   - ✅ Pose Estimation
   - ✅ Tracking & Smoothing
   - ✅ Stroke Detection
   - ✅ Feature Extraction
   - ✅ DTW Alignment
   - ✅ Comparison & Diagnosis
   - ✅ LLM Report Generation

3. زمان تقریبی: ۲-۵ دقیقه بسته به طول ویدئو

### ۵. مشاهده نتایج

#### Dashboard:
- نمای کلی جلسه
- آمار ضربات
- امتیاز شباهت کلی

#### Video Comparison:
- پخش همزمان دو ویدئو
- Overlay اسکلت بدن
- نمایش تفاوت‌ها با بردارها

#### Biomechanics Panel:
- نمودار زوایای مفاصل
- سرعت‌ها و شتاب‌ها
- مقایسه Side-by-Side

#### Coach Report:
- مسئله اصلی
- علت ریشه‌ای
- شواهد ویدئویی
- مسائل ثانویه
- نقاط قوت
- توصیه‌های اصلاحی
- برنامه تمرینی

#### AI Chat:
- پرسش سوالات درباره تکنیک
- دریافت پاسخ مبتنی بر داده
- درخواست مثال‌های ویدئویی

### ۶. ذخیره و خروج

- گزارش PDF (اختیاری)
- ذخیره جلسه در پایگاه داده
- خروج از برنامه

---

## عیب‌یابی

### مشکل: PyTorch CUDA را تشخیص نمی‌دهد

```bash
# بررسی نسخه CUDA
python -c "import torch; print(torch.version.cuda)"

# نصب مجدد با نسخه صحیح
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### مشکل: MMPose نصب نمی‌شود

```bash
# اطمینان از سازگاری نسخه‌ها
pip show mmcv
pip show torch

# نصب با نسخه خاص
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

### مشکل: LM Studio متصل نمی‌شود

1. بررسی روشن بودن LM Studio
2. بررسی شروع سرور در LM Studio
3. بررسی پورت (پیش‌فرض: 1234)
4. تست با curl:
   ```bash
   curl http://localhost:1234/v1/models
   ```

### مشکل: خطای حافظه GPU (OOM)

1. بستن سایر برنامه‌های گرافیکی
2. کاهش Context Length در LM Studio به 2048
3. استفاده از نسخه Quantized سبک‌تر (Q3_K_S)
4. کاهش resolution ویدئو ورودی

### مشکل: شخص تشخیص داده نمی‌شود

1. اطمینان از نور کافی در ویدئو
2. قرار گرفتن بازیکن در مرکز کادر
3. عدم وجود افراد متعدد در کادر
4. کاهش Confidence Threshold در تنظیمات

### مشکل: خطای FFmpeg

```bash
# نصب مجدد FFmpeg
choco uninstall ffmpeg
choco install ffmpeg

# یا دانلود دستی از ffmpeg.org
```

---

## پشتیبانی و مستندات بیشتر

- **مستندات فنی**: `docs/TECHNICAL_REQUIREMENTS.md`
- **کانفیگ نمونه**: `config.yaml`
- **تست‌ها**: `tests/`

---

## نکات امنیتی

✅ تمام پردازش‌ها محلی انجام می‌شود  
✅ ویدئوها هرگز دستگاه را ترک نمی‌کنند  
✅ هیچ داده‌ای به سرور خارجی ارسال نمی‌شود  
✅ LLM کاملاً آفلاین اجرا می‌شود  

---

## به‌روزرسانی

```bash
# دریافت آخرین نسخه کد
git pull origin main

# به‌روزرسانی dependencies
pip install -r requirements.txt --upgrade

# بررسی نسخه جدید مدل‌ها در LM Studio
```

---

## حذف نصب

```bash
# غیرفعال‌سازی محیط مجازی
deactivate

# حذف پوشه venv
rmdir /s venv  # Windows
rm -rf venv  # Linux/Mac

# حذف مدل‌ها از LM Studio (در صورت نیاز)
```

---

**تهیه شده توسط تیم توسعه TTana**  
**آخرین به‌روزرسانی: 2024**
