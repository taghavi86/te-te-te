"""
Stroke Detection and Phase Segmentation
تشخیص ضربات و تقسیم‌بندی فازهای حرکتی
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class StrokePhase(Enum):
    """فازهای مختلف ضربه"""
    PREPARATION = "preparation"
    BACKSWING = "backswing"
    ACCELERATION = "acceleration"
    CONTACT_PROXY = "contact_proxy"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


@dataclass
class Stroke:
    """ساختار داده برای یک ضربه کامل"""
    stroke_id: int
    start_frame: int
    end_frame: int
    stroke_type: str  # forehand, backhand, etc.
    phases: Dict[StrokePhase, Tuple[int, int]] = field(default_factory=dict)
    contact_frame: Optional[int] = None
    contact_confidence: float = 0.0
    peak_velocity_frame: Optional[int] = None
    peak_velocity: float = 0.0
    avg_motion_energy: float = 0.0
    confidence: float = 1.0


@dataclass
class MotionEnergyPeak:
    """پیک انرژی حرکتی برای تشخیص ضربه"""
    frame: int
    energy: float
    magnitude: float


class StrokeDetector:
    """تشخیص خودکار ضربات بر اساس ویژگی‌های بیومکانیکی"""
    
    def __init__(self, config: dict):
        self.config = config
        self.min_stroke_frames = config.get('strokes', {}).get('min_frames', 15)
        self.max_stroke_frames = config.get('strokes', {}).get('max_frames', 120)
        self.velocity_threshold_factor = config.get('strokes', {}).get('velocity_threshold', 1.5)
        self.energy_threshold_factor = config.get('strokes', {}).get('energy_threshold', 2.0)
    
    def detect_strokes(self, biomechanical_features: List[dict],
                      handedness: str = 'auto') -> List[Stroke]:
        """
        تشخیص ضربات از روی داده‌های بیومکانیکی
        
        Args:
            biomechanical_features: لیست ویژگی‌های هر فریم
            handedness: راست‌دست، چپ‌دست یا auto
            
        Returns:
            لیست ضربات تشخیص داده شده
        """
        if len(biomechanical_features) < self.min_stroke_frames:
            return []
        
        # استخراج سیگنال‌های کلیدی
        motion_energies = [f.get('motion_energy', 0) for f in biomechanical_features]
        
        # استخراج سرعت مچ دست (برای دست غالب)
        wrist_key = 'wrist_r' if handedness == 'right' else 'wrist_l'
        if handedness == 'auto':
            wrist_key = 'wrist_r'  # پیش‌فرض راست‌دست
        
        wrist_velocities = []
        for f in biomechanical_features:
            velocities = f.get('velocities', {})
            if wrist_key in velocities:
                wrist_velocities.append(velocities[wrist_key].value)
            else:
                wrist_velocities.append(0.0)
        
        # پیدا کردن پیک‌های انرژی حرکتی
        energy_peaks = self._find_peaks(motion_energies)
        
        # پیدا کردن پیک‌های سرعت مچ
        velocity_peaks = self._find_peaks(wrist_velocities)
        
        # ترکیب پیک‌ها برای کاندیداهای ضربه
        stroke_candidates = self._merge_candidates(energy_peaks, velocity_peaks)
        
        # ایجاد اشیاء Stroke
        strokes = []
        for i, candidate in enumerate(stroke_candidates):
            stroke = self._create_stroke(
                candidate, 
                biomechanical_features, 
                i,
                handedness
            )
            
            if stroke and self._validate_stroke(stroke):
                strokes.append(stroke)
        
        return strokes
    
    def _find_peaks(self, signal: List[float], 
                   min_distance: int = 10) -> List[MotionEnergyPeak]:
        """پیدا کردن پیک‌های محلی در سیگنال"""
        peaks = []
        n = len(signal)
        
        if n < 3:
            return peaks
        
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        
        threshold = mean_val + self.velocity_threshold_factor * std_val
        
        for i in range(min_distance, n - min_distance):
            # بررسی اینکه آیا نقطه فعلی پیک محلی است
            is_peak = True
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and signal[j] >= signal[i]:
                    is_peak = False
                    break
            
            if is_peak and signal[i] > threshold:
                peaks.append(MotionEnergyPeak(
                    frame=i,
                    energy=signal[i],
                    magnitude=signal[i] / mean_val if mean_val > 0 else 0
                ))
        
        # مرتب‌سازی بر اساس magnitude
        peaks.sort(key=lambda p: p.magnitude, reverse=True)
        
        return peaks
    
    def _merge_candidates(self, energy_peaks: List[MotionEnergyPeak],
                         velocity_peaks: List[MotionEnergyPeak],
                         merge_window: int = 15) -> List[MotionEnergyPeak]:
        """ادغام پیک‌های نزدیک به هم"""
        all_peaks = []
        
        # اضافه کردن پیک‌های انرژی
        for p in energy_peaks:
            all_peaks.append((p.frame, p.energy * 0.6))  # وزن انرژی
        
        # اضافه کردن پیک‌های سرعت
        for p in velocity_peaks:
            all_peaks.append((p.frame, p.magnitude * 0.4))  # وزن سرعت
        
        # گروه‌بندی پیک‌های نزدیک
        all_peaks.sort(key=lambda x: x[0])
        
        merged = []
        if not all_peaks:
            return []
        
        current_group = [all_peaks[0]]
        
        for i in range(1, len(all_peaks)):
            frame, score = all_peaks[i]
            prev_frame, prev_score = current_group[-1]
            
            if frame - prev_frame <= merge_window:
                current_group.append((frame, score))
            else:
                # محاسبه میانگین weighted برای گروه فعلی
                total_weight = sum(s for _, s in current_group)
                avg_frame = sum(f * s for f, s in current_group) / total_weight if total_weight > 0 else current_group[0][0]
                merged.append(MotionEnergyPeak(
                    frame=int(avg_frame),
                    energy=total_weight / len(current_group),
                    magnitude=total_weight
                ))
                current_group = [(frame, score)]
        
        # آخرین گروه
        if current_group:
            total_weight = sum(s for _, s in current_group)
            avg_frame = sum(f * s for f, s in current_group) / total_weight if total_weight > 0 else current_group[0][0]
            merged.append(MotionEnergyPeak(
                frame=int(avg_frame),
                energy=total_weight / len(current_group),
                magnitude=total_weight
            ))
        
        return merged
    
    def _create_stroke(self, candidate: MotionEnergyPeak,
                      features: List[dict],
                      stroke_id: int,
                      handedness: str) -> Optional[Stroke]:
        """ایجاد شیء Stroke از یک کاندیدا"""
        peak_frame = candidate.frame
        
        # پیدا کردن شروع و پایان ضربه
        start_frame = self._find_phase_boundary(
            features, 
            peak_frame, 
            direction='backward',
            min_energy_ratio=0.3
        )
        
        end_frame = self._find_phase_boundary(
            features,
            peak_frame,
            direction='forward',
            min_energy_ratio=0.3
        )
        
        if start_frame is None or end_frame is None:
            return None
        
        # تخمین نقطه تماس (Contact Proxy)
        contact_frame, contact_conf = self._estimate_contact_point(
            features, 
            start_frame, 
            end_frame,
            peak_frame
        )
        
        # تقسیم‌بندی فازها
        phases = self._segment_phases(
            features,
            start_frame,
            end_frame,
            contact_frame
        )
        
        # محاسبه میانگین انرژی حرکتی
        energies = [features[i].get('motion_energy', 0) 
                   for i in range(start_frame, end_frame + 1)]
        avg_energy = np.mean(energies) if energies else 0
        
        # پیدا کردن حداکثر سرعت
        wrist_key = 'wrist_r' if handedness == 'right' else 'wrist_l'
        max_vel = 0
        max_vel_frame = peak_frame
        
        for i in range(start_frame, end_frame + 1):
            velocities = features[i].get('velocities', {})
            if wrist_key in velocities:
                vel = velocities[wrist_key].value
                if vel > max_vel:
                    max_vel = vel
                    max_vel_frame = i
        
        return Stroke(
            stroke_id=stroke_id,
            start_frame=start_frame,
            end_frame=end_frame,
            stroke_type=self._classify_stroke(features, start_frame, end_frame, handedness),
            phases=phases,
            contact_frame=contact_frame,
            contact_confidence=contact_conf,
            peak_velocity_frame=max_vel_frame,
            peak_velocity=max_vel,
            avg_motion_energy=avg_energy,
            confidence=min(1.0, candidate.magnitude / 5.0)  # نرمال‌سازی confidence
        )
    
    def _find_phase_boundary(self, features: List[dict],
                            reference_frame: int,
                            direction: str,
                            min_energy_ratio: float,
                            max_search: int = 50) -> Optional[int]:
        """پیدا کردن مرز فاز (شروع یا پایان ضربه)"""
        ref_energy = features[reference_frame].get('motion_energy', 1)
        threshold = ref_energy * min_energy_ratio
        
        step = 1 if direction == 'forward' else -1
        start_idx = reference_frame + step
        end_idx = len(features) if direction == 'forward' else -1
        
        low_energy_count = 0
        required_low_count = 5  # نیاز به چند فریم متوالی با انرژی پایین
        
        for i in range(start_idx, end_idx, step):
            if i < 0 or i >= len(features):
                break
            
            energy = features[i].get('motion_energy', 0)
            
            if energy < threshold:
                low_energy_count += 1
                if low_energy_count >= required_low_count:
                    return i
            else:
                low_energy_count = 0
            
            # محدودیت جستجو
            if abs(i - reference_frame) > max_search:
                break
        
        # اگر مرز واضحی پیدا نشد، به حداکثر فاصله می‌رویم
        boundary = reference_frame + (step * max_search)
        return max(0, min(len(features) - 1, boundary))
    
    def _estimate_contact_point(self, features: List[dict],
                               start_frame: int,
                               end_frame: int,
                               peak_frame: int) -> Tuple[int, float]:
        """
        تخمین نقطه تماس (Contact Proxy)
        بدون تشخیص راکت و توپ، از پیک شتاب استفاده می‌کنیم
        """
        # جستجوی پیک شتاب در بازه تسریع
        acceleration_peak_frame = peak_frame
        max_acceleration = 0
        
        # محاسبه شتاب از تغییرات سرعت
        for i in range(max(start_frame, peak_frame - 10), 
                      min(end_frame, peak_frame + 10)):
            if i <= start_frame or i >= end_frame:
                continue
            
            # مشتق دوم موقعیت ≈ شتاب
            if i > 0 and i < len(features) - 1:
                curr_energy = features[i].get('motion_energy', 0)
                prev_energy = features[i-1].get('motion_energy', 0)
                next_energy = features[i+1].get('motion_energy', 0)
                
                acceleration = (next_energy - 2*curr_energy + prev_energy)
                
                if acceleration > max_acceleration:
                    max_acceleration = acceleration
                    acceleration_peak_frame = i
        
        # محاسبه confidence بر اساس وضوح پیک
        confidence = min(1.0, max_acceleration / 100.0) if max_acceleration > 0 else 0.5
        
        return acceleration_peak_frame, confidence
    
    def _segment_phases(self, features: List[dict],
                       start_frame: int,
                       end_frame: int,
                       contact_frame: int) -> Dict[StrokePhase, Tuple[int, int]]:
        """تقسیم ضربه به فازهای مختلف"""
        duration = end_frame - start_frame
        
        if duration < self.min_stroke_frames:
            return {}
        
        # تخمین زمان‌بندی فازها بر اساس الگوهای بیومکانیکی
        phases = {}
        
        # Preparation: قبل از backswing
        prep_end = start_frame + int(duration * 0.15)
        phases[StrokePhase.PREPARATION] = (start_frame, prep_end)
        
        # Backswing: عقب بردن راکت
        backswing_end = start_frame + int(duration * 0.35)
        phases[StrokePhase.BACKSWING] = (prep_end, backswing_end)
        
        # Acceleration: تا نقطه تماس
        accel_end = contact_frame
        phases[StrokePhase.ACCELERATION] = (backswing_end, accel_end)
        
        # Contact Proxy
        contact_duration = max(3, int(duration * 0.05))
        phases[StrokePhase.CONTACT_PROXY] = (
            contact_frame,
            min(contact_frame + contact_duration, end_frame)
        )
        
        # Follow-through: بعد از تماس
        follow_end = contact_frame + int(duration * 0.30)
        phases[StrokePhase.FOLLOW_THROUGH] = (
            phases[StrokePhase.CONTACT_PROXY][1],
            min(follow_end, end_frame)
        )
        
        # Recovery: بازگشت به حالت آماده‌باش
        phases[StrokePhase.RECOVERY] = (
            phases[StrokePhase.FOLLOW_THROUGH][1],
            end_frame
        )
        
        return phases
    
    def _classify_stroke(self, features: List[dict],
                        start_frame: int,
                        end_frame: int,
                        handedness: str) -> str:
        """طبقه‌بندی نوع ضربه (forehand, backhand, etc.)"""
        # تحلیل ساده بر اساس جهت حرکت
        # در نسخه کامل‌تر از الگوهای پیچیده‌تر استفاده می‌شود
        
        wrist_key = 'wrist_r' if handedness == 'right' else 'wrist_l'
        
        # بررسی جهت حرکت مچ دست
        lateral_movement = 0
        
        for i in range(start_frame + 5, min(end_frame, start_frame + 30)):
            if i < len(features):
                velocities = features[i].get('velocities', {})
                if wrist_key in velocities:
                    vx = velocities[wrist_key].vector[0] if len(velocities[wrist_key].vector) > 0 else 0
                    lateral_movement += vx
        
        if lateral_movement > 50:
            return "forehand"
        elif lateral_movement < -50:
            return "backhand"
        else:
            return "unknown"
    
    def _validate_stroke(self, stroke: Stroke) -> bool:
        """اعتبارسنجی ضربه تشخیص داده شده"""
        duration = stroke.end_frame - stroke.start_frame
        
        # بررسی حداقل و حداکثر طول
        if duration < self.min_stroke_frames:
            return False
        if duration > self.max_stroke_frames:
            return False
        
        # بررسی وجود فازهای اصلی
        required_phases = [StrokePhase.BACKSWING, StrokePhase.ACCELERATION]
        for phase in required_phases:
            if phase not in stroke.phases:
                return False
        
        # بررسی confidence
        if stroke.confidence < 0.3:
            return False
        
        return True
    
    def add_manual_stroke(self, start_frame: int, end_frame: int,
                         stroke_type: str = "manual",
                         contact_frame: Optional[int] = None) -> Stroke:
        """افزودن دستی ضربه توسط کاربر"""
        # ایجاد ساختار فاز ساده
        duration = end_frame - start_frame
        phases = {}
        
        if contact_frame is None:
            contact_frame = start_frame + int(duration * 0.5)
        
        phases[StrokePhase.PREPARATION] = (start_frame, start_frame + int(duration * 0.2))
        phases[StrokePhase.BACKSWING] = (start_frame + int(duration * 0.2), start_frame + int(duration * 0.4))
        phases[StrokePhase.ACCELERATION] = (start_frame + int(duration * 0.4), contact_frame)
        phases[StrokePhase.CONTACT_PROXY] = (contact_frame, contact_frame + 3)
        phases[StrokePhase.FOLLOW_THROUGH] = (contact_frame + 3, start_frame + int(duration * 0.8))
        phases[StrokePhase.RECOVERY] = (start_frame + int(duration * 0.8), end_frame)
        
        return Stroke(
            stroke_id=-1,  # Manual strokes have negative IDs
            start_frame=start_frame,
            end_frame=end_frame,
            stroke_type=stroke_type,
            phases=phases,
            contact_frame=contact_frame,
            contact_confidence=1.0,
            confidence=1.0
        )
