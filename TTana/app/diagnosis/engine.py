"""
Diagnosis Engine - تحلیل علل ریشه‌ای مشکلات
Root Cause Analysis برای تشخیص مسائل تکنیکی
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class IssueSeverity(Enum):
    """سطح شدت مشکل"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """دسته‌بندی مشکل"""
    TIMING = "timing"
    ANGLES = "angles"
    VELOCITY = "velocity"
    KINETIC_CHAIN = "kinetic_chain"
    POSTURE = "posture"
    COORDINATION = "coordination"


@dataclass
class TechnicalIssue:
    """ساختار داده برای یک مشکل تکنیکی"""
    issue_id: str
    category: IssueCategory
    severity: IssueSeverity
    description: str
    root_cause: str
    evidence: Dict = field(default_factory=dict)
    affected_frames: Tuple[int, int] = (0, 0)
    confidence: float = 0.0
    suggested_correction: str = ""
    related_issues: List[str] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    """نتیجه نهایی تشخیص"""
    primary_issue: Optional[TechnicalIssue]
    secondary_issues: List[TechnicalIssue]
    strengths: List[str]
    overall_similarity: float
    root_cause_chain: List[str]
    recommended_focus: str


class DiagnosisEngine:
    """موتور تشخیص و تحلیل علل ریشه‌ای"""
    
    def __init__(self, config: dict):
        self.config = config
        self.diagnosis_config = config.get('diagnosis', {})
        
        # آستانه‌های تشخیص
        self.angle_threshold = self.diagnosis_config.get('angle_threshold', 15.0)  # degrees
        self.timing_threshold = self.diagnosis_config.get('timing_threshold', 0.15)  # normalized
        self.velocity_threshold = self.diagnosis_config.get('velocity_threshold', 0.2)  # normalized
        
        # وزن ویژگی‌ها برای تشخیص
        self.feature_importance = {
            'shoulder_timing': 1.5,
            'hip_rotation': 1.4,
            'elbow_trajectory': 1.3,
            'wrist_angle': 1.0,
            'knee_angle': 0.9,
            'trunk_angle': 1.2,
            'kinetic_chain_timing': 1.6
        }
    
    def diagnose(self, comparison_data: Dict, 
                dtw_results: Dict,
                user_features: List[dict],
                pro_features: List[dict]) -> DiagnosisResult:
        """
        انجام تحلیل تشخیصی کامل
        
        Args:
            comparison_data: داده‌های مقایسه featureها
            dtw_results: نتایج DTW
            user_features: ویژگی‌های کاربر
            pro_features: ویژگی‌های بازیکن حرفه‌ای
            
        Returns:
            نتیجه تشخیص شامل masalah اصلی، علل ریشه‌ای و توصیه‌ها
        """
        issues = []
        strengths = []
        
        # تحلیل اختلافات زوایا
        angle_issues = self._analyze_angle_differences(comparison_data)
        issues.extend(angle_issues)
        
        # تحلیل اختلافات زمانی
        timing_issues = self._analyze_timing_differences(dtw_results, user_features, pro_features)
        issues.extend(timing_issues)
        
        # تحلیل زنجیره حرکتی
        chain_issues = self._analyze_kinetic_chain(user_features, pro_features)
        issues.extend(chain_issues)
        
        # تحلیل سرعت‌ها
        velocity_issues = self._analyze_velocity_differences(comparison_data)
        issues.extend(velocity_issues)
        
        # شناسایی نقاط قوت
        strengths = self._identify_strengths(comparison_data, dtw_results)
        
        # مرتب‌سازی مسائل بر اساس اهمیت
        issues.sort(key=lambda x: (
            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity.value],
            -x.confidence
        ))
        
        # پیدا کردن علت ریشه‌ای
        root_cause_chain = self._build_root_cause_chain(issues)
        
        # تعیین مسئله اصلی
        primary_issue = issues[0] if issues else None
        
        # مسائل ثانویه
        secondary_issues = issues[1:min(5, len(issues))]
        
        # توصیه تمرکزی
        recommended_focus = self._generate_focus_recommendation(primary_issue, root_cause_chain)
        
        return DiagnosisResult(
            primary_issue=primary_issue,
            secondary_issues=secondary_issues,
            strengths=strengths,
            overall_similarity=dtw_results.get('avg_similarity', 0.0),
            root_cause_chain=root_cause_chain,
            recommended_focus=recommended_focus
        )
    
    def _analyze_angle_differences(self, comparison_data: Dict) -> List[TechnicalIssue]:
        """تحلیل اختلافات زوایای مفاصل"""
        issues = []
        
        feature_comparisons = comparison_data.get('feature_comparisons', [])
        
        for comp in feature_comparisons:
            if '_angle' not in comp.get('feature', ''):
                continue
            
            diff = abs(comp.get('difference', 0))
            normalized_diff = comp.get('normalized_difference', 0)
            
            if diff > self.angle_threshold:
                # تعیین شدت
                if diff > 30:
                    severity = IssueSeverity.CRITICAL
                elif diff > 20:
                    severity = IssueSeverity.HIGH
                elif diff > 15:
                    severity = IssueSeverity.MEDIUM
                else:
                    severity = IssueSeverity.LOW
                
                joint = comp['feature'].replace('_angle', '')
                
                issue = TechnicalIssue(
                    issue_id=f"angle_{joint}",
                    category=IssueCategory.ANGLES,
                    severity=severity,
                    description=f"زاویه {self._translate_joint(joint)} {diff:.1f} درجه اختلاف دارد",
                    root_cause=self._infer_angle_root_cause(joint, diff),
                    evidence={
                        'user_value': comp.get('user', 0),
                        'pro_value': comp.get('reference', 0),
                        'difference': diff
                    },
                    confidence=min(1.0, normalized_diff),
                    suggested_correction=self._get_angle_correction(joint, diff)
                )
                
                issues.append(issue)
        
        return issues
    
    def _analyze_timing_differences(self, dtw_results: Dict,
                                   user_features: List[dict],
                                   pro_features: List[dict]) -> List[TechnicalIssue]:
        """تحلیل اختلافات زمانی با استفاده از DTW"""
        issues = []
        
        alignment_path = dtw_results.get('alignment_path', [])
        
        if not alignment_path:
            return issues
        
        # تحلیل تاخیرهای زمانی در فازهای مختلف
        phase_delays = self._calculate_phase_delays(user_features, pro_features, alignment_path)
        
        for phase, delay in phase_delays.items():
            if abs(delay) > self.timing_threshold:
                severity = IssueSeverity.HIGH if abs(delay) > 0.25 else IssueSeverity.MEDIUM
                
                issue = TechnicalIssue(
                    issue_id=f"timing_{phase}",
                    category=IssueCategory.TIMING,
                    severity=severity,
                    description=f"تاخیر {abs(delay)*100:.0f}%یی در فاز {self._translate_phase(phase)}",
                    root_cause=f"شروع دیرهنگام زنجیره حرکتی در {phase}",
                    evidence={
                        'delay': delay,
                        'phase': phase
                    },
                    confidence=min(1.0, abs(delay) / 0.3),
                    suggested_correction=f"تمرکز بر شروع به‌موقع فاز {self._translate_phase(phase)}"
                )
                
                issues.append(issue)
        
        return issues
    
    def _analyze_kinetic_chain(self, user_features: List[dict],
                              pro_features: List[dict]) -> List[TechnicalIssue]:
        """تحلیل زنجیره حرکتی پایین‌تنه به بالاتنه"""
        issues = []
        
        # بررسی توالی فعال‌سازی مفاصل
        user_activation = self._extract_activation_sequence(user_features)
        pro_activation = self._extract_activation_sequence(pro_features)
        
        if user_activation and pro_activation:
            # مقایسه توالی
            if user_activation != pro_activation:
                issue = TechnicalIssue(
                    issue_id="kinetic_chain_order",
                    category=IssueCategory.KINETIC_CHAIN,
                    severity=IssueSeverity.HIGH,
                    description="ترتیب نادرست فعال‌سازی زنجیره حرکتی",
                    root_cause="عدم رعایت توالی پایین‌تنه → لگن → تنه → شانه → آرنج → مچ",
                    evidence={
                        'user_sequence': user_activation,
                        'pro_sequence': pro_activation
                    },
                    confidence=0.8,
                    suggested_correction="تمرین حرکات آهسته با تأکید بر توالی صحیح فعال‌سازی"
                )
                issues.append(issue)
        
        # بررسی تاخیر بین لگن و شانه
        hip_shoulder_delay = self._calculate_hip_shoulder_delay(user_features, pro_features)
        
        if hip_shoulder_delay > 0.1:
            issue = TechnicalIssue(
                issue_id="hip_shoulder_delay",
                category=IssueCategory.KINETIC_CHAIN,
                severity=IssueSeverity.MEDIUM if hip_shoulder_delay < 0.2 else IssueSeverity.HIGH,
                description=f"تاخیر {hip_shoulder_delay*100:.0f} میلی‌ثانیه‌ای بین چرخش لگن و شانه",
                root_cause="عدم انتقال بهینه نیرو از پایین‌تنه به بالاتنه",
                evidence={'delay_ms': hip_shoulder_delay * 1000},
                confidence=min(1.0, hip_shoulder_delay / 0.15),
                suggested_correction="تمرین چرخش لگن قبل از شروع چرخش شانه"
            )
            issues.append(issue)
        
        return issues
    
    def _analyze_velocity_differences(self, comparison_data: Dict) -> List[TechnicalIssue]:
        """تحلیل اختلافات سرعت"""
        issues = []
        
        feature_comparisons = comparison_data.get('feature_comparisons', [])
        
        for comp in feature_comparisons:
            if '_velocity' not in comp.get('feature', ''):
                continue
            
            diff_ratio = abs(comp.get('normalized_difference', 0))
            
            if diff_ratio > self.velocity_threshold:
                severity = IssueSeverity.HIGH if diff_ratio > 0.4 else IssueSeverity.MEDIUM
                
                kp = comp['feature'].replace('_velocity', '')
                
                issue = TechnicalIssue(
                    issue_id=f"velocity_{kp}",
                    category=IssueCategory.VELOCITY,
                    severity=severity,
                    description=f"سرعت {self._translate_keypoint(kp)} {diff_ratio*100:.0f}% اختلاف دارد",
                    root_cause=self._infer_velocity_root_cause(kp, diff_ratio),
                    evidence=comp,
                    confidence=min(1.0, diff_ratio / 0.5),
                    suggested_correction=self._get_velocity_correction(kp, diff_ratio)
                )
                
                issues.append(issue)
        
        return issues
    
    def _identify_strengths(self, comparison_data: Dict, 
                           dtw_results: Dict) -> List[str]:
        """شناسایی نقاط قوت بازیکن"""
        strengths = []
        
        feature_comparisons = comparison_data.get('feature_comparisons', [])
        
        for comp in feature_comparisons:
            norm_diff = abs(comp.get('normalized_difference', 0))
            
            if norm_diff < 0.1:  # کمتر از 10% اختلاف
                feature = comp.get('feature', '')
                strengths.append(f"{self._translate_feature(feature)} مشابه الگوی حرفه‌ای است")
        
        # بررسی شباهت کلی
        overall_similarity = dtw_results.get('avg_similarity', 0)
        if overall_similarity > 0.7:
            strengths.append(f"شباهت کلی {overall_similarity*100:.0f}% با الگوی حرفه‌ای")
        
        return strengths[:5]  # حداکثر 5 نقطه قوت
    
    def _build_root_cause_chain(self, issues: List[TechnicalIssue]) -> List[str]:
        """ساخت زنجیره علل ریشه‌ای"""
        if not issues:
            return []
        
        chain = []
        
        # اولویت‌بندی بر اساس دسته‌بندی
        kinetic_issues = [i for i in issues if i.category == IssueCategory.KINETIC_CHAIN]
        timing_issues = [i for i in issues if i.category == IssueCategory.TIMING]
        angle_issues = [i for i in issues if i.category == IssueCategory.ANGLES]
        
        if kinetic_issues:
            chain.append(kinetic_issues[0].root_cause)
        
        if timing_issues and len(chain) < 3:
            chain.append(timing_issues[0].root_cause)
        
        if angle_issues and len(chain) < 3:
            chain.append(angle_issues[0].root_cause)
        
        return chain
    
    def _generate_focus_recommendation(self, primary_issue: Optional[TechnicalIssue],
                                       root_cause_chain: List[str]) -> str:
        """تولید توصیه تمرکزی"""
        if not primary_issue:
            return "ادامه تمرینات فعلی - عملکرد کلی خوب است"
        
        return f"تمرکز اصلی: {primary_issue.suggested_correction}"
    
    # Helper methods
    def _translate_joint(self, joint: str) -> str:
        """ترجمه نام مفصل به فارسی"""
        translations = {
            'elbow_l': 'آرنج چپ', 'elbow_r': 'آرنج راست',
            'shoulder_l': 'شانه چپ', 'shoulder_r': 'شانه راست',
            'hip_l': 'لگن چپ', 'hip_r': 'لگن راست',
            'knee_l': 'زانوی چپ', 'knee_r': 'زانوی راست',
            'wrist_l': 'مچ دست چپ', 'wrist_r': 'مچ دست راست'
        }
        return translations.get(joint, joint)
    
    def _translate_keypoint(self, kp: str) -> str:
        """ترجمه نام نقطه کلیدی"""
        return self._translate_joint(kp.replace('_', ' '))
    
    def _translate_phase(self, phase: str) -> str:
        """ترجمه فاز حرکتی"""
        translations = {
            'preparation': 'آماده‌سازی',
            'backswing': 'عقب بردن',
            'acceleration': 'شتاب‌گیری',
            'contact': 'تماس',
            'follow_through': 'ادامه حرکت',
            'recovery': 'بازگشت'
        }
        return translations.get(phase, phase)
    
    def _translate_feature(self, feature: str) -> str:
        """ترجمه نام ویژگی"""
        return feature.replace('_', ' ')
    
    def _infer_angle_root_cause(self, joint: str, diff: float) -> str:
        """استنتاج علت ریشه‌ای اختلاف زاویه"""
        if 'elbow' in joint:
            return "زمان‌بندی نادرست باز شدن آرنج"
        elif 'shoulder' in joint:
            return "چرخش ناکافی یا بیش‌ازحد شانه"
        elif 'hip' in joint:
            return "عدم چرخش مناسب لگن"
        elif 'knee' in joint:
            return "خم بودن نادرست زانو"
        else:
            return "خطای تکنیکی در تراز مفصل"
    
    def _infer_velocity_root_cause(self, kp: str, diff_ratio: float) -> str:
        """استنتاج علت ریشه‌ای اختلاف سرعت"""
        if 'wrist' in kp:
            return "عدم استفاده بهینه از مچ در انتهای حرکت"
        elif 'elbow' in kp:
            return "شتاب‌گیری نادرست آرنج"
        elif 'shoulder' in kp:
            return "انتقال ناقص نیرو از تنه به شانه"
        else:
            return "هماهنگی عصبی-عضلانی نیاز به بهبود دارد"
    
    def _get_angle_correction(self, joint: str, diff: float) -> str:
        """دریافت توصیه اصلاحی برای زاویه"""
        corrections = {
            'elbow': "تمرین باز شدن کنترل‌شده آرنج در آینه",
            'shoulder': "تمرین چرخش شانه با چوب تنیس",
            'hip': "تمرین چرخش لگن بدون حرکت شانه",
            'knee': "تمرین حالت صحیح زانو در وضعیت آماده‌باش"
        }
        
        for key, value in corrections.items():
            if key in joint:
                return value
        
        return "تمرین تکنیک پایه با سرعت کم"
    
    def _get_velocity_correction(self, kp: str, diff_ratio: float) -> str:
        """دریافت توصیه اصلاحی برای سرعت"""
        if 'wrist' in kp:
            return "تمرین ضربه‌زنی با تأکید بر اسنپ مچ"
        elif 'elbow' in kp:
            return "تمرین شتاب‌گیری پیوسته آرنج"
        else:
            return "تمرین انتقال نیرو از پایین‌تنه"
    
    def _calculate_phase_delays(self, user_features: List[dict],
                               pro_features: List[dict],
                               alignment_path: List[Tuple[int, int]]) -> Dict[str, float]:
        """محاسبه تاخیرهای فاز"""
        delays = {}
        
        # ساده‌سازی: محاسبه تاخیر متوسط
        if alignment_path:
            total_delay = sum(u - p for u, p in alignment_path)
            avg_delay = total_delay / len(alignment_path) / max(len(user_features), 1)
            delays['overall'] = avg_delay
        
        return delays
    
    def _extract_activation_sequence(self, features: List[dict]) -> List[str]:
        """استخراج توالی فعال‌سازی مفاصل"""
        if not features:
            return []
        
        # تحلیل ساده بر اساس زمان رسیدن به پیک سرعت
        activation_times = {}
        
        for kp in ['hip', 'shoulder', 'elbow', 'wrist']:
            peak_frame = self._find_peak_velocity_frame(features, kp)
            if peak_frame:
                activation_times[kp] = peak_frame
        
        # مرتب‌سازی بر اساس زمان
        sorted_joints = sorted(activation_times.keys(), key=lambda x: activation_times[x])
        
        return sorted_joints
    
    def _find_peak_velocity_frame(self, features: List[dict], joint_prefix: str) -> Optional[int]:
        """پیدا کردن فریم پیک سرعت برای یک مفصل"""
        max_vel = 0
        peak_frame = None
        
        for i, frame in enumerate(features):
            velocities = frame.get('velocities', {})
            
            for kp, vel_obj in velocities.items():
                if joint_prefix in kp:
                    if vel_obj.value > max_vel:
                        max_vel = vel_obj.value
                        peak_frame = i
        
        return peak_frame
    
    def _calculate_hip_shoulder_delay(self, user_features: List[dict],
                                     pro_features: List[dict]) -> float:
        """محاسبه تاخیر بین فعال‌سازی لگن و شانه"""
        user_hip_time = self._find_peak_velocity_frame(user_features, 'hip')
        user_shoulder_time = self._find_peak_velocity_frame(user_features, 'shoulder')
        
        if user_hip_time is None or user_shoulder_time is None:
            return 0.0
        
        delay = user_shoulder_time - user_hip_time
        
        # نرمال‌سازی به واحد زمانی
        if len(user_features) > 0:
            delay = delay / len(user_features)
        
        return abs(delay)
