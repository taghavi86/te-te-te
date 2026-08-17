"""
Database Repository - مدیریت ذخیره‌سازی داده‌ها در SQLite
"""
import sqlite3
import json
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path


class DatabaseRepository:
    """مدیریت پایگاه داده SQLite برای جلسات، ویدئوها و تحلیل‌ها"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def connect(self):
        """اتصال به پایگاه داده"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
    
    def close(self):
        """بستن اتصال"""
        if self.conn:
            self.conn.close()
    
    def _initialize_database(self):
        """ایجاد جداول پایگاه داده"""
        self.connect()
        cursor = self.conn.cursor()
        
        # جدول بازیکنان
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                handedness TEXT DEFAULT 'right',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')
        
        # جدول ویدئوها
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                file_hash TEXT,
                file_name TEXT,
                duration REAL,
                fps REAL,
                width INTEGER,
                height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # جدول جلسات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                user_video_id INTEGER,
                reference_video_id INTEGER,
                analysis_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(id),
                FOREIGN KEY (user_video_id) REFERENCES videos(id),
                FOREIGN KEY (reference_video_id) REFERENCES videos(id)
            )
        ''')
        
        # جدول ضربات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strokes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                stroke_id INTEGER,
                start_frame INTEGER,
                end_frame INTEGER,
                stroke_type TEXT,
                contact_frame INTEGER,
                confidence REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # جدول ویژگی‌های بیومکانیکی
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stroke_id INTEGER NOT NULL,
                frame INTEGER,
                feature_name TEXT,
                feature_value REAL,
                confidence REAL,
                FOREIGN KEY (stroke_id) REFERENCES strokes(id)
            )
        ''')
        
        # جدول مقایسه‌ها
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                stroke_pair TEXT,
                dtw_distance REAL,
                similarity_score REAL,
                alignment_data TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # جدول تشخیص‌ها
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                issue_id TEXT,
                category TEXT,
                severity TEXT,
                description TEXT,
                root_cause TEXT,
                evidence TEXT,
                confidence REAL,
                suggested_correction TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # جدول گزارش‌های مربی
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coach_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL UNIQUE,
                summary TEXT,
                primary_issue TEXT,
                root_cause TEXT,
                evidence TEXT,
                secondary_issues TEXT,
                strengths TEXT,
                corrections TEXT,
                training_plan TEXT,
                next_session_goal TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # جدول پیام‌های چت
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                context_used TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        self.conn.commit()
        self.close()
    
    # Player operations
    def create_player(self, name: str, handedness: str = 'right', 
                     notes: str = '') -> int:
        """ایجاد بازیکن جدید"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO players (name, handedness, notes)
            VALUES (?, ?, ?)
        ''', (name, handedness, notes))
        
        player_id = cursor.lastrowid
        self.conn.commit()
        self.close()
        
        return player_id
    
    def get_player(self, player_id: int) -> Optional[Dict]:
        """دریافت اطلاعات بازیکن"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT * FROM players WHERE id = ?', (player_id,))
        row = cursor.fetchone()
        
        self.close()
        
        return dict(row) if row else None
    
    # Video operations
    def create_or_get_video(self, file_path: str, metadata: Dict) -> int:
        """ایجاد یا دریافت ویدئو موجود"""
        self.connect()
        cursor = self.conn.cursor()
        
        # محاسبه hash فایل
        file_hash = self._calculate_file_hash(file_path)
        
        # بررسی وجود ویدئو
        cursor.execute('SELECT id FROM videos WHERE file_hash = ?', (file_hash,))
        row = cursor.fetchone()
        
        if row:
            self.close()
            return row[0]
        
        # ایجاد رکورد جدید
        cursor.execute('''
            INSERT INTO videos (file_path, file_hash, file_name, duration, fps, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_path,
            file_hash,
            Path(file_path).name,
            metadata.get('duration', 0),
            metadata.get('fps', 0),
            metadata.get('width', 0),
            metadata.get('height', 0)
        ))
        
        video_id = cursor.lastrowid
        self.conn.commit()
        self.close()
        
        return video_id
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """محاسبه SHA256 hash فایل"""
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return ""
    
    # Session operations
    def create_session(self, player_id: int, user_video_id: int,
                      reference_video_id: int) -> int:
        """ایجاد جلسه تحلیل جدید"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (player_id, user_video_id, reference_video_id)
            VALUES (?, ?, ?)
        ''', (player_id, user_video_id, reference_video_id))
        
        session_id = cursor.lastrowid
        self.conn.commit()
        self.close()
        
        return session_id
    
    def update_session_status(self, session_id: int, status: str):
        """به‌روزرسانی وضعیت جلسه"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            UPDATE sessions 
            SET analysis_status = ?, completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (status, session_id))
        
        self.conn.commit()
        self.close()
    
    def get_session(self, session_id: int) -> Optional[Dict]:
        """دریافت اطلاعات جلسه"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
        row = cursor.fetchone()
        
        self.close()
        
        return dict(row) if row else None
    
    def get_all_sessions(self) -> List[Dict]:
        """دریافت تمام جلسات"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT s.*, p.name as player_name,
                   v1.file_name as user_video,
                   v2.file_name as reference_video
            FROM sessions s
            LEFT JOIN players p ON s.player_id = p.id
            LEFT JOIN videos v1 ON s.user_video_id = v1.id
            LEFT JOIN videos v2 ON s.reference_video_id = v2.id
            ORDER BY s.created_at DESC
        ''')
        
        rows = cursor.fetchall()
        self.close()
        
        return [dict(row) for row in rows]
    
    # Stroke operations
    def create_stroke(self, session_id: int, stroke_data: Dict) -> int:
        """ایجاد رکورد ضربه"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO strokes (session_id, stroke_id, start_frame, end_frame,
                                stroke_type, contact_frame, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            stroke_data.get('stroke_id', 0),
            stroke_data.get('start_frame', 0),
            stroke_data.get('end_frame', 0),
            stroke_data.get('stroke_type', 'unknown'),
            stroke_data.get('contact_frame'),
            stroke_data.get('confidence', 1.0)
        ))
        
        stroke_id = cursor.lastrowid
        self.conn.commit()
        self.close()
        
        return stroke_id
    
    # Feature operations
    def create_feature(self, stroke_id: int, frame: int,
                      feature_name: str, feature_value: float,
                      confidence: float = 1.0):
        """ایجاد رکورد ویژگی بیومکانیکی"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO features (stroke_id, frame, feature_name, feature_value, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (stroke_id, frame, feature_name, feature_value, confidence))
        
        self.conn.commit()
        self.close()
    
    # Comparison operations
    def create_comparison(self, session_id: int, comparison_data: Dict):
        """ایجاد رکورد مقایسه"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO comparisons (session_id, stroke_pair, dtw_distance,
                                    similarity_score, alignment_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            session_id,
            comparison_data.get('stroke_pair', ''),
            comparison_data.get('dtw_distance', 0),
            comparison_data.get('similarity_score', 0),
            json.dumps(comparison_data.get('alignment_data', {}))
        ))
        
        self.conn.commit()
        self.close()
    
    # Diagnosis operations
    def create_diagnosis(self, session_id: int, diagnosis_data: Dict):
        """ایجاد رکورد تشخیص"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO diagnoses (session_id, issue_id, category, severity,
                                  description, root_cause, evidence, confidence,
                                  suggested_correction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            diagnosis_data.get('issue_id', ''),
            diagnosis_data.get('category', ''),
            diagnosis_data.get('severity', ''),
            diagnosis_data.get('description', ''),
            diagnosis_data.get('root_cause', ''),
            json.dumps(diagnosis_data.get('evidence', {})),
            diagnosis_data.get('confidence', 0),
            diagnosis_data.get('suggested_correction', '')
        ))
        
        self.conn.commit()
        self.close()
    
    # Coach report operations
    def save_coach_report(self, session_id: int, report_data: Dict):
        """ذخیره گزارش مربی"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO coach_reports 
            (session_id, summary, primary_issue, root_cause, evidence,
             secondary_issues, strengths, corrections, training_plan, next_session_goal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            report_data.get('summary', ''),
            report_data.get('primary_issue', ''),
            report_data.get('root_cause', ''),
            json.dumps(report_data.get('evidence', [])),
            json.dumps(report_data.get('secondary_issues', [])),
            json.dumps(report_data.get('strengths', [])),
            json.dumps(report_data.get('corrections', [])),
            json.dumps(report_data.get('training_plan', [])),
            report_data.get('next_session_goal', '')
        ))
        
        self.conn.commit()
        self.close()
    
    def get_coach_report(self, session_id: int) -> Optional[Dict]:
        """دریافت گزارش مربی"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT * FROM coach_reports WHERE session_id = ?', (session_id,))
        row = cursor.fetchone()
        
        if row:
            result = dict(row)
            # Parse JSON fields
            for field in ['evidence', 'secondary_issues', 'strengths', 'corrections', 'training_plan']:
                if result.get(field):
                    result[field] = json.loads(result[field])
            self.close()
            return result
        
        self.close()
        return None
    
    # Chat operations
    def save_chat_message(self, session_id: int, role: str, content: str,
                         context_used: str = ''):
        """ذخیره پیام چت"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_messages (session_id, role, content, context_used)
            VALUES (?, ?, ?, ?)
        ''', (session_id, role, content, context_used))
        
        self.conn.commit()
        self.close()
    
    def get_chat_history(self, session_id: int) -> List[Dict]:
        """دریافت تاریخچه چت"""
        self.connect()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT * FROM chat_messages 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        ''', (session_id,))
        
        rows = cursor.fetchall()
        self.close()
        
        return [dict(row) for row in rows]
    
    def delete_session(self, session_id: int):
        """حذف جلسه و تمام داده‌های مرتبط"""
        self.connect()
        cursor = self.conn.cursor()
        
        # حذف به ترتیب به خاطر foreign keys
        cursor.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM coach_reports WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM diagnoses WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM comparisons WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM features WHERE stroke_id IN (SELECT id FROM strokes WHERE session_id = ?)', (session_id,))
        cursor.execute('DELETE FROM strokes WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        
        self.conn.commit()
        self.close()
