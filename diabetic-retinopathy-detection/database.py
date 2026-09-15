import sqlite3
import os
import json
from contextlib import closing
from datetime import datetime
import uuid

DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), 'netra.db'))

def init_db():
    """Initialize the database schema."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Patients Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age TEXT,
                dob TEXT,
                contact TEXT,
                centre TEXT,
                sex TEXT,
                diabetes_type TEXT,
                diabetes_duration TEXT,
                previous_screening_date TEXT,
                vision_complaints TEXT,
                clinical_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Visits Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                visit_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                review_status TEXT DEFAULT 'Review pending',
                referral_status TEXT,
                destination_centre TEXT,
                followup_date TEXT,
                followup_status TEXT DEFAULT 'Not scheduled',
                followup_notes TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
            )
        ''')
        
        # Eye Screenings Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS eye_screenings (
                id TEXT PRIMARY KEY,
                visit_id TEXT NOT NULL,
                eye TEXT NOT NULL, -- 'Left' or 'Right'
                image_path TEXT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                quality_score REAL,
                quality_pass BOOLEAN,
                quality_reason TEXT,
                ai_grade INTEGER,
                ai_raw_confidence REAL,
                ai_calibrated_confidence REAL,
                ai_confidence_flag TEXT,
                model_version TEXT,
                lesion_count INTEGER,
                clinician_grade TEXT, -- '0', '1', '2', '3', '4', 'ungradable', 'unable_to_assess'
                override_reason TEXT,
                clinician_notes TEXT,
                reviewer_id TEXT,
                review_timestamp TIMESTAMP,
                original_b64_path TEXT,
                gradcam_b64_path TEXT,
                lesion_overlay_b64_path TEXT,
                FOREIGN KEY(visit_id) REFERENCES visits(visit_id)
            )
        ''')
        conn.commit()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Database Helper Functions

def add_patient(data):
    # Auto-generate a readable patient ID e.g., P-2410-001
    patient_id = data.get("patient_id") or f"P-{datetime.now().strftime('%y%m')}-{uuid.uuid4().hex[:4].upper()}"
    with closing(get_db()) as conn:
        conn.execute('''
            INSERT INTO patients (patient_id, name, age, dob, contact, centre, sex, diabetes_type, diabetes_duration, previous_screening_date, vision_complaints, clinical_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id, data.get('name'), data.get('age'), data.get('dob'), data.get('contact'),
            data.get('centre'), data.get('sex'), data.get('diabetes_type'), data.get('diabetes_duration'),
            data.get('previous_screening_date'), data.get('vision_complaints'), data.get('clinical_notes')
        ))
        conn.commit()
    return patient_id

def get_patients():
    with closing(get_db()) as conn:
        rows = conn.execute('SELECT * FROM patients ORDER BY created_at DESC').fetchall()
        return [dict(row) for row in rows]

def get_patient(patient_id):
    with closing(get_db()) as conn:
        row = conn.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,)).fetchone()
        return dict(row) if row else None

def get_patient_visits(patient_id):
    with closing(get_db()) as conn:
        rows = conn.execute('SELECT * FROM visits WHERE patient_id = ? ORDER BY date DESC', (patient_id,)).fetchall()
        visits = []
        for row in rows:
            v = dict(row)
            eyes = conn.execute('SELECT * FROM eye_screenings WHERE visit_id = ?', (v['visit_id'],)).fetchall()
            v['screenings'] = {e['eye']: dict(e) for e in eyes}
            visits.append(v)
        return visits

def create_visit(patient_id):
    visit_id = f"V-{uuid.uuid4().hex[:8].upper()}"
    with closing(get_db()) as conn:
        conn.execute('INSERT INTO visits (visit_id, patient_id) VALUES (?, ?)', (visit_id, patient_id))
        conn.commit()
    return visit_id

def get_visit(visit_id):
    with closing(get_db()) as conn:
        v_row = conn.execute('SELECT * FROM visits WHERE visit_id = ?', (visit_id,)).fetchone()
        if not v_row: return None
        v = dict(v_row)
        eyes = conn.execute('SELECT * FROM eye_screenings WHERE visit_id = ?', (visit_id,)).fetchall()
        v['screenings'] = {e['eye']: dict(e) for e in eyes}
        return v

def save_eye_screening(visit_id, eye, data):
    screening_id = f"S-{uuid.uuid4().hex[:8].upper()}"
    with closing(get_db()) as conn:
        # Check if screening already exists for this eye in this visit, delete if it does to overwrite
        conn.execute('DELETE FROM eye_screenings WHERE visit_id = ? AND eye = ?', (visit_id, eye))
        
        conn.execute('''
            INSERT INTO eye_screenings (
                id, visit_id, eye, image_path, quality_score, quality_pass, quality_reason, 
                ai_grade, ai_raw_confidence, ai_calibrated_confidence, ai_confidence_flag, 
                model_version, lesion_count, original_b64_path, gradcam_b64_path, lesion_overlay_b64_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            screening_id, visit_id, eye, data.get('image_path'), data.get('quality_score'), 
            data.get('quality_pass'), data.get('quality_reason'), data.get('ai_grade'),
            data.get('ai_raw_confidence'), data.get('ai_calibrated_confidence'), data.get('ai_confidence_flag'),
            data.get('model_version'), data.get('lesion_count'), data.get('original_b64_path'),
            data.get('gradcam_b64_path'), data.get('lesion_overlay_b64_path')
        ))
        conn.commit()
    return screening_id

def update_visit_review(visit_id, data):
    with closing(get_db()) as conn:
        # Update visit
        conn.execute('''
            UPDATE visits 
            SET review_status = 'Reviewed', referral_status = ?, destination_centre = ?, 
                followup_date = ?, followup_status = ?, followup_notes = ?
            WHERE visit_id = ?
        ''', (
            data.get('referral_status'), data.get('destination_centre'), data.get('followup_date'),
            data.get('followup_status'), data.get('followup_notes'), visit_id
        ))
        
        # Update screenings if provided
        screenings = data.get('screenings', {})
        for eye, s_data in screenings.items():
            conn.execute('''
                UPDATE eye_screenings
                SET clinician_grade = ?, override_reason = ?, clinician_notes = ?,
                    reviewer_id = ?, review_timestamp = CURRENT_TIMESTAMP
                WHERE visit_id = ? AND eye = ?
            ''', (
                s_data.get('clinician_grade'), s_data.get('override_reason'), s_data.get('clinician_notes'),
                data.get('reviewer_id', 'Dr. Demo'), visit_id, eye
            ))
            
        conn.commit()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
