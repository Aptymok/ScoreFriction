import sqlite3
from datetime import datetime
import json

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT UNIQUE,
                    timestamp TEXT,
                    user TEXT,
                    text TEXT,
                    ihg REAL,
                    nti REAL,
                    r REAL,
                    error REAL,
                    error_smoothed REAL,
                    params_kp REAL,
                    params_ki REAL,
                    params_kd REAL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS parameters (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS scenarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    scenario TEXT
                )
            ''')

    def save_prediction(self, pred_id, user, text, state, error=None, error_smoothed=None, params=None):
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO predictions (prediction_id, timestamp, user, text, ihg, nti, r, error, error_smoothed, params_kp, params_ki, params_kd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (pred_id, datetime.utcnow().isoformat(), user, text,
                  state.get('ihg'), state.get('nti'), state.get('r'),
                  error, error_smoothed,
                  params.get('kp') if params else None,
                  params.get('ki') if params else None,
                  params.get('kd') if params else None))

    def get_history(self, limit=100):
        with self.get_connection() as conn:
            cur = conn.execute('''
                SELECT timestamp, error, error_smoothed FROM predictions
                WHERE error IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            rows = cur.fetchall()
            return [{'timestamp': r[0], 'error': r[1], 'error_smoothed': r[2]} for r in rows]

    def save_parameters(self, key, value):
        with self.get_connection() as conn:
            conn.execute('REPLACE INTO parameters (key, value) VALUES (?, ?)', (key, json.dumps(value)))

    def get_parameters(self, key):
        with self.get_connection() as conn:
            cur = conn.execute('SELECT value FROM parameters WHERE key = ?', (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
            return None

    def save_scenario(self, scenario):
        with self.get_connection() as conn:
            conn.execute('INSERT INTO scenarios (timestamp, scenario) VALUES (?, ?)', (datetime.utcnow().isoformat(), scenario))