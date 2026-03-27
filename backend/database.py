import sqlite3
import os

class Database:
    def __init__(self, db_path='instance/friction.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user TEXT,
                    text TEXT,
                    ihg REAL,
                    nti REAL,
                    r REAL,
                    error REAL,
                    prediction_id TEXT UNIQUE
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT,
                    outcome REAL,
                    error REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kp REAL,
                    ki REAL,
                    kd REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def save_prediction(self, user, text, ihg, nti, r, prediction_id):
        with self.get_connection() as conn:
            conn.execute(
                'INSERT INTO predictions (user, text, ihg, nti, r, prediction_id) VALUES (?,?,?,?,?,?)',
                (user, text, ihg, nti, r, prediction_id)
            )

    def save_learning(self, prediction_id, outcome, error):
        with self.get_connection() as conn:
            conn.execute(
                'INSERT INTO learning (prediction_id, outcome, error) VALUES (?,?,?)',
                (prediction_id, outcome, error)
            )

    def get_history(self, limit=100):
        with self.get_connection() as conn:
            cur = conn.execute('''
                SELECT timestamp, error FROM predictions
                WHERE error IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            return [{'timestamp': row[0], 'error': row[1]} for row in cur.fetchall()]

    def save_params(self, kp, ki, kd):
        with self.get_connection() as conn:
            conn.execute('INSERT INTO params (kp, ki, kd) VALUES (?,?,?)', (kp, ki, kd))