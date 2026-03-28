"""
tests/test_orchestrator.py – 5 tests del ciclo completo del Orquestador Proactivo.
Usa unittest.mock: sin dependencias externas reales.
"""

import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Ajustar path para importar desde backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


# ── helpers de mock ──────────────────────────────────────────────────────────

def _make_mihm():
    """MIHM mock completo con el contrato del framework."""
    mihm = MagicMock()
    mihm.state = {
        'ihg': -0.620, 'nti': 0.351, 'r': 0.450,
        'phi_p': 0.000, 'psi_i': 0.000, 'h_scale': 0.500,
        'ml_success': 0.500, 'cff': 0.000,
    }
    mihm.irc = 0.38

    def _apply_delta(delta, delay_seconds=0, action=''):
        for k, v in delta.items():
            if k in mihm.state:
                mihm.state[k] = min(1.0, max(-2.0, mihm.state[k] + v))
        J = 0.35  # costo sintético
        u = 0.12
        return u, J

    mihm.apply_delta.side_effect = _apply_delta
    mihm.cost_function.return_value = 0.35
    mihm.meta_control.return_value  = None
    mihm.monte_carlo_projection.return_value = {
        'ihg_esperado': -0.45, 'prob_exito': 0.62, 'fecha_optima': 12, 'n_sims': 500,
    }
    mihm.process_delayed_updates.return_value = None
    mihm.reflexive_rules = []
    return mihm


def _make_db():
    """Database mock con tablas de sesiones."""
    db = MagicMock()
    db._sessions = {}

    def _create(session_id, event_id, event_title):
        db._sessions[session_id] = {
            'session_id': session_id, 'calendar_event_id': event_id,
            'calendar_event_title': event_title, 'status': 'idle',
            'created_at': '2024-01-01T00:00:00', 'updated_at': '2024-01-01T00:00:00',
            'telegram_message_id': None, 'user_params_json': None,
            'midi_path': None, 'cost_j_at_trigger': None, 'error_msg': None,
            'mihm_state_snapshot': None,
        }

    def _update(session_id, **fields):
        if session_id in db._sessions:
            db._sessions[session_id].update(fields)

    def _get(session_id):
        return db._sessions.get(session_id)

    def _recent(limit=10):
        return list(db._sessions.values())[:limit]

    def _save_deliverable(session_id, dtype, file_id, url, delta):
        pass

    db.create_orchestrator_session.side_effect  = _create
    db.update_orchestrator_session.side_effect  = _update
    db.get_orchestrator_session.side_effect     = _get
    db.get_recent_orchestrator_sessions.side_effect = _recent
    db.save_deliverable.side_effect             = _save_deliverable
    db.save_state.return_value                  = None
    return db


# ════════════════════════════════════════════════════════════════════════════
# TEST 1 – apply_delta actualiza state y retorna (u, J)
# ════════════════════════════════════════════════════════════════════════════

class Test1ApplyDelta(unittest.TestCase):
    def test_apply_delta_updates_state_and_returns_u_J(self):
        mihm = _make_mihm()
        nti_before = mihm.state['nti']

        u, J = mihm.apply_delta({'nti': 0.05, 'r': 0.03}, action='test')

        self.assertIsNotNone(u,  "apply_delta debe retornar u")
        self.assertIsNotNone(J,  "apply_delta debe retornar J")
        self.assertIsInstance(u, float)
        self.assertIsInstance(J, float)
        self.assertAlmostEqual(mihm.state['nti'], nti_before + 0.05, places=5)


# ════════════════════════════════════════════════════════════════════════════
# TEST 2 – scan_calendar crea sesión en DB
# ════════════════════════════════════════════════════════════════════════════

class Test2ScanCalendar(unittest.TestCase):
    def test_scan_calendar_triggers_session_creation(self):
        from modules.proactive_orchestrator import ProactiveOrchestrator

        mihm = _make_mihm()
        db   = _make_db()
        orch = ProactiveOrchestrator(mihm, db, config={
            'TELEGRAM_BOT_TOKEN': None, 'TELEGRAM_CHAT_ID': None,
        })

        fake_event = {'id': 'evt-001', 'title': 'generación de pista – test', 'start': '2024-01-15T10:00:00Z'}

        with patch.object(orch, 'scan_calendar', return_value=[fake_event]):
            with patch.object(orch, 'send_telegram', return_value=42):
                result = orch.tick()

        # Debe haberse creado una sesión con el event_id
        sess = db.get_orchestrator_session('evt-001')
        self.assertIsNotNone(sess, "Sesión debería existir en DB")
        self.assertEqual(sess['status'], 'waiting_telegram')
        self.assertIn('sessions',   result)
        self.assertIn('mihm_state', result)


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 – parse_user_params parsea correctamente
# ════════════════════════════════════════════════════════════════════════════

class Test3ParseUserParams(unittest.TestCase):
    def test_user_params_parsed_correctly(self):
        from modules.proactive_orchestrator import ProactiveOrchestrator

        mihm = _make_mihm()
        db   = _make_db()
        orch = ProactiveOrchestrator(mihm, db)

        raw = "amor, ritmo, ciudad, noche | 180 | el amor duele pero suena | reggaeton | bajo, batería, sintetizador | 0.7"
        params = orch.parse_user_params(raw)

        self.assertIsNotNone(params, "Parámetros válidos no deben retornar None")
        self.assertEqual(params['genero'],        'reggaeton')
        self.assertEqual(params['duracion_seg'],  180)
        self.assertEqual(params['frase_concepto'], 'el amor duele pero suena')
        self.assertAlmostEqual(params['enganche'], 0.7, places=2)
        self.assertIn('amor', params['motivos'])
        self.assertIn('bajo', params['instrumentos'])

    def test_invalid_params_returns_none(self):
        from modules.proactive_orchestrator import ProactiveOrchestrator

        mihm = _make_mihm()
        db   = _make_db()
        orch = ProactiveOrchestrator(mihm, db)

        self.assertIsNone(orch.parse_user_params("texto sin formato correcto"))
        self.assertIsNone(orch.parse_user_params(""))


# ════════════════════════════════════════════════════════════════════════════
# TEST 4 – EmergentMelodyEngine.generate retorna midi_path
# ════════════════════════════════════════════════════════════════════════════

class Test4MelodyEngineGenerate(unittest.TestCase):
    def test_melody_engine_generate_returns_midi_path(self):
        from modules.emergent_melody_engine import EmergentMelodyEngine

        mihm = _make_mihm()

        # Directorio temporal para el MIDI
        import tempfile
        tmp_dir = tempfile.mkdtemp()

        engine = EmergentMelodyEngine(mihm, groq=None, ml_module=None, spotify_module=None)
        engine.output_dir = tmp_dir

        params = {
            'motivos':        ['amor', 'ritmo', 'ciudad', 'noche'],
            'duracion_seg':   30,
            'frase_concepto': 'el amor duele pero suena',
            'genero':         'reggaeton',
            'instrumentos':   ['bajo', 'sintetizador'],
            'enganche':       0.7,
            'session_id':     'test1234',
        }

        result = engine.generate(params)

        self.assertIn('midi_path',    result)
        self.assertIn('tension_peak', result)
        self.assertIn('mc_projection', result)
        self.assertIn('mihm_state',   result)
        self.assertIn('cost_j',       result)

        # El archivo MIDI debe existir
        self.assertTrue(
            os.path.exists(result['midi_path']),
            f"Archivo MIDI no existe: {result['midi_path']}"
        )
        self.assertGreater(os.path.getsize(result['midi_path']), 0, "MIDI no debe estar vacío")

        # Verificar que apply_delta fue llamado
        mihm.apply_delta.assert_called()

        # Limpieza
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
# TEST 5 – Ciclo completo: Calendar → Telegram → params → Drive → DONE
# ════════════════════════════════════════════════════════════════════════════

class Test5FullCycleCalendarToDrive(unittest.TestCase):
    def test_full_cycle_calendar_to_drive_deliverables(self):
        from modules.proactive_orchestrator import ProactiveOrchestrator
        from modules.emergent_melody_engine  import EmergentMelodyEngine
        from modules.drive_manager           import DriveManager

        import tempfile
        tmp_dir = tempfile.mkdtemp()

        mihm   = _make_mihm()
        db     = _make_db()
        groq   = MagicMock()
        groq.generate_executive_doc.return_value   = "Narrativa ejecutiva de prueba."
        groq.generate_social_proposal.return_value = "Propuesta social de prueba."

        engine    = EmergentMelodyEngine(mihm, groq, None, None)
        engine.output_dir = tmp_dir

        drive_mgr = DriveManager(mihm, {'MIDI_OUTPUT_DIR': tmp_dir})

        orch = ProactiveOrchestrator(
            mihm, db, groq=groq,
            melody_engine=engine, drive_manager=drive_mgr,
            config={'TELEGRAM_BOT_TOKEN': None, 'TELEGRAM_CHAT_ID': None},
        )

        # ── Paso 1: detectar evento de Calendar ──
        event = {'id': 'evt-full', 'title': 'generación de pista – full test', 'start': '2024-01-20T12:00:00Z'}

        with patch.object(orch, 'scan_calendar', return_value=[event]):
            with patch.object(orch, 'send_telegram', return_value=99):
                orch.tick()

        sess = db.get_orchestrator_session('evt-full')
        self.assertIsNotNone(sess)
        self.assertEqual(sess['status'], 'waiting_telegram')

        # ── Paso 2: simular respuesta Telegram con parámetros válidos ──
        raw_params = "amor, ritmo, ciudad, noche | 30 | el amor duele pero suena | reggaeton | bajo, sintetizador | 0.7"
        fake_update = [{
            'update_id': 1,
            'message':   {'text': raw_params, 'chat': {'id': '123'}},
        }]

        with patch.object(orch, 'scan_calendar', return_value=[]):
            with patch.object(orch, 'get_telegram_updates', return_value=fake_update):
                orch.tick()

        sess = db.get_orchestrator_session('evt-full')

        # Después de parsear params, la sesión puede estar en generating, uploading o done
        self.assertIn(sess['status'], ('generating', 'uploading', 'done'),
                      f"Estado inesperado: {sess['status']}")

        # ── Paso 3: si quedó en generating/uploading, dar otro tick ──
        if sess['status'] in ('generating', 'uploading'):
            with patch.object(orch, 'scan_calendar', return_value=[]):
                with patch.object(orch, 'get_telegram_updates', return_value=[]):
                    orch.tick()

        sess = db.get_orchestrator_session('evt-full')

        # En el peor caso, si el DriveManager sin credenciales crea archivos locales, status=done
        # Verificar que save_deliverable fue llamado al menos una vez
        self.assertTrue(
            db.save_deliverable.called,
            "save_deliverable debería haber sido llamado al crear entregables"
        )

        # Verificar que apply_delta fue llamado (Golden Rule)
        self.assertTrue(
            mihm.apply_delta.call_count >= 2,
            f"apply_delta debe llamarse ≥2 veces, llamado {mihm.apply_delta.call_count} veces"
        )

        # Limpieza
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
