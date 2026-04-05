import sys
import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication, QAbstractSpinBox


class MicroscopeAnalogControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)

    def _make_window(self):
        from ui_qt.microscope_window import MicroscopeWindow

        win = MicroscopeWindow(samples_dir=Path.cwd())
        self.addCleanup(win.close)
        self.addCleanup(win.deleteLater)
        return win

    def test_objective_dial_is_removed_and_spin_remains(self) -> None:
        win = self._make_window()
        self.assertFalse(hasattr(win, "objective_dial"))
        self.assertTrue(hasattr(win, "objective_spin"))
        self.assertTrue(hasattr(win, "objective_up_btn"))
        self.assertTrue(hasattr(win, "objective_down_btn"))
        self.assertEqual(win.objective_up_btn.text(), "▲")
        self.assertEqual(win.objective_down_btn.text(), "▼")
        self.assertTrue(win.objective_up_btn.autoRepeat())
        self.assertTrue(win.objective_down_btn.autoRepeat())
        self.assertEqual(
            win.objective_spin.buttonSymbols(), QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.assertTrue(win.objective_spin.isAccelerated())
        self.assertFalse(hasattr(win, "objective_slider"))

    def test_focus_dial_is_removed_and_spin_drives_distance(self) -> None:
        win = self._make_window()
        win._set_objective(200)
        self.assertFalse(hasattr(win, "focus_dial"))
        self.assertTrue(hasattr(win, "focus_up_btn"))
        self.assertTrue(hasattr(win, "focus_down_btn"))
        self.assertEqual(win.focus_up_btn.text(), "▲")
        self.assertEqual(win.focus_down_btn.text(), "▼")
        self.assertTrue(win.focus_up_btn.autoRepeat())
        self.assertTrue(win.focus_down_btn.autoRepeat())
        self.assertTrue(win.focus_distance_spin.isAccelerated())
        low, high = win._focus_distance_limits_mm()
        win.focus_distance_spin.setValue(low)
        self.assertAlmostEqual(win._current_focus_distance_mm(), low, places=2)
        win.focus_distance_spin.setValue(high)
        self.assertAlmostEqual(win._current_focus_distance_mm(), high, places=2)

    def test_objective_spin_snaps_to_supported_values(self) -> None:
        win = self._make_window()
        win.objective_spin.setValue(347)

        self.assertEqual(win._current_objective(), 350)
        self.assertEqual(win.objective_spin.value(), 350)

    def test_focus_spin_updates_distance_and_marks_focus_configured(self) -> None:
        win = self._make_window()
        win._set_objective(200)
        low, high = win._focus_distance_limits_mm()
        target = round((low + high) * 0.5, 2)
        win.focus_distance_spin.setValue(target)

        self.assertAlmostEqual(win._current_focus_distance_mm(), target, places=1)
        self.assertAlmostEqual(win.focus_distance_spin.value(), target, places=1)
        self.assertTrue(win.focus_user_configured)

    def test_arrow_buttons_change_objective_and_focus(self) -> None:
        win = self._make_window()
        start_objective = win._current_objective()
        start_focus = win._current_focus_distance_mm()

        win._step_objective(1)
        after_objective_focus = win._current_focus_distance_mm()
        win._step_focus_dial(1)

        self.assertGreater(win._current_objective(), start_objective)
        low, high = win._focus_distance_limits_mm(win._current_objective())
        self.assertGreaterEqual(after_objective_focus, low)
        self.assertLessEqual(after_objective_focus, high)
        self.assertNotEqual(win._current_focus_distance_mm(), after_objective_focus)

    def test_focus_starts_unconfigured_at_lower_bound(self) -> None:
        win = self._make_window()
        low, _high = win._focus_distance_limits_mm()
        self.assertFalse(win.focus_user_configured)
        self.assertAlmostEqual(win._current_focus_distance_mm(), low, places=2)
        self.assertIn("не настроен", win.status_focus_label.text())

    def test_focus_target_depends_only_on_objective_and_xy(self) -> None:
        win = self._make_window()
        win.current_image_path = Path("a.png")
        win.current_source_metadata = {"sample_id": "x", "final_stage": "y"}
        target_a = win._focus_target_mm(300, 0.5, 0.5)
        win.current_image_path = Path("b.png")
        win.current_source_metadata = {"sample_id": "other", "final_stage": "other"}
        target_b = win._focus_target_mm(300, 0.5, 0.5)
        self.assertAlmostEqual(target_a, target_b, places=6)

    def test_focus_cell_has_question_mark_and_tooltip(self) -> None:
        win = self._make_window()
        self.assertEqual(
            win.focus_distance_spin.buttonSymbols(),
            QAbstractSpinBox.ButtonSymbols.NoButtons,
        )
        self.assertTrue(hasattr(win, "focus_help_btn"))
        self.assertEqual(win.focus_help_btn.text(), "?")
        self.assertIn("X - 0.5", win.focus_distance_spin.toolTip())
        self.assertIn("X - 0.5", win.focus_help_btn.toolTip())

    def test_objective_change_requests_immediate_view_update(self) -> None:
        win = self._make_window()
        calls: list[bool] = []

        def fake_queue_view_update(*, immediate: bool = False) -> None:
            calls.append(bool(immediate))

        win._queue_view_update = fake_queue_view_update  # type: ignore[method-assign]
        win._set_objective(300)
        self.assertEqual(calls, [True])

    def test_objective_repeat_release_flushes_single_immediate_render(self) -> None:
        win = self._make_window()
        calls: list[bool] = []

        def fake_queue_view_update(*, immediate: bool = False) -> None:
            calls.append(bool(immediate))

        win._queue_view_update = fake_queue_view_update  # type: ignore[method-assign]
        win._on_objective_button_press()
        win._step_objective(1, immediate=False)
        win._step_objective(1, immediate=False)
        self.assertEqual(calls, [])
        win._on_objective_button_release()
        self.assertEqual(calls, [True])

    def test_instrument_controls_have_safe_widths(self) -> None:
        win = self._make_window()
        self.assertGreaterEqual(win.objective_spin.width(), 132)
        self.assertGreaterEqual(win.focus_distance_spin.width(), 136)
        self.assertGreaterEqual(win.brightness.width(), 104)
        self.assertGreaterEqual(win.contrast.width(), 104)

    def test_left_panel_has_safe_minimum_width(self) -> None:
        win = self._make_window()
        self.assertGreaterEqual(win.left_scroll.minimumWidth(), 430)
