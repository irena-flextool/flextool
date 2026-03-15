"""Smoke test for GUI startup.

Verifies that the MainWindow can be created and destroyed without errors.
Requires a display (or Xvfb on headless CI).
"""

import unittest


class TestGuiStartup(unittest.TestCase):
    """Test that the GUI starts up without crashing."""

    def test_main_window_creates_and_destroys(self) -> None:
        """MainWindow should initialize all widgets and shut down cleanly."""
        import tkinter as tk

        # Verify we can create a Tk root (will fail on headless without Xvfb)
        try:
            test_root = tk.Tk()
            test_root.destroy()
        except tk.TclError:
            self.skipTest("No display available")

        from flextool.gui.main_window import MainWindow

        root = MainWindow(initial_theme="dark")
        try:
            # Process pending events so all widgets are realized
            root.update_idletasks()
            root.update()

            # Basic sanity checks
            self.assertIsNotNone(root.project_combo)
            self.assertIsNotNone(root.input_sources_tree)
            self.assertIsNotNone(root.available_tree)
            self.assertIsNotNone(root.executed_tree)
            self.assertIsNotNone(root.output_frame)
        finally:
            # Cancel any pending after() timers before destroying
            try:
                if root._lock_check_timer_id is not None:
                    root.after_cancel(root._lock_check_timer_id)
                    root._lock_check_timer_id = None
            except Exception:
                pass
            root.destroy()

    def test_main_window_light_theme(self) -> None:
        """MainWindow should also work with light theme."""
        import tkinter as tk

        try:
            test_root = tk.Tk()
            test_root.destroy()
        except tk.TclError:
            self.skipTest("No display available")

        from flextool.gui.main_window import MainWindow

        root = MainWindow(initial_theme="light")
        try:
            root.update_idletasks()
            root.update()
        finally:
            try:
                if root._lock_check_timer_id is not None:
                    root.after_cancel(root._lock_check_timer_id)
                    root._lock_check_timer_id = None
            except Exception:
                pass
            root.destroy()


if __name__ == "__main__":
    unittest.main()
