# GUI rules (`flextool/gui/`)

- Never open Tk windows on the owner's live display: run anything that creates a Tk root (including `tests/gui`)
  under `xvfb-run -a`.
- xvfb maps windows faster than a real window manager, so timing bugs (grab, focus, map order) don't reproduce
  headless.  Example: `grab_set()` on a Toplevel not yet viewable raises `TclError` out of the constructor and the
  dialog silently does nothing; defer it (`try: self.grab_set() except tk.TclError: self.after(20, ...)`, as
  `ColorPickerDialog` in `dialogs/plot_settings_picker.py`).
- No native message boxes (`tests/gui/test_no_native_messagebox.py`).
