from pathlib import Path
import ast
import py_compile

APP = Path(__file__).resolve().parent / "app.py"
MARKER = "# OW_RAILWAY_PERSISTENCE_V1_2026_08_26"

text = APP.read_text(encoding="utf-8")
if MARKER in text:
    py_compile.compile(str(APP), doraise=True)
    print("Railway persistence patch already present.")
    raise SystemExit(0)

old = 'DRIVE_DIR = "/content/drive/MyDrive/mlb_engine"\nLOCAL_DIR = "mlb_engine"\n'
new = '''DRIVE_DIR = "/content/drive/MyDrive/mlb_engine"\n# OW_RAILWAY_PERSISTENCE_V1_2026_08_26\n# Data/persistence only. No projection or UI logic changes.\n_ow_explicit_storage = str(os.getenv("MLB_STORAGE_DIR", "") or "").strip()\n_ow_railway_volume = str(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "") or "").strip()\nif _ow_explicit_storage:\n    LOCAL_DIR = _ow_explicit_storage\nelif _ow_railway_volume:\n    LOCAL_DIR = os.path.join(_ow_railway_volume, "mlb_engine")\nelse:\n    LOCAL_DIR = "mlb_engine"\n'''

# Handle CRLF source safely.
normalized = text.replace("\r\n", "\n")
if old not in normalized:
    raise RuntimeError("Could not find the storage block safely; app.py was not changed")
normalized = normalized.replace(old, new, 1)

ast.parse(normalized)
APP.write_text(normalized, encoding="utf-8")
py_compile.compile(str(APP), doraise=True)
print("Applied Railway persistent STORAGE_DIR routing; UI/formulas unchanged.")
