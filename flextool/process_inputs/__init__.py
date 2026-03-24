"""Input data reading: converts CSV/Excel/ODS files to Spine database format."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from flextool.process_inputs.read_tabular_with_specification import TabularReader
from flextool.process_inputs.write_to_input_db import write_to_flextool_input_db
from flextool.update_flextool import FLEXTOOL_DB_VERSION as CURRENT_FLEXTOOL_DB_VERSION

__all__ = [
    'TabularReader', 'write_to_flextool_input_db',
    'ExcelFormat', 'ExcelFormatInfo', 'detect_excel_format',
    'CURRENT_FLEXTOOL_DB_VERSION',
]

logger = logging.getLogger(__name__)


class ExcelFormat(Enum):
    """Excel input format variants recognised by FlexTool."""
    SELF_DESCRIBING = "self_describing"   # New format — metadata embedded in sheets
    SPECIFICATION = "specification"       # FlexTool 3.x — needs import_excel_input.json
    OLD_V2 = "old_v2"                     # FlexTool 2.x .xlsm — completely different layout
    UNKNOWN = "unknown"


@dataclass
class ExcelFormatInfo:
    """Result of :func:`detect_excel_format`."""
    format: ExcelFormat
    version: int | None = None  # DB schema version found in the file (None = unknown)


def detect_excel_format(file_path: str | Path) -> ExcelFormatInfo:
    """Detect which FlexTool Excel format *file_path* uses.

    Returns an :class:`ExcelFormatInfo` with the detected format and,
    where possible, the database schema version embedded in the file.

    Detection logic:

    * **SELF_DESCRIBING** — ``version`` sheet, cell A1 starts with
      ``"Generated from FlexTool sqlite version"``.  The version number
      is parsed from that same cell.
    * **SPECIFICATION** — ``version`` sheet, row 3 contains column
      headers ``Version number, Date, Author, Description``
      (FlexTool 3.x format that needs ``import_excel_input.json``).
      The version is always 25 (the schema these files target).
    * **OLD_V2** — file has a ``master`` sheet (FlexTool 2.x).
    * **UNKNOWN** — none of the above.
    """
    import openpyxl

    file_path = Path(file_path)
    try:
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    except Exception:
        logger.warning("Cannot open '%s' as an Excel file.", file_path)
        return ExcelFormatInfo(ExcelFormat.UNKNOWN)

    try:
        sheet_names_lower = {s.lower(): s for s in wb.sheetnames}

        # --- Check version sheet ---
        if "version" in sheet_names_lower:
            ws = wb[sheet_names_lower["version"]]
            rows = list(ws.iter_rows(max_row=4, values_only=True))

            # New self-describing: first cell starts with
            # "Generated from FlexTool sqlite version: <N>"
            if rows and rows[0] and rows[0][0]:
                first_cell = str(rows[0][0]).strip()
                if first_cell.lower().startswith("generated from flextool sqlite"):
                    version: int | None = None
                    m = re.search(r"version:\s*(\d+)", first_cell)
                    if m:
                        version = int(m.group(1))
                    logger.info(
                        "Detected SELF_DESCRIBING format (version %s): %s",
                        version, file_path.name,
                    )
                    return ExcelFormatInfo(ExcelFormat.SELF_DESCRIBING, version)

            # Old 3.x: row 2 (0-indexed) has "Version number", "Date", …
            if len(rows) >= 3 and rows[2]:
                header = [str(c).strip().lower() if c else "" for c in rows[2]]
                if "version number" in header and "date" in header:
                    # Specification-format files always target the v25 schema
                    logger.info(
                        "Detected SPECIFICATION (3.x) format: %s", file_path.name,
                    )
                    return ExcelFormatInfo(ExcelFormat.SPECIFICATION, 25)

        # --- Check for old 2.x .xlsm (has 'master' sheet) ---
        if "master" in sheet_names_lower:
            logger.info("Detected OLD_V2 format: %s", file_path.name)
            return ExcelFormatInfo(ExcelFormat.OLD_V2)

        logger.warning("Could not detect Excel format for '%s'.", file_path.name)
        return ExcelFormatInfo(ExcelFormat.UNKNOWN)

    finally:
        wb.close()
