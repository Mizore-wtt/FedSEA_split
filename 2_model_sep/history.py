"""Optional stage-1 references; live full/split comparisons never require old runs."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fedsea.project import project_path


def validate_selection(selection):
    if selection is not None and (not isinstance(selection, str) or not selection.strip()):
        raise ValueError("stage1_run must be null, 'none', 'auto', or a stage-1 run directory.")


def validate_record(record):
    if not isinstance(record, dict) or record.get("stage") != "1_model_base":
        raise ValueError("Expected a stage-1 run.json, not another stage's report.")
    if not isinstance(record.get("model"), dict) or not isinstance(record.get("config"), dict):
        raise ValueError("Historical run needs model and config objects.")
    profile = record.get("model_profile", {})
    if not isinstance(profile, dict) or not isinstance(profile.get("chat_template_kwargs", {}), dict):
        raise ValueError("Historical model_profile and chat_template_kwargs must be objects.")
    results = record.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Historical run has no completed cases.")
    ids = set()
    for row in results:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not isinstance(row.get("messages"), list) or not row["messages"]):
            raise ValueError("Historical cases need an id and messages.")
        if row["id"] in ids:
            raise ValueError("Historical case IDs must be unique.")
        ids.add(row["id"])


def load_reference(selection, profile, *, root=ROOT):
    """Return (history, audit metadata); only explicit paths make missing files fatal."""
    validate_selection(selection)
    info = {"selection": selection, "path": None, "status": "skipped", "reason": ""}
    if selection in (None, "none"):
        info["reason"] = "Historical comparison disabled; comparing against the live full model."
        return None, info
    automatic = selection == "auto"
    location = profile.settings.get("baseline_run") if automatic else selection
    if location is None:
        info["reason"] = "No baseline_run registered; comparing against the live full model."
        return None, info
    validate_selection(location)
    directory = project_path(location, root=root)
    runs_root = (Path(root) / "1_model_base/runs").resolve()
    if directory == runs_root or not directory.is_relative_to(runs_root):
        raise ValueError("Historical references must select one directory under 1_model_base/runs/.")
    info["path"] = directory.relative_to(Path(root).resolve()).as_posix()
    filename = directory / "run.json"
    try:
        raw = filename.read_bytes()
        record = json.loads(raw.decode("utf-8-sig"))
        validate_record(record)
    except (OSError, UnicodeError, ValueError) as error:
        # Auto is best-effort, but a user-requested comparison must not silently disappear.
        reason = f"Historical reference unavailable: {filename} ({type(error).__name__})."
        if not automatic:
            exception = FileNotFoundError if isinstance(error, FileNotFoundError) else ValueError
            raise exception(
                reason + " Choose an existing run, or use --stage1-run none for live-only comparison."
            ) from error
        info["reason"] = reason + " Continuing with the live full model; no alternate run was selected."
        return None, info
    info.update(
        status="loaded", reason="Historical report loaded; each compatible case still needs comparison.",
        run_json_sha256=hashlib.sha256(raw).hexdigest(),
    )
    return (directory, record), info
