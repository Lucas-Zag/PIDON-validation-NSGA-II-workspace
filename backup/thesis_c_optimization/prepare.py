"""Read-only inspection of siblings. Writes reports only in this project."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys

from cycle import VARIABLES, cycle_timing
from contracts import read_case, validate_case

ROOT = Path(__file__).resolve().parent
EXCLUDED = {".git", ".venv", "venv", "__pycache__", "node_modules", "outputs", "figures"}
PATTERNS = {
    "http": r"\b(requests|httpx|urllib|FastAPI|Flask|uvicorn)\b|https?://|localhost|127\.0\.0\.1",
    "python": r"\b(importlib|sys\.path)\b|from\s+(fe_reference|models\.pidon|utils)\b",
    "command": r"\bsubprocess\b|\bpython[0-9.]*\b.*\.py",
}


def git_head(path):
    # Never read a parent repository if this directory itself is not a checkout.
    if not (path / ".git").exists():
        return None
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def source_files(path):
    if not path.is_dir():
        return []
    result = []
    for f in path.rglob("*"):
        rel = f.relative_to(path)
        if any(p in EXCLUDED for p in rel.parts) or f.is_symlink():
            continue
        if f.is_file() and f.suffix in {".py", ".sh", ".json", ".yaml", ".yml", ".md"}:
            result.append(f)
    return sorted(result)


def inspect(workspace, config):
    report = {"workspace": str(workspace), "stage": "preparation", "projects": {},
              "blockers": [], "integration_claim": "Python call contract confirmed from uploaded scripts; local weights/runtime not yet verified"}
    for role, name in config["siblings"].items():
        path = workspace / name
        info = {"path": str(path), "exists": path.is_dir(), "git_head": git_head(path),
                "source_sha256": {}, "call_evidence": []}
        if not path.is_dir():
            report["blockers"].append(f"Missing sibling: {role} ({path})")
        for f in source_files(path):
            data = f.read_bytes()
            info["source_sha256"][str(f.relative_to(path))] = hashlib.sha256(data).hexdigest()
            if role == "comparison":
                for line_no, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                    kinds = [k for k, pattern in PATTERNS.items() if re.search(pattern, line)]
                    if kinds:
                        # File/line references only; don't copy code, URLs or credentials.
                        info["call_evidence"].append({"file": str(f.relative_to(path)),
                                                       "line": line_no, "kinds": kinds})
        report["projects"][role] = info

    xp = workspace / config["siblings"]["xpidon"]
    model_dir = xp / "outputs" / "models"
    info = report["projects"]["xpidon"]
    info["default_model_directory"] = str(model_dir)
    info["checkpoint_filenames"] = sorted(f.name for f in model_dir.glob("*.pkl"))
    report["blockers"].append("Select one trained run and confirm exact subdomain intervals/config; filenames alone are insufficient.")
    if not info["checkpoint_filenames"]:
        report["blockers"].append("No checkpoints in default outputs/models; identify the actual trained run location.")

    params_path = xp / "exp_params.json"
    if params_path.exists():
        p = json.loads(params_path.read_text())
        info["configured_bounds_not_yet_validated"] = {k: [p[k+"_min"], p[k+"_max"]] for k in VARIABLES}
        info["time_horizon_s"] = p["t_max"]
        info["temperature_normalization_scale_C"] = p["T_max"]
        corners = []
        for values in itertools.product(*info["configured_bounds_not_yet_validated"].values()):
            case = dict(zip(VARIABLES, values))
            timing = cycle_timing(case, float(p["T_ini"]), float(p["t_max"]))
            if not timing["positive_cooling_interval"]:
                corners.append({"case": case, **timing})
        info["invalid_box_corners"] = corners

    comparison = workspace / config["siblings"]["comparison"]
    report['reference_cases'] = []
    phy_path = xp/'phy_params.json'
    if params_path.exists() and phy_path.exists() and (comparison/'cases.csv').exists():
        phy = json.loads(phy_path.read_text())
        for case_id in config['reference_cases']['case_ids']:
            try:
                case = read_case(comparison/'cases.csv',case_id)
                timing = validate_case(case,p,phy)
                expected = config['reference_cases']['fixed']
                changes = {k:case[k] for k,v in expected.items() if abs(float(case[k])-v)>1e-9}
                report['reference_cases'].append({'case':case,'timing':timing,'fixed_condition_changes_since_upload':changes})
                if changes:
                    report['blockers'].append(f'{case_id}: fixed conditions differ from uploaded case; review before optimisation.')
            except (ValueError,KeyError) as e:
                report['blockers'].append(f'{case_id}: {e}')
    report['existing_comparison_results'] = {}
    for name in ('fe_vs_xpidon_metrics.csv','fe_vs_xpidon_metrics_by_position.csv','random_zero_shot_summary.csv'):
        f = comparison/'outputs'/name
        if f.exists():
            with f.open(newline='',encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            selected = [r for r in rows if r.get('case_id') in config['reference_cases']['case_ids']]
            if name == 'random_zero_shot_summary.csv':
                selected = rows
            report['existing_comparison_results'][name] = {'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
                'row_count':len(rows),'reference_rows':selected}
        else:
            report['existing_comparison_results'][name] = {'exists':False}
    report["blockers"].extend([
        "Check numerical comparison errors and confirm a reference case for optimisation from existing results.",
        "Define two objectives and evaluation time; fixed full-cycle horizon cannot serve as a variable time objective.",
        "Provide sourced alpha_min and composite temperature limit, distinct from normalization scales.",
        "Benchmark warm model inference separately from loading/JAX compilation; FE remains final verification only.",
    ])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    if workspace == ROOT or ROOT in workspace.parents:
        parser.error("Workspace must be the parent containing the sibling projects")
    config = json.loads((ROOT / "config/project.json").read_text())
    report = inspect(workspace, config)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "preparation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print("Report:", out / "preparation_report.json")
    for role, project in report["projects"].items():
        print(role, "FOUND" if project["exists"] else "MISSING", project["path"])
    for issue in report["blockers"]:
        print("PENDING:", issue)
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    sys.exit(main())
