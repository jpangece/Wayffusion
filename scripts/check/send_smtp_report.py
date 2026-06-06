from __future__ import annotations

import argparse
import json
import os
import smtplib
import ssl
import subprocess
import sys
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_ENV = [
    "WAYFFUSION_SMTP_HOST",
    "WAYFFUSION_SMTP_PORT",
    "WAYFFUSION_SMTP_USER",
    "WAYFFUSION_SMTP_PASSWORD",
    "WAYFFUSION_SMTP_FROM",
    "WAYFFUSION_SMTP_TO",
    "WAYFFUSION_SMTP_USE_TLS",
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"SMTP env file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def apply_smtp_aliases() -> None:
    aliases = {
        "WAYFFUSION_SMTP_HOST": "SMTP_HOST",
        "WAYFFUSION_SMTP_PORT": "SMTP_PORT",
        "WAYFFUSION_SMTP_USER": "SMTP_USER",
        "WAYFFUSION_SMTP_PASSWORD": "SMTP_PASSWORD",
        "WAYFFUSION_SMTP_FROM": "SMTP_FROM",
        "WAYFFUSION_SMTP_TO": "EMAIL_TO",
    }
    for target, source in aliases.items():
        if not os.environ.get(target) and os.environ.get(source):
            os.environ[target] = os.environ[source]
    if not os.environ.get("WAYFFUSION_SMTP_USE_TLS"):
        if os.environ.get("SMTP_STARTTLS") is not None:
            os.environ["WAYFFUSION_SMTP_USE_TLS"] = "true" if _truthy(os.environ.get("SMTP_STARTTLS")) else "false"
        elif os.environ.get("SMTP_SSL") is not None:
            os.environ["WAYFFUSION_SMTP_USE_TLS"] = "false" if _truthy(os.environ.get("SMTP_SSL")) else "true"


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(report_md: Path, summary_json: Path) -> str:
    if report_md.exists():
        return report_md.read_text(encoding="utf-8")
    output_root = report_md.parent
    summary = load_json(summary_json)
    source = load_json(output_root / "mpe_core_source_check" / "source_check_summary.json")
    calibration = load_json(output_root / "adapter_calibration" / "local_core_velocity_tracker_v1" / "adapter_calibration_summary.json")
    lines = [
        "# Wayffusion Debug Report",
        "",
        f"- git_branch: `{git_value(['branch', '--show-current'])}`",
        f"- git_commit: `{git_value(['rev-parse', '--short', 'HEAD'])}`",
        f"- mpe_source: `{source.get('mpe_source', '')}`",
        f"- uses_real_mpe_core: `{source.get('uses_real_mpe_core', False)}`",
        f"- source_check_passed: `{source.get('passed', False)}`",
        f"- adapter_calibration_passed: `{calibration.get('passed', False)}`",
        f"- mappo_health_passed: `{summary.get('passed', False)}`",
        f"- output_dir: `{output_root}`",
        "",
        "## Adapter Calibration",
        "",
        f"- calibration_failed: `{calibration.get('calibration_failed', True)}`",
        f"- num_settings: `{calibration.get('num_settings', 0)}`",
        f"- num_passed: `{calibration.get('num_passed', 0)}`",
        "- selected_profiles: none" if not calibration.get("selected_profiles") else "- selected_profiles: see below",
        "",
    ]
    for name, profile in calibration.get("selected_profiles", {}).items():
        lines.append(f"### Adapter {name}")
        for key, value in profile.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    if calibration.get("top_rows"):
        lines.extend(["## Top Calibration Rows", ""])
        for row in calibration.get("top_rows", [])[:5]:
            lines.append(
                "- rank `{rank}` score `{score}` failed `{failures}` params "
                "`max_command_distance={mcd}, slowdown_radius={sr}, velocity_gain={vg}, max_accel={ma}, control_smoothing={cs}, acceptance_radius={ar}`".format(
                    rank=row.get("rank", ""),
                    score=row.get("score", ""),
                    failures=row.get("failed_constraints", ""),
                    mcd=row.get("max_command_distance", ""),
                    sr=row.get("slowdown_radius", ""),
                    vg=row.get("velocity_gain", ""),
                    ma=row.get("max_accel", ""),
                    cs=row.get("control_smoothing", ""),
                    ar=row.get("acceptance_radius", ""),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## MAPPO Runs",
        "",
        ]
    )
    for run in summary.get("runs", []):
        lines.extend(
            [
                f"### {run.get('run_name', 'unknown')}",
                f"- passed: `{run.get('passed', False)}`",
                f"- output_dir: `{run.get('output_dir', '')}`",
                f"- policy_class: `{run.get('policy_class', '')}`",
                f"- tasks: `{run.get('tasks', '')}`",
                f"- final_eval_reward: `{run.get('final_eval_reward', '')}`",
                f"- best_eval_reward: `{run.get('best_eval_reward', '')}`",
                f"- action_validity_rate: `{run.get('action_validity_rate', '')}`",
                f"- max_speed_observed: `{run.get('max_speed_observed', '')}`",
                f"- mean_control_norm: `{run.get('mean_control_norm', '')}`",
                f"- collision_count: `{run.get('collision_count', '')}`",
                f"- connectivity_violation_rate: `{run.get('connectivity_violation_rate', '')}`",
                f"- mpe_source: `{run.get('mpe_source', '')}`",
                f"- mpe_world_step_calls: `{run.get('mpe_world_step_calls', '')}`",
                f"- failure_reason: `{run.get('failure_reason', '')}`",
                f"- gif_paths: `{run.get('gif_paths', '[]')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- Adapter calibration did not pass hard constraints in the full 80-setting run, so `configs/env/waypoint_missions_tuned.yaml` was not kept as an active tuned config.",
            "- MAPPO health checks used the baseline adapter config and validate rollout/update/eval/media health, not final learning performance.",
            "- Vendored OpenAI MPE license status remains a warning before external redistribution; see `docs/third_party_mpe_source.md`.",
        ]
    )
    text = "\n".join(lines) + "\n"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-md", default=str(ROOT / "outputs" / "debug" / "final_debug_report.md"))
    parser.add_argument("--summary-json", default=str(ROOT / "outputs" / "debug" / "mappo_health_check" / "mappo_health_summary.json"))
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--fail-if-missing-env", action="store_true")
    parser.add_argument("--no-fail-if-missing-env", action="store_true")
    args = parser.parse_args()

    if args.env_file:
        env_file = Path(args.env_file)
        if not env_file.is_absolute():
            env_file = ROOT / env_file
        load_env_file(env_file)
    apply_smtp_aliases()

    report_md = Path(args.report_md)
    summary_json = Path(args.summary_json)
    if not report_md.is_absolute():
        report_md = ROOT / report_md
    if not summary_json.is_absolute():
        summary_json = ROOT / summary_json
    body = build_report(report_md, summary_json)

    missing = [key for key in REQUIRED_ENV if not os.environ.get(key)]
    if missing:
        notice = {
            "sent": False,
            "reason": "missing_smtp_environment",
            "missing_env": missing,
            "report_md": str(report_md),
            "summary_json": str(summary_json),
        }
        path = ROOT / "outputs" / "debug" / "email_not_sent_missing_env.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notice, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(notice, indent=2, sort_keys=True))
        if args.fail_if_missing_env and not args.no_fail_if_missing_env:
            raise SystemExit(1)
        return

    host = os.environ["WAYFFUSION_SMTP_HOST"]
    port = int(os.environ["WAYFFUSION_SMTP_PORT"])
    user = os.environ["WAYFFUSION_SMTP_USER"]
    password = os.environ["WAYFFUSION_SMTP_PASSWORD"]
    sender = os.environ["WAYFFUSION_SMTP_FROM"]
    recipients = [item.strip() for item in os.environ["WAYFFUSION_SMTP_TO"].split(",") if item.strip()]
    use_tls = os.environ["WAYFFUSION_SMTP_USE_TLS"].strip().lower() in {"1", "true", "yes", "on"}
    subject_prefix = os.environ.get("WAYFFUSION_SMTP_SUBJECT_PREFIX", "")
    subject = f"{subject_prefix}[Wayffusion] Adapter calibration and MAPPO health check completed"

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(body)

    try:
        if use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(host, port, timeout=30) as smtp:
                smtp.starttls(context=context)
                smtp.login(user, password)
                smtp.send_message(message)
        else:
            with smtplib.SMTP_SSL(host, port, timeout=30) as smtp:
                smtp.login(user, password)
                smtp.send_message(message)
    except Exception as exc:
        status = {
            "sent": False,
            "reason": "smtp_send_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "report_md": str(report_md),
        }
        (ROOT / "outputs" / "debug" / "email_send_failed.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(status, indent=2, sort_keys=True))
        raise SystemExit(1)

    status = {"sent": True, "to": recipients, "report_md": str(report_md), "summary_json": str(summary_json)}
    (ROOT / "outputs" / "debug" / "email_sent.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
