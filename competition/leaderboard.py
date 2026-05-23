"""
Display and export the competition leaderboard.

Usage:
    python competition/leaderboard.py
    python competition/leaderboard.py --detailed
    python competition/leaderboard.py --export html --output leaderboard.html
"""

import json
import argparse
from pathlib import Path


def display(leaderboard_path: Path, detailed: bool = False):
    if not leaderboard_path.exists():
        print("No submissions yet.")
        return

    lb = json.loads(leaderboard_path.read_text())
    if not lb:
        print("No submissions yet.")
        return

    print(f"\n{'='*55}")
    print(f"  SIMPLEFOLD HACKATHON LEADERBOARD")
    print(f"{'='*55}")
    print(f"  {'Rank':<6}{'Team':<25}{'lDDT':<10}{'Scored':<8}")
    print(f"  {'-'*49}")
    for entry in lb:
        print(f"  {entry['rank']:<6}{entry['team']:<25}{entry['mean_lddt']:<10.4f}{entry['num_scored']:<8}")
    print(f"{'='*55}\n")

    if detailed:
        runs_dir = leaderboard_path.parent / "runs"
        for entry in lb:
            results_path = runs_dir / entry["team"] / "results.json"
            if results_path.exists():
                results = json.loads(results_path.read_text())
                print(f"  {entry['team']}:")
                for pid, r in sorted(results.get("per_protein", {}).items()):
                    lddt = r.get("lddt", 0.0)
                    err = f" ({r['error']})" if "error" in r else ""
                    print(f"    {pid}: {lddt:.4f}{err}")
                print()


def export_html(leaderboard_path: Path, output_path: Path):
    if not leaderboard_path.exists():
        print("No submissions yet.")
        return

    lb = json.loads(leaderboard_path.read_text())

    rows = ""
    for entry in lb:
        lddt = entry["mean_lddt"]
        if lddt >= 0.7:
            color = "#22c55e"
        elif lddt >= 0.4:
            color = "#eab308"
        else:
            color = "#ef4444"
        rows += f"""
        <tr>
            <td>{entry['rank']}</td>
            <td><strong>{entry['team']}</strong></td>
            <td style="color: {color}; font-weight: bold;">{lddt:.4f}</td>
            <td>{entry['num_scored']}</td>
            <td>{entry['timestamp'][:16]}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SimpleFold Hackathon</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 40px; }}
        h1 {{ text-align: center; font-size: 2.5em; margin-bottom: 8px; }}
        .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 40px; }}
        table {{ width: 100%; max-width: 800px; margin: 0 auto; border-collapse: collapse; }}
        th {{ background: #1e293b; padding: 14px 20px; text-align: left; font-size: 0.9em;
              text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; }}
        td {{ padding: 14px 20px; border-bottom: 1px solid #1e293b; font-size: 1.1em; }}
        tr:first-child td {{ font-size: 1.3em; }}
        tr:hover {{ background: #1e293b; }}
    </style>
</head>
<body>
    <h1>SimpleFold Hackathon</h1>
    <p class="subtitle">Scored by mean lDDT on held-out test proteins &middot; auto-refreshes every 30s</p>
    <table>
        <tr><th>Rank</th><th>Team</th><th>Mean lDDT</th><th>Proteins</th><th>Submitted</th></tr>
        {rows}
    </table>
</body>
</html>"""

    output_path.write_text(html)
    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--export", choices=["html", "csv"])
    parser.add_argument("--output", type=str, default="competition/leaderboard.html")
    args = parser.parse_args()

    lb_path = Path(__file__).parent / "leaderboard.json"

    if args.export == "html":
        export_html(lb_path, Path(args.output))
    elif args.export == "csv":
        lb = json.loads(lb_path.read_text()) if lb_path.exists() else []
        lines = ["rank,team,mean_lddt,num_scored,timestamp"]
        for e in lb:
            lines.append(f"{e['rank']},{e['team']},{e['mean_lddt']},{e['num_scored']},{e['timestamp']}")
        Path(args.output).write_text("\n".join(lines))
        print(f"Exported to {args.output}")
    else:
        display(lb_path, args.detailed)


if __name__ == "__main__":
    main()
