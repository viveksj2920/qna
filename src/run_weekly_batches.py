"""
Run the pipeline in weekly batches to avoid slow Azure Search pagination.

Each week fetches fast (~seconds for ~700 records), processes with LLM,
saves to JSON. After all weeks complete, merges results and generates report.

Usage:
  python run_weekly_batches.py --start_date "2026-04-01" --end_date "2026-05-15"
  python run_weekly_batches.py --start_date "2026-04-01" --end_date "2026-05-15" --max_records 1000
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from collections import Counter


SOURCE_INDEX = "transcripts-mira"
DEST_INDEX = "transcript-index-test-qna-v1"
LOOKUP_INDEX = "transcripts-qna-subtopic-groupings-test-1"


def generate_weeks(start_date, end_date):
    """Generate weekly date ranges."""
    weeks = []
    current = start_date
    while current < end_date:
        week_end = min(current + timedelta(days=7), end_date)
        weeks.append((current, week_end))
        current = week_end
    return weeks


def run_week(start, end, max_records, output_dir):
    """Run pipeline for one week."""
    start_str = start.strftime('%Y-%m-%dT00:00:00Z')
    end_str = end.strftime('%Y-%m-%dT00:00:00Z')

    cmd = [
        sys.executable, "main.py",
        "--dry_run",
        "--input_type", "index",
        "--project", "MIRA",
        "--source_data_name", SOURCE_INDEX,
        "--destination_data_name", DEST_INDEX,
        "--lookup_data_name", LOOKUP_INDEX,
        "--start_date", start_str,
        "--end_date", end_str,
        "--max_records", str(max_records),
    ]

    print(f"\n{'='*60}")
    print(f"Running: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: Week exited with code {result.returncode}")
        return None

    # Find the latest output file
    candidates = []
    for f in os.listdir(output_dir):
        if f.startswith(DEST_INDEX) and f.endswith(".json"):
            path = os.path.join(output_dir, f)
            candidates.append((os.path.getmtime(path), path))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    return None


def merge_all(json_files, output_path):
    """Merge all weekly JSONs, deduplicate by Ucid+question."""
    all_data = []
    for f in json_files:
        try:
            with open(f) as fh:
                content = fh.read()
            for chunk in content.split(']['):
                chunk = chunk.strip()
                if not chunk.startswith('['):
                    chunk = '[' + chunk
                if not chunk.endswith(']'):
                    chunk = chunk + ']'
                try:
                    all_data.extend(json.loads(chunk))
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"WARNING: Could not read {f}: {e}")

    # Deduplicate
    seen = {}
    for rec in all_data:
        if not rec.get('question') or not rec.get('topic'):
            continue
        key = (rec.get('Ucid', ''), rec.get('question', '')[:100])
        if key not in seen:
            seen[key] = rec
        else:
            existing = seen[key]
            if not existing.get('sub_topic') and rec.get('sub_topic'):
                seen[key] = rec

    merged = list(seen.values())
    with open(output_path, 'w') as fh:
        json.dump(merged, fh, indent=2)

    print(f"\nMerged {len(merged)} unique records (from {len(all_data)} total) into {output_path}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Run pipeline in weekly batches")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--max_records", type=int, default=1000, help="Max records per week (default: 1000)")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    weeks = generate_weeks(start, end)

    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Weekly batches: {len(weeks)}")
    print(f"Max records per week: {args.max_records}")
    for i, (s, e) in enumerate(weeks, 1):
        print(f"  Week {i}: {s.strftime('%Y-%m-%d')} to {e.strftime('%Y-%m-%d')}")

    json_files = []
    summary = {}

    for i, (week_start, week_end) in enumerate(weeks, 1):
        print(f"\n[{i}/{len(weeks)}]")
        start_time = datetime.now()
        output_file = run_week(week_start, week_end, args.max_records, output_dir)

        if output_file:
            json_files.append(output_file)
            try:
                with open(output_file) as f:
                    content = f.read()
                data = []
                for chunk in content.split(']['):
                    chunk = chunk.strip()
                    if not chunk.startswith('['):
                        chunk = '[' + chunk
                    if not chunk.endswith(']'):
                        chunk = chunk + ']'
                    try:
                        data.extend(json.loads(chunk))
                    except json.JSONDecodeError:
                        continue
                qna_count = sum(1 for r in data if r.get('question') and r.get('topic'))
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                week_label = f"{week_start.strftime('%m/%d')}-{week_end.strftime('%m/%d')}"
                summary[week_label] = {"records": qna_count, "time_min": round(elapsed, 1)}
                print(f"  -> {qna_count} Q&A records ({round(elapsed, 1)} min)")
            except Exception:
                pass
        else:
            week_label = f"{week_start.strftime('%m/%d')}-{week_end.strftime('%m/%d')}"
            summary[week_label] = {"records": 0, "time_min": 0}

    # Print summary
    print(f"\n{'='*60}")
    print("WEEKLY SUMMARY")
    print(f"{'='*60}")
    total_records = 0
    total_time = 0
    for week, s in summary.items():
        count = s.get("records", 0)
        t = s.get("time_min", 0)
        total_records += count if isinstance(count, int) else 0
        total_time += t
        print(f"  {week}: {count} records, {t} min")
    print(f"  TOTAL: {total_records} records, {round(total_time, 1)} min")

    if json_files:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        merged_path = os.path.join(output_dir, f"merged_weekly_{timestamp}.json")
        records = merge_all(json_files, merged_path)

        topics = Counter(r.get('topic', '').lower() for r in records)
        print(f"\nRecords per topic:")
        for t, c in topics.most_common():
            marker = ' OK' if c >= 200 else ''
            print(f"  {t}: {c}{marker}")

        print(f"\nNext step - run subtopic extraction:")
        print(f"  python run_subtopic_only.py --input {merged_path} --topic \"all\" --max_records 200")
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    main()
