"""
Automated topic-by-topic subtopic testing.

Runs the pipeline for each topic with --topic_filter, then merges
all results and generates a combined Excel report.

Usage:
  python run_topic_test.py --start_date "2026-01-01T00:00:00Z" --end_date "2026-05-06T00:00:00Z" --max_records 8000
  python run_topic_test.py --start_date "2026-01-01T00:00:00Z" --end_date "2026-05-06T00:00:00Z" --max_records 8000 --topics "enrollment,eligibility,dental"
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


TOPICS = [
    "dental",
    "digital friction",
    "drug coverage termination",
    "eligibility",
    "enrollment",
    "general plan information",
    "medicare id",
    "member renewal",
    "others",
    "plan changes",
    "plan comparison",
    "plan costs",
    "prescription drug coverage",
    "providers",
    "service area reduction",
    "transportation, food and gym benefits",
    "vision",
]

SOURCE_INDEX = "transcripts-mira"
DEST_INDEX = "transcript-index-test-qna-v1"
LOOKUP_INDEX = "transcripts-qna-subtopic-groupings-test-1"


def run_topic(topic, start_date, end_date, max_records, output_dir):
    """Run the pipeline for a single topic and return the output JSON path."""
    safe_name = topic.replace(" ", "_").replace(",", "")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"topic_{safe_name}_{timestamp}.json")

    cmd = [
        sys.executable, "main.py",
        "--dry_run",
        "--input_type", "index",
        "--project", "MIRA",
        "--source_data_name", SOURCE_INDEX,
        "--destination_data_name", DEST_INDEX,
        "--lookup_data_name", LOOKUP_INDEX,
        "--start_date", start_date,
        "--end_date", end_date,
        "--max_records", str(max_records),
        "--topic_filter", topic,
    ]

    print(f"\n{'='*60}")
    print(f"Running topic: {topic}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: Topic '{topic}' exited with code {result.returncode}")
        return None

    # Find the output file (dry_run saves with dest index name prefix)
    latest = find_latest_json(output_dir, DEST_INDEX)
    if latest:
        # Rename to our topic-specific name
        os.rename(latest, output_file)
        print(f"Output saved: {output_file}")
        return output_file

    print(f"WARNING: No output file found for topic '{topic}'")
    return None


def find_latest_json(directory, prefix):
    """Find the most recently created JSON file with the given prefix."""
    candidates = []
    for f in os.listdir(directory):
        if f.startswith(prefix) and f.endswith(".json"):
            path = os.path.join(directory, f)
            candidates.append((os.path.getmtime(path), path))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


def merge_results(json_files, output_path):
    """Merge multiple JSON result files into one."""
    all_data = []
    for f in json_files:
        try:
            with open(f) as fh:
                content = fh.read()
            # Handle concatenated JSON arrays
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

    # Filter to only records that have a topic
    filtered = [r for r in all_data if r.get('topic')]

    # Deduplicate by Ucid — keep the record with subtopics if available
    seen = {}
    for rec in filtered:
        ucid = rec.get('Ucid', '')
        if ucid not in seen:
            seen[ucid] = rec
        else:
            # Keep the one that has subtopics
            existing = seen[ucid]
            if not existing.get('sub_topic') and rec.get('sub_topic'):
                seen[ucid] = rec
    filtered = list(seen.values())
    print(f"  Deduplicated: {len(all_data)} total → {len(filtered)} unique records")

    with open(output_path, 'w') as fh:
        json.dump(filtered, fh, indent=2)

    print(f"\nMerged {len(filtered)} records (from {len(all_data)} total) into {output_path}")
    return output_path


def generate_report(merged_json, max_per_topic):
    """Run generate_report.py on the merged JSON."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_xlsx = f"data/output/subtopic_report_all_topics_{timestamp}.xlsx"

    cmd = [
        sys.executable, "generate_report.py",
        "--input", merged_json,
        "--output", output_xlsx,
        "--max_per_topic", str(max_per_topic),
    ]

    print(f"\nGenerating report: {output_xlsx}")
    subprocess.run(cmd)
    return output_xlsx


def main():
    parser = argparse.ArgumentParser(description="Run subtopic testing for all topics")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--max_records", type=int, default=8000, help="Max records per topic run (default: 8000)")
    parser.add_argument("--max_per_topic", type=int, default=200, help="Max rows per topic in report (default: 200)")
    parser.add_argument("--topics", required=False, help="Comma-separated list of topics to run (default: all)")
    args = parser.parse_args()

    topics = TOPICS
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",")]

    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting topic-by-topic test run")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Max records per run: {args.max_records}")
    print(f"Topics to test: {len(topics)}")
    print(f"Topics: {', '.join(topics)}")

    json_files = []
    results = {}

    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] Processing: {topic}")
        output_file = run_topic(topic, args.start_date, args.end_date, args.max_records, output_dir)
        if output_file:
            json_files.append(output_file)
            # Count records
            try:
                with open(output_file) as f:
                    data = json.load(f)
                topic_count = sum(1 for r in data if r.get('topic', '').lower() == topic.lower())
                results[topic] = topic_count
                print(f"  -> {topic_count} records matched topic '{topic}'")
            except Exception:
                results[topic] = "error"

    # Print summary
    print(f"\n{'='*60}")
    print("TOPIC RUN SUMMARY")
    print(f"{'='*60}")
    for topic in topics:
        count = results.get(topic, 0)
        status = f"{count} records" if isinstance(count, int) else count
        print(f"  {topic}: {status}")

    if json_files:
        # Merge all results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        merged_path = os.path.join(output_dir, f"merged_all_topics_{timestamp}.json")
        merge_results(json_files, merged_path)

        # Generate Excel report
        report_path = generate_report(merged_path, args.max_per_topic)
        print(f"\nDone! Report: {report_path}")
    else:
        print("\nNo results to merge.")


if __name__ == "__main__":
    main()
