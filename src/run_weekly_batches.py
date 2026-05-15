"""
Run the pipeline per-topic to get 200+ records for each topic.

Since transcripts-mira has a 'topic' field (StringCollection), we filter
directly by topic in the Azure Search query. Each topic fetch is fast
(~seconds for 300 records) instead of pulling 721K records.

Usage:
  python run_weekly_batches.py --per_topic 300
  python run_weekly_batches.py --per_topic 300 --topics "enrollment,eligibility,dental"
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

ALL_TOPICS = [
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


def run_topic(topic, per_topic, output_dir):
    """Run pipeline for one topic with topic filter in the search query."""
    # Use a wide date range - the topic filter keeps the result set small
    start_str = "2026-01-01T00:00:00Z"
    end_str = "2026-05-16T00:00:00Z"

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
        "--max_records", str(per_topic),
        "--topic_filter", topic,
    ]

    print(f"\n{'='*60}")
    print(f"Topic: {topic} (fetching up to {per_topic} transcripts)")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: Topic '{topic}' exited with code {result.returncode}")
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
    """Merge all topic JSONs, deduplicate by Ucid+question."""
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
    parser = argparse.ArgumentParser(description="Run pipeline per-topic for 200+ records each")
    parser.add_argument("--per_topic", type=int, default=300, help="Transcripts to fetch per topic (default: 300)")
    parser.add_argument("--topics", required=False, help="Comma-separated topics (default: all 17)")
    args = parser.parse_args()

    topics = ALL_TOPICS
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",")]

    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Topics: {len(topics)}")
    print(f"Transcripts per topic: {args.per_topic}")
    print(f"Estimated total: {len(topics) * args.per_topic} transcripts")

    json_files = []
    summary = {}

    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] {topic}")
        start_time = datetime.now()
        output_file = run_topic(topic, args.per_topic, output_dir)

        if output_file:
            json_files.append(output_file)
            # Count records for this topic
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
                summary[topic] = {"records": qna_count, "time_min": round(elapsed, 1)}
                print(f"  -> {qna_count} Q&A records ({round(elapsed, 1)} min)")
            except Exception:
                summary[topic] = {"records": "error", "time_min": 0}
        else:
            summary[topic] = {"records": 0, "time_min": 0}

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for topic in topics:
        s = summary.get(topic, {})
        count = s.get("records", 0)
        t = s.get("time_min", 0)
        marker = " OK" if isinstance(count, int) and count >= 200 else ""
        print(f"  {topic}: {count} records, {t} min{marker}")

    if json_files:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        merged_path = os.path.join(output_dir, f"merged_per_topic_{timestamp}.json")
        records = merge_all(json_files, merged_path)

        topics_count = Counter(r.get('topic', '').lower() for r in records)
        print(f"\nFinal records per topic:")
        for t, c in topics_count.most_common():
            marker = ' OK' if c >= 200 else ''
            print(f"  {t}: {c}{marker}")

        print(f"\nNext step - run subtopic extraction:")
        print(f"  python run_subtopic_only.py --input {merged_path} --topic \"all\" --max_records 200")
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    main()
