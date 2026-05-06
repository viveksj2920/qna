"""
Generate Excel report from dry_run JSON output.
Creates a workbook with:
  - Summary sheet (topic distribution, stats)
  - One sheet per topic with all Q&A results
  - Entity extraction sheet (questions where entities were found)
  - Empty subtopic sheet (questions with [] for gap analysis)

Usage:
  python generate_report.py --input <dry_run_json_file> --output <output_xlsx_file>
"""

import argparse
import json
import os
import pandas as pd
from datetime import datetime


def load_dry_run_json(filepath):
    """Load dry_run JSON file (handles concatenated JSON arrays)."""
    with open(filepath) as f:
        content = f.read()

    data = []
    # Handle concatenated JSON arrays (batch output format)
    for chunk in content.split(']['):
        chunk = chunk.strip()
        if not chunk.startswith('['):
            chunk = '[' + chunk
        if not chunk.endswith(']'):
            chunk = chunk + ']'
        try:
            batch = json.loads(chunk)
            data.extend(batch)
        except json.JSONDecodeError:
            continue

    # Fallback: try as single JSON array
    if not data:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            pass

    return data


def generate_report(input_file, output_file):
    """Generate Excel report from dry_run JSON."""
    data = load_dry_run_json(input_file)

    if not data:
        print(f"No data found in {input_file}")
        return

    print(f"Loaded {len(data)} records from {input_file}")

    # Prepare dataframe
    records = []
    for rec in data:
        sub_topics = rec.get('sub_topic', [])
        if isinstance(sub_topics, str):
            sub_topics = [sub_topics]

        # Separate predefined subtopics from entities
        # (entities are typically lowercase single words or short phrases added by post-processor)
        predefined = []
        entities = []
        for st in sub_topics:
            if st and st in get_all_subtopics(rec.get('topic', '')):
                predefined.append(st)
            else:
                # Could be an entity or a predefined subtopic we can't verify without the JSON
                predefined.append(st)

        records.append({
            'Ucid': rec.get('Ucid', ''),
            'Question': rec.get('question', ''),
            'Answer': rec.get('answer', ''),
            'Topic': rec.get('topic', ''),
            'Subtopics': ', '.join(sub_topics) if sub_topics else '(empty)',
            'Subtopic Count': len(sub_topics),
            'Is Useful': rec.get('is_useful', ''),
            'Has Entities': bool(rec.get('entities', {})),
        })

    df = pd.DataFrame(records)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        # --- Sheet 1: Summary ---
        topic_counts = df['Topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Question Count']

        empty_subtopic_counts = df[df['Subtopics'] == '(empty)']['Topic'].value_counts().reset_index()
        empty_subtopic_counts.columns = ['Topic', 'Empty Subtopic Count']

        summary = topic_counts.merge(empty_subtopic_counts, on='Topic', how='left')
        summary['Empty Subtopic Count'] = summary['Empty Subtopic Count'].fillna(0).astype(int)
        summary['Coverage %'] = ((summary['Question Count'] - summary['Empty Subtopic Count']) / summary['Question Count'] * 100).round(1)

        # Add overall stats row
        overall = pd.DataFrame([{
            'Topic': 'TOTAL',
            'Question Count': len(df),
            'Empty Subtopic Count': len(df[df['Subtopics'] == '(empty)']),
            'Coverage %': round((1 - len(df[df['Subtopics'] == '(empty)']) / len(df)) * 100, 1) if len(df) > 0 else 0
        }])
        summary = pd.concat([summary, overall], ignore_index=True)

        summary.to_excel(writer, sheet_name='Summary', index=False)
        print(f"  Summary sheet: {len(topic_counts)} topics")

        # --- Sheet 2: All Results ---
        df.to_excel(writer, sheet_name='All Results', index=False)
        print(f"  All Results sheet: {len(df)} rows")

        # --- Per-topic sheets ---
        for topic in sorted(df['Topic'].unique()):
            topic_df = df[df['Topic'] == topic][['Question', 'Subtopics', 'Is Useful']].copy()
            # Sheet name max 31 chars
            sheet_name = topic[:31].replace('/', ' ')
            topic_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  {sheet_name}: {len(topic_df)} rows")

        # --- Empty Subtopics sheet (gap analysis) ---
        empty_df = df[df['Subtopics'] == '(empty)'][['Question', 'Topic', 'Is Useful']].copy()
        if not empty_df.empty:
            empty_df.to_excel(writer, sheet_name='Empty Subtopics (Gaps)', index=False)
            print(f"  Empty Subtopics (Gaps): {len(empty_df)} rows")

    print(f"\nReport saved to: {output_file}")


def get_all_subtopics(topic):
    """Load predefined subtopics for verification. Returns empty set if file not found."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'data/input/new_sub_topic_prompt_mira.json')
        with open(config_path) as f:
            config = json.load(f)
        topic_lower = topic.lower()
        if topic_lower in config:
            return set(config[topic_lower].get('subtopics', []))
    except Exception:
        pass
    return set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Excel report from dry_run output")
    parser.add_argument("--input", required=True, help="Path to dry_run JSON file")
    parser.add_argument("--output", required=False, help="Output Excel file path")
    args = parser.parse_args()

    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"data/output/subtopic_report_{timestamp}.xlsx"

    generate_report(args.input, args.output)
