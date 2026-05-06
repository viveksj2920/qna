"""
Generate Excel report from dry_run JSON output.
Creates a workbook with:
  - Summary sheet (topic distribution, stats)
  - One sheet per topic with Q&A results (capped at --max_per_topic)
  - Empty subtopic sheet (questions with [] for gap analysis)

Usage:
  python generate_report.py --input <dry_run_json_file> --output <output_xlsx_file>
  python generate_report.py --input <json_file> --max_per_topic 200
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

    if not data:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            pass

    return data


def generate_report(input_file, output_file, max_per_topic=200):
    """Generate Excel report from dry_run JSON."""
    data = load_dry_run_json(input_file)

    if not data:
        print(f"No data found in {input_file}")
        return

    print(f"Loaded {len(data)} records from {input_file}")

    # Load predefined subtopics for validation
    all_predefined = load_all_predefined_subtopics()

    records = []
    for rec in data:
        sub_topics = rec.get('sub_topic', [])
        if isinstance(sub_topics, str):
            sub_topics = [sub_topics]

        topic = rec.get('topic', '')
        topic_subtopics = all_predefined.get(topic.lower(), set())

        # Separate predefined subtopics from extracted entities
        predefined = []
        entities = []
        for st in sub_topics:
            if st and st in topic_subtopics:
                predefined.append(st)
            elif st:
                entities.append(st)

        records.append({
            'Ucid': rec.get('Ucid', ''),
            'Question': rec.get('question', ''),
            'Answer': rec.get('answer', ''),
            'Topic': topic,
            'Predefined Subtopics': ', '.join(predefined) if predefined else '(empty)',
            'Extracted Entities': ', '.join(entities) if entities else '',
            'All Subtopics': ', '.join(sub_topics) if sub_topics else '(empty)',
            'Subtopic Count': len(sub_topics),
            'Is Useful': rec.get('is_useful', ''),
        })

    df = pd.DataFrame(records)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        # --- Sheet 1: Summary ---
        topic_counts = df['Topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Total Questions']

        empty_counts = df[df['All Subtopics'] == '(empty)']['Topic'].value_counts().reset_index()
        empty_counts.columns = ['Topic', 'Empty Subtopic Count']

        entity_counts = df[df['Extracted Entities'] != '']['Topic'].value_counts().reset_index()
        entity_counts.columns = ['Topic', 'With Entities']

        summary = topic_counts.merge(empty_counts, on='Topic', how='left')
        summary = summary.merge(entity_counts, on='Topic', how='left')
        summary['Empty Subtopic Count'] = summary['Empty Subtopic Count'].fillna(0).astype(int)
        summary['With Entities'] = summary['With Entities'].fillna(0).astype(int)
        summary['Coverage %'] = ((summary['Total Questions'] - summary['Empty Subtopic Count']) / summary['Total Questions'] * 100).round(1)
        capped = summary['Total Questions'].clip(upper=max_per_topic)
        summary['Shown in Report'] = capped

        overall = pd.DataFrame([{
            'Topic': 'TOTAL',
            'Total Questions': len(df),
            'Empty Subtopic Count': len(df[df['All Subtopics'] == '(empty)']),
            'With Entities': len(df[df['Extracted Entities'] != '']),
            'Coverage %': round((1 - len(df[df['All Subtopics'] == '(empty)']) / len(df)) * 100, 1) if len(df) > 0 else 0,
            'Shown in Report': min(len(df), max_per_topic * len(df['Topic'].unique()))
        }])
        summary = pd.concat([summary, overall], ignore_index=True)

        summary.to_excel(writer, sheet_name='Summary', index=False)
        print(f"  Summary: {len(topic_counts)} topics, {len(df)} total questions")

        # --- Per-topic sheets (capped at max_per_topic) ---
        for topic in sorted(df['Topic'].unique()):
            topic_df = df[df['Topic'] == topic][['Question', 'Predefined Subtopics', 'Extracted Entities', 'Is Useful']].copy()
            total = len(topic_df)
            topic_df = topic_df.head(max_per_topic)
            sheet_name = topic[:31].replace('/', ' ')
            topic_df.to_excel(writer, sheet_name=sheet_name, index=False)
            shown = len(topic_df)
            suffix = f" (showing {shown}/{total})" if total > max_per_topic else ""
            print(f"  {sheet_name}: {shown} rows{suffix}")

        # --- Empty Subtopics sheet (gap analysis) ---
        empty_df = df[df['All Subtopics'] == '(empty)'][['Question', 'Topic', 'Is Useful']].copy()
        if not empty_df.empty:
            empty_df.to_excel(writer, sheet_name='Empty Subtopics (Gaps)', index=False)
            print(f"  Empty Subtopics (Gaps): {len(empty_df)} rows")

        # --- Entity Extraction sheet ---
        entity_df = df[df['Extracted Entities'] != ''][['Question', 'Topic', 'Predefined Subtopics', 'Extracted Entities']].copy()
        if not entity_df.empty:
            entity_df.to_excel(writer, sheet_name='Entity Extractions', index=False)
            print(f"  Entity Extractions: {len(entity_df)} rows")

    print(f"\nReport saved to: {output_file}")


def load_all_predefined_subtopics():
    """Load all predefined subtopics from the config JSON."""
    result = {}
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'data/input/new_sub_topic_prompt_mira.json')
        with open(config_path) as f:
            config = json.load(f)
        for topic, topic_config in config.items():
            result[topic.lower()] = set(topic_config.get('subtopics', []))
    except Exception:
        pass
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Excel report from dry_run output")
    parser.add_argument("--input", required=True, help="Path to dry_run JSON file")
    parser.add_argument("--output", required=False, help="Output Excel file path")
    parser.add_argument("--max_per_topic", type=int, default=200, help="Max rows per topic sheet (default: 200)")
    args = parser.parse_args()

    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"data/output/subtopic_report_{timestamp}.xlsx"

    generate_report(args.input, args.output, args.max_per_topic)
