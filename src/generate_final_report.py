"""
Generate clean Excel report for Dinesh/Pauline review.
Drops unnecessary columns, adds analysis sheets.
"""

import json, os
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SRC_DIR, "data/output")
JSON_PATH = os.path.join(OUTPUT_DIR, "fast_results_20260515_191043.json")
SUBTOPIC_JSON = os.path.join(SRC_DIR, "data/input/new_sub_topic_prompt_mira.json")

with open(JSON_PATH) as f:
    data = json.load(f)

with open(SUBTOPIC_JSON) as f:
    subtopic_config = json.load(f)

# Flatten sub_topic lists to strings
for r in data:
    st = r.get('sub_topic', [])
    if isinstance(st, list):
        r['sub_topic'] = ", ".join(st)
    elif not st:
        r['sub_topic'] = ""

# Columns to KEEP (drop Is_Digital, Is_Enrollment, drugs, providers, zip — mostly empty/useless)
KEEP_COLS = ['Ucid', 'question', 'answer', 'topic', 'sub_topic', 'is_useful',
             'plan_name', 'state_processed', 'region_processed']

df = pd.DataFrame(data)
df = df[[c for c in KEEP_COLS if c in df.columns]]

# Clean up nan display
df = df.fillna('')
df = df.replace('nan', '')

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
excel_path = os.path.join(OUTPUT_DIR, f"subtopic_analysis_final_{timestamp}.xlsx")

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

    # ── 1. Summary ──
    summary_rows = []
    for topic in sorted(df['topic'].unique()):
        topic_df = df[df['topic'] == topic]
        has_sub = (topic_df['sub_topic'] != '').sum()
        useful = topic_df['is_useful'].str.lower().eq('yes').sum()
        unique_subs = set()
        for s in topic_df['sub_topic']:
            if s:
                unique_subs.update(x.strip() for x in s.split(',') if x.strip())
        summary_rows.append({
            'Topic': topic,
            'Total Q&As': len(topic_df),
            'With Subtopic': has_sub,
            'Subtopic Coverage %': round(has_sub / len(topic_df) * 100, 1),
            'Useful Questions': useful,
            'Useful %': round(useful / len(topic_df) * 100, 1),
            'Unique Subtopics': len(unique_subs),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values('Total Q&As', ascending=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

    # ── 2. Subtopic Distribution (the key sheet for Pauline) ──
    st_rows = []
    for topic in sorted(df['topic'].unique()):
        topic_df = df[df['topic'] == topic]
        st_counter = Counter()
        for s in topic_df['sub_topic']:
            if s:
                for sub in s.split(','):
                    sub = sub.strip()
                    if sub:
                        st_counter[sub] += 1
        for sub, cnt in st_counter.most_common():
            pct = round(cnt / len(topic_df) * 100, 1)
            st_rows.append({'Topic': topic, 'Subtopic': sub, 'Count': cnt, '% of Topic': pct})
    pd.DataFrame(st_rows).to_excel(writer, sheet_name='Subtopic Distribution', index=False)

    # ── 3. Gap Analysis — predefined vs actual ──
    gap_rows = []
    for topic_key, config in subtopic_config.items():
        predefined = set(config.get('subtopics', []))
        topic_df = df[df['topic'] == topic_key]
        seen = set()
        for s in topic_df['sub_topic']:
            if s:
                seen.update(x.strip() for x in s.split(',') if x.strip())
        matched = predefined & seen
        unused = predefined - seen
        invented = seen - predefined
        gap_rows.append({
            'Topic': topic_key,
            'Predefined Count': len(predefined),
            'Matched': len(matched),
            'Predefined Coverage %': round(len(matched) / len(predefined) * 100, 1) if predefined else 0,
            'Unused Predefined': len(unused),
            'LLM Invented': len(invented),
            'Unused Examples': ", ".join(sorted(unused)[:5]),
            'Invented Examples': ", ".join(sorted(invented)[:5]),
        })
    gap_df = pd.DataFrame(gap_rows).sort_values('Predefined Coverage %', ascending=False)
    gap_df.to_excel(writer, sheet_name='Gap Analysis', index=False)

    # ── 4. All Records ──
    df.to_excel(writer, sheet_name='All Records', index=False)

    # ── 5. Per-topic sheets ──
    for topic in sorted(df['topic'].unique()):
        topic_df = df[df['topic'] == topic]
        sheet_name = topic[:28].replace('/', '-')
        topic_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Report saved: {excel_path}")
print(f"Total records: {len(df)}")
print(f"Topics: {df['topic'].nunique()}")
print(f"Sheets: Summary, Subtopic Distribution, Gap Analysis, All Records, + {df['topic'].nunique()} topic sheets")
