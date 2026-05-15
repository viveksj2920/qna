"""
Generate clean deduplicated Excel report for Pauline/Dinesh review.
- Removes duplicate questions within same topic
- Removes near-duplicates (first 60 chars match within topic)
- Drops unnecessary columns
- Adds analysis sheets including subtopic descriptions
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

# ── Deduplicate within each topic ──
print("Deduplicating...")
deduped = []
seen_by_topic = defaultdict(set)

for r in data:
    topic = r.get('topic', '')
    q = r.get('question', '').strip().lower()

    # Exact duplicate check
    if q[:100] in seen_by_topic[topic]:
        continue

    # Near-duplicate check (first 60 chars)
    prefix = q[:60]
    near_match = False
    for existing in seen_by_topic[topic]:
        if existing[:60] == prefix:
            near_match = True
            break
    if near_match:
        continue

    seen_by_topic[topic].add(q[:100])
    deduped.append(r)

print(f"Before dedup: {len(data)}")
print(f"After dedup: {len(deduped)}")
print(f"Removed: {len(data) - len(deduped)}")

data = deduped

# Flatten sub_topic lists to strings
for r in data:
    st = r.get('sub_topic', [])
    if isinstance(st, list):
        r['sub_topic'] = ", ".join(st)
    elif not st:
        r['sub_topic'] = ""

# Columns to keep
KEEP_COLS = ['Ucid', 'question', 'answer', 'topic', 'sub_topic', 'is_useful',
             'plan_name', 'state_processed', 'region_processed']

df = pd.DataFrame(data)
df = df[[c for c in KEEP_COLS if c in df.columns]]
df = df.fillna('').replace('nan', '')

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

    # ── 2. Subtopic Distribution with Descriptions ──
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

        # Get descriptions from config
        topic_config = subtopic_config.get(topic, {})
        descriptions = topic_config.get("descriptions", {})

        for sub, cnt in st_counter.most_common():
            pct = round(cnt / len(topic_df) * 100, 1)
            desc_info = descriptions.get(sub, {})
            desc = desc_info.get("description", "")
            examples = desc_info.get("examples", "")
            is_predefined = sub in set(topic_config.get("subtopics", []))
            st_rows.append({
                'Topic': topic,
                'Subtopic': sub,
                'Count': cnt,
                '% of Topic': pct,
                'Predefined': 'Yes' if is_predefined else 'No (LLM invented)',
                'Description': desc,
                'Examples': examples,
            })
    pd.DataFrame(st_rows).to_excel(writer, sheet_name='Subtopic Distribution', index=False)

    # ── 3. Gap Analysis ──
    gap_rows = []
    for topic_key, config in subtopic_config.items():
        predefined = set(config.get('subtopics', []))
        descriptions = config.get('descriptions', {})
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
            'Total Predefined': len(predefined),
            'Matched': len(matched),
            'Coverage %': round(len(matched) / len(predefined) * 100, 1) if predefined else 0,
            'Unused (no questions matched)': len(unused),
            'LLM Invented (outside list)': len(invented),
            'Unused Subtopics': ", ".join(sorted(unused)),
            'Invented Subtopics': ", ".join(sorted(invented)),
        })
    gap_df = pd.DataFrame(gap_rows).sort_values('Coverage %', ascending=False)
    gap_df.to_excel(writer, sheet_name='Gap Analysis', index=False)

    # ── 4. Pauline Review Sheet — all predefined subtopics with status ──
    review_rows = []
    for topic_key, config in sorted(subtopic_config.items()):
        predefined = config.get('subtopics', [])
        descriptions = config.get('descriptions', {})
        topic_df = df[df['topic'] == topic_key]

        # Count actual usage
        st_counter = Counter()
        for s in topic_df['sub_topic']:
            if s:
                for sub in s.split(','):
                    st_counter[sub.strip()] += 1

        for sub in predefined:
            desc_info = descriptions.get(sub, {})
            count = st_counter.get(sub, 0)
            review_rows.append({
                'Topic': topic_key,
                'Subtopic': sub,
                'Description (from Pauline)': desc_info.get('description', '-- needs description --'),
                'Questions Matched': count,
                'Status': 'Active' if count > 0 else 'No matches — review needed',
            })
    pd.DataFrame(review_rows).to_excel(writer, sheet_name='Pauline Review', index=False)

    # ── 5. All Records ──
    df.to_excel(writer, sheet_name='All Records', index=False)

    # ── 6. Per-topic sheets ──
    for topic in sorted(df['topic'].unique()):
        topic_df = df[df['topic'] == topic]
        sheet_name = topic[:28].replace('/', '-')
        topic_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\nReport: {excel_path}")
print(f"Total records: {len(df)}")
print(f"Topics: {df['topic'].nunique()}")

# Print per-topic counts after dedup
print(f"\nPer-topic after dedup:")
for topic, count in df['topic'].value_counts().sort_values(ascending=False).items():
    print(f"  {topic}: {count}")
