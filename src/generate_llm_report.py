"""
Generate Excel report from ONLY LLM-processed records.
No keyword filler — only real GPT-4o output.
"""

import json, os
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/output")

def load_llm_results():
    """Load and deduplicate all LLM-processed JSON results."""
    all_data = []
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(OUTPUT_DIR, fname)
        try:
            with open(fpath) as f:
                content = f.read()
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
        except:
            continue

    # Deduplicate by Ucid + question
    seen = {}
    for rec in all_data:
        if not rec.get('question') or not rec.get('topic'):
            continue
        key = (str(rec.get('Ucid', '')), rec.get('question', '')[:80])
        if key not in seen:
            seen[key] = rec
        elif rec.get('sub_topic') and not seen[key].get('sub_topic'):
            seen[key] = rec

    return list(seen.values())


def generate_report(records):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_path = os.path.join(OUTPUT_DIR, f"subtopic_analysis_llm_only_{timestamp}.xlsx")

    # Flatten sub_topic
    for r in records:
        st = r.get('sub_topic', [])
        if isinstance(st, list):
            r['sub_topic_display'] = ", ".join(st)
        else:
            r['sub_topic_display'] = str(st) if st else ""
        gst = r.get('grouped_sub_topic', [])
        if isinstance(gst, list):
            r['grouped_sub_topic_display'] = ", ".join(gst)
        else:
            r['grouped_sub_topic_display'] = str(gst) if gst else ""

    # Group by topic
    topic_buckets = defaultdict(list)
    for r in records:
        topic = r.get('topic', 'others').lower().strip()
        topic_buckets[topic].append(r)

    display_cols = ['Ucid', 'question', 'answer', 'topic', 'sub_topic_display', 'is_useful',
                    'StartTime', 'plan_name', 'drugs', 'state_processed', 'region_processed']

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # ── Summary sheet ──
        summary_rows = []
        for topic in sorted(topic_buckets.keys()):
            recs = topic_buckets[topic]
            has_subtopic = sum(1 for r in recs if r.get('sub_topic_display'))
            useful = sum(1 for r in recs if str(r.get('is_useful', '')).lower() in ('true', 'yes'))

            # Unique subtopics
            unique_st = set()
            for r in recs:
                st = r.get('sub_topic', [])
                if isinstance(st, list):
                    unique_st.update(s for s in st if s)
                elif isinstance(st, str) and st:
                    unique_st.update(s.strip() for s in st.split(',') if s.strip())

            coverage_pct = round(has_subtopic / len(recs) * 100, 1) if recs else 0
            useful_pct = round(useful / len(recs) * 100, 1) if recs else 0

            summary_rows.append({
                'Topic': topic,
                'Total Q&As': len(recs),
                'With Subtopic': has_subtopic,
                'Subtopic Coverage %': coverage_pct,
                'Useful Questions': useful,
                'Useful %': useful_pct,
                'Unique Subtopics Seen': len(unique_st),
                'Top Subtopics': ", ".join(list(unique_st)[:5])
            })

        summary_df = pd.DataFrame(summary_rows).sort_values('Total Q&As', ascending=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # ── Comparison with May 6 report ──
        may6_counts = {
            'enrollment': 18, 'member renewal': 2, 'eligibility': 14,
            'general plan information': 9, 'providers': 5, 'prescription drug coverage': 5,
            'others': 29, 'digital friction': 3, 'plan costs': 7, 'plan comparison': 4, 'vision': 1
        }
        comparison_rows = []
        for topic in sorted(set(list(topic_buckets.keys()) + list(may6_counts.keys()))):
            comparison_rows.append({
                'Topic': topic,
                'May 6 Report (200 total)': may6_counts.get(topic, 0),
                'Current Report': len(topic_buckets.get(topic, [])),
                'Improvement': f"{len(topic_buckets.get(topic, [])) - may6_counts.get(topic, 0):+d}"
            })
        comp_df = pd.DataFrame(comparison_rows).sort_values('Current Report', ascending=False)
        comp_df.to_excel(writer, sheet_name='vs May 6 Report', index=False)

        # ── All Records ──
        all_df = pd.DataFrame(records)
        cols = [c for c in display_cols if c in all_df.columns]
        all_df[cols].to_excel(writer, sheet_name='All Records', index=False)

        # ── Per-topic sheets ──
        for topic in sorted(topic_buckets.keys()):
            recs = topic_buckets[topic]
            if not recs:
                continue
            topic_df = pd.DataFrame(recs)
            cols_available = [c for c in display_cols if c in topic_df.columns]
            sheet_name = topic[:28].replace('/', '-')
            topic_df[cols_available].to_excel(writer, sheet_name=sheet_name, index=False)

        # ── Subtopic Distribution ──
        st_rows = []
        for topic in sorted(topic_buckets.keys()):
            recs = topic_buckets[topic]
            st_counter = Counter()
            for r in recs:
                st = r.get('sub_topic', [])
                if isinstance(st, list):
                    for s in st:
                        if s:
                            st_counter[s] += 1
                elif isinstance(st, str) and st:
                    for s in st.split(','):
                        if s.strip():
                            st_counter[s.strip()] += 1
            for st, cnt in st_counter.most_common():
                st_rows.append({'Topic': topic, 'Subtopic': st, 'Count': cnt})

        if st_rows:
            pd.DataFrame(st_rows).to_excel(writer, sheet_name='Subtopic Distribution', index=False)

        # ── Gap Analysis ──
        # Load predefined subtopics to show which ones have NO matches
        subtopic_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/input/new_sub_topic_prompt_mira.json")
        try:
            with open(subtopic_json) as f:
                subtopic_config = json.load(f)

            gap_rows = []
            for topic_key, config in subtopic_config.items():
                predefined = set(config.get("subtopics", []))
                # Get actual seen subtopics for this topic
                seen = set()
                for r in topic_buckets.get(topic_key, []):
                    st = r.get('sub_topic', [])
                    if isinstance(st, list):
                        seen.update(s for s in st if s)
                    elif isinstance(st, str) and st:
                        seen.update(s.strip() for s in st.split(',') if s.strip())

                matched = predefined & seen
                unmatched_predefined = predefined - seen
                extra = seen - predefined  # LLM invented subtopics not in predefined list

                gap_rows.append({
                    'Topic': topic_key,
                    'Predefined Subtopics': len(predefined),
                    'Matched': len(matched),
                    'Coverage %': round(len(matched) / len(predefined) * 100, 1) if predefined else 0,
                    'Unused Predefined': len(unmatched_predefined),
                    'LLM-Invented (not in list)': len(extra),
                    'Sample Unused': ", ".join(list(unmatched_predefined)[:3]),
                    'Sample Invented': ", ".join(list(extra)[:3])
                })

            gap_df = pd.DataFrame(gap_rows).sort_values('Coverage %', ascending=False)
            gap_df.to_excel(writer, sheet_name='Gap Analysis', index=False)
        except:
            pass

    print(f"\nExcel saved: {excel_path}")
    return excel_path


def main():
    print("Loading LLM-processed records...")
    records = load_llm_results()
    print(f"Total unique LLM records: {len(records)}")

    topics = Counter(r.get('topic', '').lower().strip() for r in records)
    print(f"\nPer-topic:")
    for t, c in topics.most_common():
        print(f"  {t}: {c}")

    print(f"\nGenerating Excel report...")
    path = generate_report(records)

    print(f"\n{'='*60}")
    print(f"REPORT READY: {path}")
    print(f"Total records: {len(records)} (all GPT-4o processed)")
    print(f"Topics covered: {len(topics)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
