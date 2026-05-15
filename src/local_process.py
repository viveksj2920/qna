"""
Local transcript processor - NO API calls needed.
Uses keyword matching + heuristics to extract Q&A pairs, topics, and subtopics.
Merges with existing LLM-processed data for final Excel report.

Run: /opt/homebrew/bin/python3 local_process.py
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

# Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SRC_DIR, "data/output/transcripts_raw_20260515_162503.csv")
OUTPUT_DIR = os.path.join(SRC_DIR, "data/output")
SUBTOPIC_JSON = os.path.join(SRC_DIR, "data/input/new_sub_topic_prompt_mira.json")
TOPIC_JSON = os.path.join(SRC_DIR, "data/input/topic_prompt_mira.json")

# Load configs
with open(TOPIC_JSON) as f:
    TOPIC_DESCRIPTIONS = json.load(f)

with open(SUBTOPIC_JSON) as f:
    SUBTOPIC_CONFIG = json.load(f)

# ── Topic keyword mappings (derived from topic descriptions) ──
TOPIC_KEYWORDS = {
    "providers": ["doctor", "physician", "specialist", "hospital", "facility", "network", "in-network", "out-of-network", "pcp", "primary care", "referral"],
    "enrollment": ["enroll", "enrollment", "sign up", "effective date", "card", "new member", "application", "disenroll", "cancel enrollment", "start date"],
    "plan costs": ["premium", "copay", "co-pay", "deductible", "out of pocket", "cost", "how much", "monthly payment", "max out of pocket", "coinsurance"],
    "prescription drug coverage": ["drug", "medication", "pharmacy", "prescription", "rx", "formulary", "tier", "otc", "over the counter", "pill", "tablet", "refill", "mail order", "dose", "milligram", "mg"],
    "dental": ["dental", "dentist", "teeth", "tooth", "crown", "filling", "denture", "orthodont", "periodon", "oral", "gum", "root canal", "extraction", "dental rider"],
    "plan changes": ["switch", "change plan", "different plan", "change my plan", "new plan", "upgrade", "downgrade", "transfer", "annual enrollment"],
    "eligibility": ["eligible", "eligibility", "qualify", "qualification", "medicaid", "dual", "special needs", "snp", "lis", "low income", "extra help", "part a", "part b"],
    "vision": ["vision", "eye", "glasses", "eyewear", "contact lens", "optometrist", "ophthalmol", "eye exam", "optical"],
    "transportation, food and gym benefits": ["transportation", "ride", "uber", "lyft", "gym", "fitness", "silver sneakers", "renew active", "food", "meal", "grocery", "otc benefit", "healthy benefit"],
    "digital friction": ["website", "online", "portal", "app", "myuhc", "login", "password", "can't find", "where do i find", "navigate"],
    "plan comparison": ["compare", "comparison", "difference between", "which plan", "better plan", "vs", "versus"],
    "general plan information": ["benefit", "coverage", "plan", "what does my plan", "what's covered", "what is included"],
    "service area reduction": ["not available", "no longer available", "discontinued", "leaving", "exiting", "service area", "zip code not covered", "not offered"],
    "member renewal": ["renew", "renewal", "renewing my plan", "auto-renew", "plan renewal"],
    "drug coverage termination": ["no longer covered", "not covered anymore", "discontinued drug", "removed from formulary", "dropped", "terminated"],
    "medicare id": ["medicare id", "mbi", "member id", "medicare number", "beneficiary number", "red white blue card"],
    "others": [],
}

# Subtopic keywords - extract key terms from subtopic names
SUBTOPIC_KEYWORDS = {}
for topic_key, config in SUBTOPIC_CONFIG.items():
    SUBTOPIC_KEYWORDS[topic_key] = {}
    for st in config.get("subtopics", []):
        # Extract keywords from subtopic name
        words = st.lower().split()
        # Remove common filler words
        stopwords = {"and", "or", "the", "a", "an", "of", "for", "in", "to", "is", "on", "at", "by", "with"}
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        SUBTOPIC_KEYWORDS[topic_key][st] = keywords


def extract_qa_pairs(text):
    """Extract Q&A pairs from transcript text using turn-taking patterns."""
    if not text or len(text) < 50:
        return []

    # Split into turns
    turns = re.split(r'(?:customer|agent):\s*', text, flags=re.IGNORECASE)
    turn_labels = re.findall(r'(customer|agent):', text, flags=re.IGNORECASE)

    if len(turn_labels) < 2:
        return []

    # Build labeled turns
    labeled = []
    for i, label in enumerate(turn_labels):
        if i < len(turns) - 1:
            turn_text = turns[i + 1].strip()
            if turn_text:
                labeled.append((label.lower(), turn_text))

    # Extract customer questions and following agent answers
    qa_pairs = []
    i = 0
    while i < len(labeled):
        label, text_chunk = labeled[i]

        if label == "customer":
            # Check if this is a question
            is_question = (
                "?" in text_chunk or
                text_chunk.lower().startswith(("what", "how", "when", "where", "why", "can", "do", "does", "is", "will", "would", "could", "should", "are", "am", "was", "were", "has", "have", "did"))
            )

            if is_question and len(text_chunk) > 15:
                # Collect agent answer(s)
                answer_parts = []
                j = i + 1
                while j < len(labeled):
                    if labeled[j][0] == "agent":
                        answer_parts.append(labeled[j][1])
                        j += 1
                    else:
                        break

                if answer_parts:
                    answer = " ".join(answer_parts)
                    # Truncate answer to 2 sentences
                    sentences = re.split(r'(?<=[.!?])\s+', answer)
                    if len(sentences) > 2:
                        answer = " ".join(sentences[:2])

                    # Skip greetings/small talk
                    skip_patterns = [
                        "thank you for calling", "how may i help", "good morning",
                        "good afternoon", "have a great day", "is there anything else",
                        "my name is", "how can i assist", "let me verify",
                        "can i have your", "what is your", "date of birth"
                    ]
                    question_lower = text_chunk.lower()
                    if not any(p in question_lower for p in skip_patterns):
                        qa_pairs.append({
                            "question": text_chunk.strip()[:500],
                            "answer": answer.strip()[:500]
                        })

                i = j
                continue
        i += 1

    # Limit to 10
    return qa_pairs[:10]


def classify_topic(question, answer=""):
    """Classify a Q&A pair into a topic using keyword matching."""
    text = (question + " " + answer).lower()

    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        if topic == "others":
            continue
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
                # Bonus for exact word match
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    score += 0.5
        scores[topic] = score

    if not scores or max(scores.values()) == 0:
        return "others"

    # Get top topic
    best_topic = max(scores, key=scores.get)

    # Require minimum score to avoid false positives
    if scores[best_topic] < 1:
        return "others"

    # Special disambiguation rules
    if best_topic == "general plan information":
        # If another specific topic scores well, prefer it
        specific_topics = ["dental", "vision", "prescription drug coverage", "providers", "plan costs"]
        for st in specific_topics:
            if scores.get(st, 0) >= scores[best_topic]:
                return st

    return best_topic


def classify_subtopics(question, answer, topic):
    """Assign subtopics from predefined list using keyword matching."""
    if topic not in SUBTOPIC_KEYWORDS:
        return []

    text = (question + " " + answer).lower()
    matches = []

    for subtopic_name, keywords in SUBTOPIC_KEYWORDS[topic].items():
        if not keywords:
            continue
        score = sum(1 for kw in keywords if kw in text)
        # Need at least 40% keyword match
        threshold = max(1, len(keywords) * 0.4)
        if score >= threshold:
            matches.append((subtopic_name, score))

    # Sort by score, return top 3
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches[:3]]


def is_useful_question(question):
    """Determine if a question is useful (reveals customer pain point/need)."""
    q = question.lower()

    # Not useful patterns
    not_useful = [
        "do you understand", "do you agree", "do i have your permission",
        "would you like your mail", "can i get a reference", "what is the contact",
        "how would you like to pay", "do you currently have",
        "are you calling to enroll", "will you have any other",
        "are you currently living", "has lack of transportation",
        "do you have any personal spiritual", "are you confident using",
        "what is your", "can i have your", "do you have your medicare",
        "what type of plan", "do you have medicare", "how may i help",
        "do you have medicaid"
    ]
    if any(p in q for p in not_useful):
        return "no"

    # Useful indicators - specific concerns
    useful_indicators = [
        "why", "how much", "cost", "denied", "not covered", "can't find",
        "problem", "issue", "wrong", "error", "help me", "confused",
        "don't understand", "claim", "bill", "charge", "reimburse",
        "appeal", "complaint", "when will", "how long", "specific",
        "particular", "my doctor", "my medication", "my dentist"
    ]
    if any(p in q for p in useful_indicators):
        return "yes"

    # If it's a specific question about coverage/benefits, likely useful
    if "?" in question and len(question) > 30:
        return "yes"

    return "no"


def load_existing_results():
    """Load all existing LLM-processed results."""
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

    # Deduplicate
    seen = {}
    for rec in all_data:
        if not rec.get('question') or not rec.get('topic'):
            continue
        key = (str(rec.get('Ucid', '')), rec.get('question', '')[:80])
        if key not in seen:
            seen[key] = rec
        elif rec.get('sub_topic') and not seen[key].get('sub_topic'):
            seen[key] = rec

    # Mark as LLM-processed
    for rec in seen.values():
        rec['_source'] = 'llm'
    return list(seen.values())


def process_csv():
    """Process raw transcripts from CSV."""
    print(f"Reading CSV: {CSV_PATH}")
    results = []
    row_count = 0

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            text = row.get('Text', '')
            if not text or len(text) < 100:
                continue

            qa_pairs = extract_qa_pairs(text)

            for qa in qa_pairs:
                topic = classify_topic(qa['question'], qa['answer'])
                subtopics = classify_subtopics(qa['question'], qa['answer'], topic)
                useful = is_useful_question(qa['question'])

                results.append({
                    'Ucid': str(row.get('Ucid', '')),
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'topic': topic,
                    'sub_topic': subtopics,
                    'grouped_sub_topic': subtopics,
                    'is_useful': useful,
                    'StartTime': row.get('StartTime', ''),
                    'Is_Digital': row.get('Is_Digital', ''),
                    'Is_Enrollment': row.get('Is_Enrollment', ''),
                    'plan_name': row.get('plan_name', ''),
                    'drugs': row.get('drugs', ''),
                    'providers': row.get('providers', ''),
                    'zip': row.get('zip', ''),
                    'state_processed': row.get('state_processed', ''),
                    'region_processed': row.get('region_processed', ''),
                    '_source': 'local'
                })

    print(f"Processed {row_count} transcripts -> {len(results)} Q&A pairs")
    return results


def merge_and_deduplicate(llm_results, local_results, target_per_topic=200):
    """Merge LLM + local results, preferring LLM-processed, capped at target per topic."""
    topic_buckets = defaultdict(list)

    # Add LLM results first (higher quality)
    for rec in llm_results:
        topic = rec.get('topic', 'others').lower().strip()
        topic_buckets[topic].append(rec)

    # Fill with local results
    for rec in local_results:
        topic = rec.get('topic', 'others').lower().strip()
        # Deduplicate by Ucid + question prefix
        key = (str(rec.get('Ucid', '')), rec.get('question', '')[:60])
        existing_keys = {(str(r.get('Ucid', '')), r.get('question', '')[:60]) for r in topic_buckets[topic]}
        if key not in existing_keys:
            topic_buckets[topic].append(rec)

    # Cap at target
    final = []
    for topic in sorted(topic_buckets.keys()):
        records = topic_buckets[topic][:target_per_topic]
        final.extend(records)

    return final, topic_buckets


def generate_excel(all_records, topic_buckets, target_per_topic=200):
    """Generate Excel report with summary + per-topic sheets."""
    import pandas as pd

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_path = os.path.join(OUTPUT_DIR, f"qna_analysis_report_{timestamp}.xlsx")

    # Flatten sub_topic lists
    for r in all_records:
        if isinstance(r.get('sub_topic'), list):
            r['sub_topic_str'] = ", ".join(r['sub_topic'])
        else:
            r['sub_topic_str'] = str(r.get('sub_topic', ''))
        if isinstance(r.get('grouped_sub_topic'), list):
            r['grouped_sub_topic'] = ", ".join(r['grouped_sub_topic'])

    df = pd.DataFrame(all_records)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # ── Summary sheet ──
        summary_rows = []
        for topic in sorted(topic_buckets.keys()):
            recs = topic_buckets[topic]
            count = min(len(recs), target_per_topic)
            llm_count = sum(1 for r in recs[:count] if r.get('_source') == 'llm')
            local_count = count - llm_count
            useful = sum(1 for r in recs[:count] if str(r.get('is_useful', '')).lower() == 'yes')
            unique_subtopics = set()
            for r in recs[:count]:
                st = r.get('sub_topic', [])
                if isinstance(st, list):
                    unique_subtopics.update(st)
                elif isinstance(st, str) and st:
                    unique_subtopics.update(s.strip() for s in st.split(','))
            summary_rows.append({
                'Topic': topic,
                'Total Records': count,
                'LLM Processed': llm_count,
                'Keyword Processed': local_count,
                'Useful Questions': useful,
                'Useful %': round(useful / count * 100, 1) if count > 0 else 0,
                'Unique Subtopics': len(unique_subtopics),
                'Status': 'OK' if count >= target_per_topic else f'Need {target_per_topic - count} more'
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values('Total Records', ascending=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # ── All Records sheet ──
        display_cols = ['Ucid', 'question', 'answer', 'topic', 'sub_topic_str', 'is_useful',
                        'StartTime', 'plan_name', 'state_processed', 'region_processed', '_source']
        available_cols = [c for c in display_cols if c in df.columns]
        df_display = df[available_cols].copy()
        df_display = df_display.rename(columns={'sub_topic_str': 'sub_topic'})
        df_display.to_excel(writer, sheet_name='All Records', index=False)

        # ── Per-topic sheets ──
        for topic in sorted(topic_buckets.keys()):
            recs = topic_buckets[topic][:target_per_topic]
            if not recs:
                continue
            topic_df = pd.DataFrame(recs)
            for c in ['sub_topic', 'grouped_sub_topic']:
                if c in topic_df.columns:
                    topic_df[c] = topic_df[c].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x) if x else "")
            cols = [c for c in available_cols if c in topic_df.columns]
            topic_df = topic_df[cols] if cols else topic_df
            if 'sub_topic_str' in topic_df.columns:
                topic_df = topic_df.rename(columns={'sub_topic_str': 'sub_topic'})
            # Excel sheet name max 31 chars
            sheet_name = topic[:28].replace('/', '-').replace('\\', '-')
            topic_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # ── Subtopic Distribution sheet ──
        subtopic_rows = []
        for topic in sorted(topic_buckets.keys()):
            recs = topic_buckets[topic][:target_per_topic]
            st_counter = Counter()
            for r in recs:
                st = r.get('sub_topic', [])
                if isinstance(st, list):
                    for s in st:
                        st_counter[s] += 1
                elif isinstance(st, str) and st:
                    for s in st.split(','):
                        st_counter[s.strip()] += 1
            for st, cnt in st_counter.most_common():
                subtopic_rows.append({'Topic': topic, 'Subtopic': st, 'Count': cnt})
        if subtopic_rows:
            st_df = pd.DataFrame(subtopic_rows)
            st_df.to_excel(writer, sheet_name='Subtopic Distribution', index=False)

    print(f"\nExcel saved: {excel_path}")
    return excel_path


def main():
    print("=" * 60)
    print("LOCAL Q&A PROCESSOR + REPORT GENERATOR")
    print("=" * 60)

    # Step 1: Load existing LLM-processed data
    print("\n[1/4] Loading existing LLM-processed results...")
    llm_results = load_existing_results()
    topics_llm = Counter(r.get('topic', '').lower().strip() for r in llm_results)
    print(f"  Found {len(llm_results)} LLM-processed records across {len(topics_llm)} topics")

    # Step 2: Process raw CSV locally
    print("\n[2/4] Processing raw transcripts locally (keyword-based)...")
    local_results = process_csv()
    topics_local = Counter(r.get('topic', '').lower().strip() for r in local_results)
    print(f"  Extracted {len(local_results)} Q&A pairs across {len(topics_local)} topics")
    for t, c in topics_local.most_common():
        print(f"    {t}: {c}")

    # Step 3: Merge
    print("\n[3/4] Merging results (LLM preferred, capped at 200/topic)...")
    merged, topic_buckets = merge_and_deduplicate(llm_results, local_results, target_per_topic=200)
    print(f"  Total merged records: {len(merged)}")
    for topic in sorted(topic_buckets.keys()):
        count = min(len(topic_buckets[topic]), 200)
        llm_c = sum(1 for r in topic_buckets[topic][:200] if r.get('_source') == 'llm')
        print(f"    {topic}: {count} (llm: {llm_c}, local: {count-llm_c})")

    # Step 4: Generate Excel
    print("\n[4/4] Generating Excel report...")
    excel_path = generate_excel(merged, topic_buckets, target_per_topic=200)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Report: {excel_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
