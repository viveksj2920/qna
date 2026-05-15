"""
Rerun ONLY subtopic assignment on existing Q&A+topic data.
Does NOT redo Q&A extraction or topic assignment.
Uses updated descriptions from Pauline's Subtopic Hierarchy (2).xlsx.

This is what Dinesh asked for — modular, skip to subtopic step directly.

Usage (on company laptop with real Azure OpenAI credentials):
  python rerun_subtopics.py --input data/output/fast_results_20260515_191043.json --max_concurrent 60
"""

import argparse
import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime

import aiohttp
from dotenv import load_dotenv
load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
API_KEY = os.getenv("AZURE_OPENAI_KEY", "")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-transcripts")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

_src_dir = os.path.dirname(os.path.abspath(__file__))

# Load updated subtopic config with Pauline's descriptions
SUBTOPIC_CONFIG_PATH = os.path.join(_src_dir, "data/input/new_sub_topic_prompt_mira_v2.json")
with open(SUBTOPIC_CONFIG_PATH) as f:
    SUBTOPIC_CONFIG = json.load(f)

# Build formatted subtopic text per topic
SUBTOPIC_TEXT = {}
for topic_key, config in SUBTOPIC_CONFIG.items():
    subtopics = config.get("subtopics", [])
    descs = config.get("descriptions", {})
    lines = []
    for i, name in enumerate(subtopics, 1):
        desc_info = descs.get(name, {})
        desc = desc_info.get("description", "")
        if desc:
            lines.append(f"  {i}. {name} — {desc}")
        else:
            lines.append(f"  {i}. {name}")
    SUBTOPIC_TEXT[topic_key] = "\n".join(lines)


def build_subtopic_prompt(question, answer, topic):
    """Prompt for subtopic-only assignment with descriptions."""
    subtopic_list = SUBTOPIC_TEXT.get(topic, "")
    if not subtopic_list:
        # Fallback: check similar topic names
        for key in SUBTOPIC_TEXT:
            if topic in key or key in topic:
                subtopic_list = SUBTOPIC_TEXT[key]
                break

    prompt = f"""You are an intelligent healthcare knowledge classification assistant.
Your task is to assign subtopics to a customer question from a Medicare/healthcare call transcript.

## Topic: {topic}

## Available Sub-Topics:
{subtopic_list}

## Instructions:
1. Select ALL applicable sub-topics from the list above.
2. You MUST ONLY use sub-topic names exactly as they appear in the list. Do NOT invent, rephrase, or modify names.
3. A question may belong to multiple sub-topics if clearly applicable.
4. Be STRICT: only select a sub-topic if the question is genuinely about that specific sub-topic.
5. Use the descriptions provided to guide your selection — match the MEANING, not just keywords.
6. If none of the sub-topics clearly fit, return an empty list [].
7. Do NOT tag agent scripting/verification questions (e.g., "Do you have your Medicare number?", "What is your date of birth?").

## Response Format (valid JSON only, no other text):
{{"sub_topic": ["subtopic1", "subtopic2"]}}

Question: {question}
Answer: {answer}"""
    return prompt


async def call_llm(session, semaphore, prompt, idx, retries=2):
    url = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 500,
    }

    for attempt in range(retries + 1):
        async with semaphore:
            try:
                async with session.post(url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        if content.startswith("```"):
                            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                        if content.endswith("```"):
                            content = content[:-3]
                        return json.loads(content.strip())
                    elif resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", "5"))
                        await asyncio.sleep(retry_after)
                    else:
                        if attempt < retries:
                            await asyncio.sleep(2 ** attempt)
            except (json.JSONDecodeError, KeyError):
                if attempt < retries:
                    await asyncio.sleep(1)
            except asyncio.TimeoutError:
                if attempt < retries:
                    await asyncio.sleep(2)
            except Exception:
                if attempt < retries:
                    await asyncio.sleep(2)
    return None


async def process_all(data, max_concurrent):
    semaphore = asyncio.Semaphore(max_concurrent)
    processed = 0
    errors = 0
    start_time = time.time()
    lock = asyncio.Lock()

    async def process_record(session, idx, record):
        nonlocal processed, errors
        question = record.get("question", "")
        answer = record.get("answer", "")
        topic = record.get("topic", "").lower().strip()

        if not question or not topic:
            return

        prompt = build_subtopic_prompt(question, answer, topic)
        result = await call_llm(session, semaphore, prompt, idx)

        async with lock:
            if result and "sub_topic" in result:
                record["sub_topic"] = result["sub_topic"]
            else:
                errors += 1
            processed += 1

            if processed % 50 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed * 60
                print(f"  [{processed}/{len(data)}] {rate:.0f}/min | errors: {errors}")

    connector = aiohttp.TCPConnector(limit=max_concurrent + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in batches
        batch_size = 200
        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start:batch_start + batch_size]
            tasks = [process_record(session, batch_start + i, rec) for i, rec in enumerate(batch)]
            await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} min | {processed} processed | {errors} errors")
    return data


def save_results(data, output_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, f"subtopic_rerun_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON: {json_path}")

    # Excel
    try:
        import pandas as pd
        excel_path = os.path.join(output_dir, f"subtopic_rerun_{timestamp}.xlsx")

        for r in data:
            st = r.get('sub_topic', [])
            if isinstance(st, list):
                r['sub_topic_display'] = ", ".join(st)
            else:
                r['sub_topic_display'] = str(st) if st else ""

        df = pd.DataFrame(data)
        keep = ['Ucid', 'question', 'answer', 'topic', 'sub_topic_display', 'is_useful',
                'plan_name', 'state_processed', 'region_processed']
        cols = [c for c in keep if c in df.columns]
        df = df[cols].rename(columns={'sub_topic_display': 'sub_topic'})
        df = df.fillna('').replace('nan', '')

        # Deduplicate
        before = len(df)
        df = df.drop_duplicates(subset=['question', 'topic'], keep='first')
        print(f"Deduped: {before} -> {len(df)}")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary
            summary = df.groupby('topic').agg(
                total=('question', 'count'),
                with_subtopic=('sub_topic', lambda x: (x != '').sum()),
                useful=('is_useful', lambda x: x.str.lower().eq('yes').sum())
            ).reset_index()
            summary['coverage_pct'] = (summary['with_subtopic'] / summary['total'] * 100).round(1)
            summary['useful_pct'] = (summary['useful'] / summary['total'] * 100).round(1)
            summary = summary.sort_values('total', ascending=False)
            summary.to_excel(writer, sheet_name='Summary', index=False)

            # Per-topic sheets
            for topic in sorted(df['topic'].unique()):
                topic_df = df[df['topic'] == topic]
                sheet = topic[:28].replace('/', '-')
                topic_df.to_excel(writer, sheet_name=sheet, index=False)

            # Subtopic distribution
            st_rows = []
            for topic in sorted(df['topic'].unique()):
                topic_df = df[df['topic'] == topic]
                st_counter = Counter()
                for s in topic_df['sub_topic']:
                    if s:
                        for sub in s.split(','):
                            st_counter[sub.strip()] += 1
                for sub, cnt in st_counter.most_common():
                    st_rows.append({'Topic': topic, 'Subtopic': sub, 'Count': cnt})
            pd.DataFrame(st_rows).to_excel(writer, sheet_name='Subtopic Distribution', index=False)

        print(f"Excel: {excel_path}")
    except ImportError:
        print("openpyxl not installed, skipping Excel")


def main():
    parser = argparse.ArgumentParser(description="Rerun subtopic-only assignment")
    parser.add_argument("--input", required=True, help="Input JSON with Q&A+topic data")
    parser.add_argument("--max_concurrent", type=int, default=60, help="Max concurrent API calls")
    parser.add_argument("--output_dir", default="data/output", help="Output directory")
    args = parser.parse_args()

    if not API_KEY or "dummy" in API_KEY.lower():
        print("ERROR: Real Azure OpenAI credentials required")
        return

    with open(args.input) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {args.input}")
    print(f"Using subtopic config: {SUBTOPIC_CONFIG_PATH}")
    print(f"Descriptions available for: {sum(1 for t, c in SUBTOPIC_CONFIG.items() if c.get('descriptions'))}/{len(SUBTOPIC_CONFIG)} topics")
    print(f"Max concurrent: {args.max_concurrent}")
    print()

    data = asyncio.run(process_all(data, args.max_concurrent))
    save_results(data, args.output_dir)


if __name__ == "__main__":
    main()
