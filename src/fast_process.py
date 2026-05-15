"""
Fast single-call transcript processor.

Combines Q&A extraction + topic + is_useful + subtopic into ONE LLM call per transcript.
4x fewer API calls than the standard pipeline. Stops early once every topic hits target.

Usage (run on the machine with real Azure OpenAI credentials):
  python fast_process.py --input data/output/transcripts_raw_20260515_162503.csv --target_per_topic 200
  python fast_process.py --input data/output/transcripts_raw_20260515_162503.csv --target_per_topic 200 --max_concurrent 80
"""

import argparse
import asyncio
import csv
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

# Load topic descriptions
_src_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_src_dir, "data/input/topic_prompt_mira.json")) as f:
    TOPIC_DESCRIPTIONS = json.load(f)

# Load subtopic config
with open(os.path.join(_src_dir, "data/input/new_sub_topic_prompt_mira.json")) as f:
    SUBTOPIC_CONFIG = json.load(f)

# Build topic descriptions text
TOPIC_TEXT = "\n".join(f"- {t}: {d}" for t, d in TOPIC_DESCRIPTIONS.items())

# Build subtopic text per topic
SUBTOPIC_TEXT = {}
for topic_key, config in SUBTOPIC_CONFIG.items():
    subtopics = config.get("subtopics", [])
    descs = config.get("descriptions", {})
    lines = []
    for i, name in enumerate(subtopics, 1):
        desc_info = descs.get(name, {})
        desc = desc_info.get("description", "")
        line = f"  {i}. {name}"
        if desc:
            line += f" -- {desc}"
        lines.append(line)
    SUBTOPIC_TEXT[topic_key] = "\n".join(lines)


def build_combined_prompt(transcript_text):
    """Single prompt that does QnA extraction + topic + is_useful + subtopic all at once."""

    # Collect all subtopic lists
    all_subtopics_section = ""
    for topic_key in sorted(SUBTOPIC_TEXT.keys()):
        all_subtopics_section += f"\n### Subtopics for '{topic_key}':\n{SUBTOPIC_TEXT[topic_key]}\n"

    prompt = f"""You are an intelligent healthcare knowledge extraction assistant.
You will analyze a call transcript between an agent and a customer at a Medicare/healthcare company.

## STEP 1: Extract Q&A pairs
- Extract question-answer pairs from the transcript (customer questions, agent answers).
- Combine multiple answers to the same question into one comprehensive answer (max 2 sentences).
- Skip greetings, small talk, closing remarks unless they contain insurance-related questions.
- Focus only on questions asked by the CUSTOMER, not the agent.
- Max 10 Q&A pairs per transcript, most important first.

## STEP 2: For EACH Q&A pair, classify:

### Topic (pick exactly ONE from this list):
{TOPIC_TEXT}

### Is Useful?
A question is USEFUL if it reveals a specific customer pain point, need, or concern about their healthcare coverage.
A question is NOT USEFUL if it is generic enrollment scripting, consent questions, basic plan inquiry without specific concern, or agent verification questions.

### Sub-topics (pick from the PREDEFINED list for the assigned topic):
{all_subtopics_section}

## IMPORTANT RULES:
- You MUST ONLY select sub-topic names from the predefined lists above.
- Do NOT invent or rephrase sub-topic names.
- A question may have multiple sub-topics if clearly applicable.
- If no sub-topic fits, use an empty list [].
- PII redacted with **** does not affect analysis.

## Response Format (valid JSON array only, no other text):
```json
[
  {{
    "question": "the customer's question",
    "answer": "the agent's answer (max 2 sentences)",
    "topic": "one topic from the list",
    "is_useful": "yes" or "no",
    "sub_topic": ["subtopic1", "subtopic2"]
  }}
]
```

Return ONLY the JSON array. No explanation, no markdown fences.

Call Transcript:
{transcript_text}"""
    return prompt


async def call_llm(session, semaphore, prompt, row_idx, retries=2):
    """Call Azure OpenAI with retry."""
    url = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4000,
    }

    for attempt in range(retries + 1):
        async with semaphore:
            try:
                async with session.post(url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        # Clean markdown fences
                        if content.startswith("```"):
                            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                        if content.endswith("```"):
                            content = content[:-3]
                        content = content.strip()
                        return json.loads(content)
                    elif resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", "5"))
                        print(f"  [Row {row_idx}] Rate limited, waiting {retry_after}s...")
                        await asyncio.sleep(retry_after)
                    else:
                        text = await resp.text()
                        print(f"  [Row {row_idx}] HTTP {resp.status}: {text[:100]}")
                        if attempt < retries:
                            await asyncio.sleep(2 ** attempt)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [Row {row_idx}] Parse error: {e}")
                if attempt < retries:
                    await asyncio.sleep(1)
            except asyncio.TimeoutError:
                print(f"  [Row {row_idx}] Timeout, attempt {attempt+1}")
                if attempt < retries:
                    await asyncio.sleep(2)
            except Exception as e:
                print(f"  [Row {row_idx}] Error: {e}")
                if attempt < retries:
                    await asyncio.sleep(2)
    return None


async def process_all(csv_path, target_per_topic, max_concurrent):
    """Process all transcripts, stopping when every topic reaches target."""

    # Read CSV
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transcripts from {csv_path}")

    # Track per-topic counts
    topic_counts = Counter()
    all_results = []
    topic_target = target_per_topic
    all_topic_names = list(TOPIC_DESCRIPTIONS.keys())
    lock = asyncio.Lock()

    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    # Stats
    processed = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    async def process_row(session, idx, row):
        nonlocal processed, skipped, errors

        text = str(row.get("Text", ""))
        if not text or len(text) < 50:
            return

        prompt = build_combined_prompt(text)
        result = await call_llm(session, semaphore, prompt, idx)

        if not result or not isinstance(result, list):
            async with lock:
                errors += 1
            return

        # Process each Q&A pair
        new_records = []
        for qna in result:
            topic = qna.get("topic", "others").lower().strip()
            # Validate topic
            if topic not in TOPIC_DESCRIPTIONS:
                topic = "others"

            record = {
                "Ucid": str(row.get("Ucid", "")),
                "question": qna.get("question", ""),
                "answer": qna.get("answer", ""),
                "topic": topic,
                "sub_topic": qna.get("sub_topic", []),
                "is_useful": qna.get("is_useful", "no"),
                "StartTime": str(row.get("StartTime", "")),
                "Is_Digital": row.get("Is_Digital", ""),
                "Is_Enrollment": row.get("Is_Enrollment", ""),
                "plan_name": row.get("plan_name", ""),
                "drugs": row.get("drugs", ""),
                "providers": row.get("providers", ""),
                "zip": row.get("zip", ""),
                "state_processed": row.get("state_processed", ""),
                "region_processed": row.get("region_processed", ""),
            }
            new_records.append(record)

        async with lock:
            for rec in new_records:
                topic = rec["topic"]
                if topic_counts[topic] < topic_target:
                    all_results.append(rec)
                    topic_counts[topic] += 1
                else:
                    skipped += 1
            processed += 1

            if processed % 25 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed * 60
                topics_done = sum(1 for t in all_topic_names if topic_counts[t] >= topic_target)
                total_qna = len(all_results)
                print(f"  [{processed}/{len(df)}] {rate:.0f} transcripts/min | {total_qna} Q&As | {topics_done}/{len(all_topic_names)} topics done | errors: {errors}")

                # Check if we have enough for all topics
                if topics_done >= len(all_topic_names):
                    print(f"\n  ALL TOPICS REACHED {topic_target}! Stopping early.")
                    return "DONE"

    # Process with aiohttp session
    connector = aiohttp.TCPConnector(limit=max_concurrent + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in batches to allow early stopping
        batch_size = 200
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch = df.iloc[batch_start:batch_end]

            tasks = []
            for idx, (_, row) in enumerate(batch.iterrows(), batch_start):
                tasks.append(process_row(session, idx, row))

            results = await asyncio.gather(*tasks)

            # Check for early stopping
            if "DONE" in results:
                break

            # Print topic status after each batch
            topics_done = sum(1 for t in all_topic_names if topic_counts[t] >= topic_target)
            if topics_done >= len(all_topic_names):
                break

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"Processed: {processed} transcripts, {len(all_results)} Q&A records, {errors} errors")
    print(f"\nPer-topic counts:")
    for t, c in topic_counts.most_common():
        marker = " OK" if c >= topic_target else f" (need {topic_target - c} more)"
        print(f"  {t}: {c}{marker}")

    return all_results


def save_results(results, output_dir):
    """Save as both JSON and Excel."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, f"fast_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Excel with per-topic sheets
    try:
        import pandas as pd
        excel_path = os.path.join(output_dir, f"fast_results_{timestamp}.xlsx")

        # Flatten sub_topic list to string
        for r in results:
            if isinstance(r.get("sub_topic"), list):
                r["sub_topic"] = ", ".join(r["sub_topic"])

        df = pd.DataFrame(results)

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = df.groupby('topic').agg(
                count=('question', 'count'),
                useful_count=('is_useful', lambda x: sum(1 for v in x if str(v).lower() == 'yes')),
            ).reset_index()
            summary_data['useful_pct'] = (summary_data['useful_count'] / summary_data['count'] * 100).round(1)
            summary_data.to_excel(writer, sheet_name='Summary', index=False)

            # All data sheet
            df.to_excel(writer, sheet_name='All Records', index=False)

            # Per-topic sheets (first 200 each)
            for topic in sorted(df['topic'].unique()):
                topic_df = df[df['topic'] == topic].head(200)
                sheet_name = topic[:31].replace('/', '-')  # Excel sheet name limit
                topic_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved Excel: {excel_path}")
    except ImportError:
        print("openpyxl not installed, skipping Excel output")


def main():
    parser = argparse.ArgumentParser(description="Fast single-call transcript processor")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--target_per_topic", type=int, default=200, help="Target records per topic (default: 200)")
    parser.add_argument("--max_concurrent", type=int, default=60, help="Max concurrent API calls (default: 60)")
    parser.add_argument("--output_dir", default="data/output", help="Output directory")
    args = parser.parse_args()

    if not API_KEY or "dummy" in API_KEY.lower():
        print("ERROR: Real Azure OpenAI credentials required in .env")
        print(f"  ENDPOINT: {ENDPOINT}")
        print(f"  KEY: {API_KEY[:10]}...")
        return

    print(f"Fast transcript processor")
    print(f"  Input: {args.input}")
    print(f"  Target per topic: {args.target_per_topic}")
    print(f"  Max concurrent: {args.max_concurrent}")
    print(f"  Topics: {len(TOPIC_DESCRIPTIONS)}")
    print(f"  Endpoint: {ENDPOINT}")
    print()

    results = asyncio.run(process_all(args.input, args.target_per_topic, args.max_concurrent))
    if results:
        save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
