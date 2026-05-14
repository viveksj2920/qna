"""
Subtopic-only extraction script.

Reads questions that already have topics assigned from either:
  1. A previous dry_run JSON output file (--input)
  2. An Azure Search QnA index (--index)

Runs ONLY the subtopic extraction step using async parallel processing
(60 concurrent LLM calls). Much faster than the full pipeline.

Usage:
  # From a previous dry_run JSON file (recommended)
  python run_subtopic_only.py --input data/output/merged_all_topics_20260512.json --topic "enrollment" --max_records 200
  python run_subtopic_only.py --input data/output/merged_all_topics_20260512.json --topic "all" --max_records 200

  # From an Azure Search index (if QnA index exists)
  python run_subtopic_only.py --index "qna-index-name" --topic "enrollment" --max_records 200
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime

import nest_asyncio
nest_asyncio.apply()

from utils.logger_config import logger
from utils.helper import load_project_config, clean_topic
from prompts.prompt_config import prompt_sub_topic_format, sub_topic_extraction_prompt
import config as app_config

try:
    from index import IndexProcessor
except ImportError:
    IndexProcessor = None

try:
    from openai import AsyncAzureOpenAI
except ImportError:
    AsyncAzureOpenAI = None

PROJECT = "MIRA"
MAX_CONCURRENT = 60


def load_questions_from_json(filepath, topic, max_records=200):
    """Load questions for a given topic from a dry_run JSON file."""
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
            data.extend(json.loads(chunk))
        except json.JSONDecodeError:
            continue

    if not data:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            pass

    filtered = [
        r for r in data
        if r.get('topic', '').lower() == topic.lower() and r.get('question')
    ]

    logger.info(f"Loaded {len(filtered)} '{topic}' records from JSON (out of {len(data)} total)")
    return filtered[:max_records]


def load_questions_from_index(index_name, topic, max_records=200):
    """Load questions for a given topic from an Azure Search index."""
    if IndexProcessor is None:
        logger.error("IndexProcessor not available.")
        return []

    index_processor = IndexProcessor(index_name=index_name)
    topic_filter = f"topic eq '{topic}' and question ne ''"
    fields = ["Ucid", "question", "answer", "topic", "sub_topic", "is_useful"]

    try:
        documents = index_processor.azure_search.search(
            filter=topic_filter, select=fields, top=max_records
        )
        records = list(documents)
        logger.info(f"Fetched {len(records)} records for topic '{topic}' from index")
        return records[:max_records]
    except Exception as e:
        logger.error(f"Error fetching records for topic '{topic}': {e}")
        return []


def build_subtopic_prompt(question, topic, project="MIRA"):
    """Build the subtopic extraction prompt for a question."""
    topic_cleaned = clean_topic(project, topic)
    sub_topic_prompt_dict = load_project_config(project, "new_subtopic_extraction")

    if topic_cleaned not in sub_topic_prompt_dict:
        return None

    sub_topic_prompt = sub_topic_prompt_dict[topic_cleaned]
    sub_topic_prompt = prompt_sub_topic_format(project, sub_topic_prompt)

    prompt = sub_topic_extraction_prompt(
        question=question, topic=topic, sub_topic_descriptions=sub_topic_prompt
    )
    cleaned_lines = [line.lstrip() for line in prompt.splitlines()]
    return "\n".join(cleaned_lines)


async def extract_subtopic_async(question, topic, semaphore, rate_limiter, project="MIRA"):
    """Async subtopic extraction with rate limiting."""
    async with semaphore:
        await rate_limit_wait(rate_limiter)

        prompt = build_subtopic_prompt(question, topic, project)
        if not prompt:
            return []

        message = [{"role": "user", "content": prompt}]

        try:
            client = AsyncAzureOpenAI(
                azure_endpoint=app_config.AZURE_OPENAI_ENDPOINT,
                api_version=app_config.AZURE_OPENAI_API_VERSION,
                api_key=app_config.AZURE_OPENAI_KEY,
            )

            response = await client.chat.completions.create(
                model=app_config.AZURE_OPENAI_DEPLOYMENT,
                messages=message,
                max_tokens=2000,
                temperature=1e-9
            )

            result = response.choices[0].message.content
            if result:
                parsed = json.loads(result) if isinstance(result, str) else result
                return parsed.get("sub_topic", [])
        except Exception as e:
            logger.error(f"Error extracting subtopic: {e}")
            # Back off on rate limit
            if "429" in str(e):
                rate_limiter["qps"] *= 0.8
                await asyncio.sleep(2)

        return []


async def rate_limit_wait(rate_limiter):
    """Simple rate limiter."""
    now = time.time()
    min_interval = 1.0 / rate_limiter["qps"]
    elapsed = now - rate_limiter["last"]
    if elapsed < min_interval:
        await asyncio.sleep(min_interval - elapsed)
    rate_limiter["last"] = time.time()


async def run_topic_async(records, topic):
    """Run subtopic extraction for all records under a topic using async."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    rate_limiter = {"qps": 40, "last": 0}

    total = len(records)
    results = [None] * total

    async def process_one(i, rec):
        question = rec.get("question", "")
        if not question:
            return

        new_subtopics = await extract_subtopic_async(question, topic, semaphore, rate_limiter, PROJECT)

        results[i] = {
            "Ucid": rec.get("Ucid", ""),
            "question": question,
            "answer": rec.get("answer", ""),
            "topic": topic,
            "sub_topic": new_subtopics,
            "old_sub_topic": rec.get("sub_topic", []),
            "is_useful": rec.get("is_useful", ""),
        }

    tasks = [process_one(i, rec) for i, rec in enumerate(records)]

    # Process with progress tracking
    completed = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        completed += 1
        if completed % 20 == 0 or completed == total:
            logger.info(f"  Progress: {completed}/{total} done")

    return [r for r in results if r is not None]


def get_available_topics(filepath):
    """Get list of unique topics from a JSON file."""
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
            data.extend(json.loads(chunk))
        except json.JSONDecodeError:
            continue

    topics = set()
    for r in data:
        t = r.get('topic', '')
        if t:
            topics.add(t)
    return sorted(topics)


def main():
    parser = argparse.ArgumentParser(description="Subtopic-only extraction (async)")
    parser.add_argument("--input", required=False, help="Path to dry_run JSON file with Q&A + topics")
    parser.add_argument("--index", required=False, help="Azure Search QnA index name")
    parser.add_argument("--topic", required=True, help="Topic to process ('all' for all, or comma-separated)")
    parser.add_argument("--max_records", type=int, default=200, help="Max records per topic (default: 200)")
    parser.add_argument("--concurrency", type=int, default=60, help="Max concurrent LLM calls (default: 60)")
    args = parser.parse_args()

    if not args.input and not args.index:
        parser.error("Either --input (JSON file) or --index (Azure Search index) is required")

    global MAX_CONCURRENT
    MAX_CONCURRENT = args.concurrency

    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    source_type = "JSON" if args.input else "index"

    # Determine topics
    if args.topic.lower() == "all":
        if args.input:
            topics = get_available_topics(args.input)
        else:
            sub_topic_config = load_project_config(PROJECT, "new_subtopic_extraction")
            topics = list(sub_topic_config.keys())
    else:
        topics = [t.strip() for t in args.topic.split(",")]

    print(f"Source: {source_type} ({args.input or args.index})")
    print(f"Topics: {', '.join(topics)}")
    print(f"Max records per topic: {args.max_records}")
    print(f"Concurrency: {MAX_CONCURRENT}")

    all_results = []
    summary = {}

    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] Processing: {topic}")
        start = time.time()

        if args.input:
            records = load_questions_from_json(args.input, topic, args.max_records)
        else:
            records = load_questions_from_index(args.index, topic, args.max_records)

        if not records:
            print(f"  -> 0 records found, skipping")
            summary[topic] = {"total": 0, "with_subtopics": 0, "time_sec": 0}
            continue

        # Run async
        results = asyncio.get_event_loop().run_until_complete(run_topic_async(records, topic))
        all_results.extend(results)

        elapsed = time.time() - start
        with_sub = sum(1 for r in results if r.get("sub_topic"))
        summary[topic] = {"total": len(results), "with_subtopics": with_sub, "time_sec": round(elapsed)}
        print(f"  -> {len(results)} records, {with_sub} with subtopics ({round(elapsed)}s)")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"subtopic_only_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_all = 0
    total_with = 0
    for topic, stats in summary.items():
        total_all += stats["total"]
        total_with += stats["with_subtopics"]
        pct = round(stats["with_subtopics"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
        print(f"  {topic}: {stats['total']} records, {stats['with_subtopics']} with subtopics ({pct}%), {stats['time_sec']}s")

    overall_pct = round(total_with / total_all * 100, 1) if total_all > 0 else 0
    print(f"\n  TOTAL: {total_all} records, {total_with} with subtopics ({overall_pct}%)")
    print(f"\nResults saved to: {output_file}")
    print(f"\nGenerate report with:")
    print(f"  python generate_report.py --input {output_file} --max_per_topic 200")


if __name__ == "__main__":
    main()
