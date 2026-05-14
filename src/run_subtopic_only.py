"""
Subtopic-only extraction script.

Reads questions that already have topics assigned from the destination QnA index,
runs ONLY the subtopic extraction step (skips Q&A extraction and topic assignment),
and generates a report.

This is much faster than the full pipeline since it skips LLM calls #1 (Q&A extraction)
and #2 (topic assignment), only doing LLM call #3 (subtopic) per question.

Usage:
  python run_subtopic_only.py --topic "enrollment" --max_records 200
  python run_subtopic_only.py --topic "all" --max_records 200
  python run_subtopic_only.py --topic "enrollment,eligibility" --max_records 200
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

import nest_asyncio
nest_asyncio.apply()

from utils.logger_config import logger
from utils.helper import load_project_config, clean_topic
from prompts.prompt_config import prompt_sub_topic_format, sub_topic_extraction_prompt
from llm.llm_config import chat_completion
from qna_extractor import QnAExtractor

try:
    from index import IndexProcessor
except ImportError:
    IndexProcessor = None


QNA_INDEX = "transcript-index-test-qna-v1"
PROJECT = "MIRA"


def fetch_questions_by_topic(topic, max_records=200, index_name=QNA_INDEX):
    """Fetch questions from the QnA destination index that already have a topic assigned."""
    if IndexProcessor is None:
        logger.error("IndexProcessor not available. Cannot query index.")
        return []

    index_processor = IndexProcessor(index_name=index_name)

    # Query for records with this topic that have questions
    topic_filter = f"topic eq '{topic}' and question ne ''"
    fields = ["Ucid", "question", "answer", "topic", "sub_topic", "is_useful"]

    try:
        documents = index_processor.azure_search.search(
            filter=topic_filter,
            select=fields,
            top=max_records
        )
        records = list(documents)
        logger.info(f"Fetched {len(records)} records for topic '{topic}'")
        return records[:max_records]
    except Exception as e:
        logger.error(f"Error fetching records for topic '{topic}': {e}")
        return []


def extract_subtopic_sync(question, topic, project="MIRA"):
    """Run subtopic extraction for a single question (synchronous)."""
    try:
        topic_cleaned = clean_topic(project, topic)
        sub_topic_prompt_dict = load_project_config(project, "new_subtopic_extraction")

        if topic_cleaned not in sub_topic_prompt_dict:
            logger.warning(f"Topic '{topic_cleaned}' not found in subtopic config")
            return []

        sub_topic_prompt = sub_topic_prompt_dict[topic_cleaned]
        sub_topic_prompt = prompt_sub_topic_format(project, sub_topic_prompt)

        prompt = sub_topic_extraction_prompt(
            question=question, topic=topic, sub_topic_descriptions=sub_topic_prompt
        )

        cleaned_lines = [line.lstrip() for line in prompt.splitlines()]
        prompt = "\n".join(cleaned_lines)

        message = [{"role": "user", "content": prompt}]
        result = chat_completion(messages=message, max_tokens=2000, temperature=1e-9, task_type="sub_topic_extraction")

        if result:
            parsed = json.loads(result) if isinstance(result, str) else result
            return parsed.get("sub_topic", [])
    except Exception as e:
        logger.error(f"Error extracting subtopic: {e}")

    return []


def run_topic(topic, max_records, output_dir, index_name=QNA_INDEX):
    """Run subtopic extraction for all questions under a given topic."""
    logger.info(f"Processing topic: {topic} (max {max_records} records)")

    records = fetch_questions_by_topic(topic, max_records, index_name)
    if not records:
        logger.info(f"No records found for topic '{topic}'")
        return []

    results = []
    total = len(records)

    for i, rec in enumerate(records, 1):
        question = rec.get("question", "")
        if not question:
            continue

        logger.info(f"[{i}/{total}] Extracting subtopics for: {question[:80]}...")

        new_subtopics = extract_subtopic_sync(question, topic, PROJECT)

        results.append({
            "Ucid": rec.get("Ucid", ""),
            "question": question,
            "answer": rec.get("answer", ""),
            "topic": topic,
            "sub_topic": new_subtopics,
            "old_sub_topic": rec.get("sub_topic", []),
            "is_useful": rec.get("is_useful", ""),
        })

        if i % 10 == 0:
            logger.info(f"  Progress: {i}/{total} done")

    return results


def main():
    parser = argparse.ArgumentParser(description="Subtopic-only extraction from QnA index")
    parser.add_argument("--topic", required=True, help="Topic to process ('all' for all topics, or comma-separated)")
    parser.add_argument("--max_records", type=int, default=200, help="Max records per topic (default: 200)")
    parser.add_argument("--index", default=QNA_INDEX, help="QnA index name")
    args = parser.parse_args()

    qna_index = args.index

    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    # Determine which topics to run
    if args.topic.lower() == "all":
        sub_topic_config = load_project_config(PROJECT, "new_subtopic_extraction")
        topics = list(sub_topic_config.keys())
    else:
        topics = [t.strip() for t in args.topic.split(",")]

    print(f"Topics to process: {', '.join(topics)}")
    print(f"Max records per topic: {args.max_records}")
    print(f"Index: {qna_index}")

    all_results = []
    summary = {}

    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] Processing: {topic}")
        start = time.time()

        results = run_topic(topic, args.max_records, output_dir, qna_index)
        all_results.extend(results)

        elapsed = time.time() - start
        with_sub = sum(1 for r in results if r.get("sub_topic"))
        summary[topic] = {"total": len(results), "with_subtopics": with_sub, "time_sec": round(elapsed)}
        print(f"  -> {len(results)} records, {with_sub} with subtopics ({round(elapsed)}s)")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"subtopic_only_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
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
