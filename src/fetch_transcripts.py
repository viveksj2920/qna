"""
Fast transcript fetcher using Azure Search REST API directly.
Bypasses the AzureSearchIndexUtility wrapper that downloads ALL records.

Fetches exactly N records and saves as CSV for file-based pipeline processing.

Usage:
  python fetch_transcripts.py --max_records 5000 --start_date "2026-04-01" --end_date "2026-05-15"
  python fetch_transcripts.py --max_records 1000 --start_date "2026-05-01" --end_date "2026-05-15"
"""

import argparse
import json
import os
import csv
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
load_dotenv()


SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT_EASTUS")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY_EASTUS")
INDEX_NAME = "transcripts-mira"
API_VERSION = "2023-11-01"

FIELDS = [
    "Ucid", "Text", "StartTime", "Is_Digital", "Is_Enrollment",
    "plan_name", "drugs", "providers", "zip",
    "county_processed", "state_processed", "region_processed", "subregion_processed"
]


def fetch_records(start_date, end_date, max_records, batch_size=1000):
    """Fetch records using REST API with pagination control."""
    url = f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version={API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": SEARCH_API_KEY
    }

    start_str = f"{start_date}T00:00:00Z"
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    end_str = end_dt.strftime("%Y-%m-%dT00:00:00Z")

    filter_str = (
        f"StartTime ge {start_str} and StartTime lt {end_str} "
        f"and metadata_processed_time ne null and Text ne ''"
    )

    all_records = []
    skip = 0

    while len(all_records) < max_records:
        remaining = max_records - len(all_records)
        top = min(batch_size, remaining)

        body = {
            "search": "*",
            "filter": filter_str,
            "select": ",".join(FIELDS),
            "top": top,
            "skip": skip,
            "count": True,
            "queryType": "simple"
        }

        print(f"  Fetching {skip} to {skip + top}...", end=" ")
        resp = requests.post(url, headers=headers, json=body)

        if resp.status_code != 200:
            print(f"ERROR: {resp.status_code} - {resp.text[:200]}")
            break

        data = resp.json()
        docs = data.get("value", [])

        if not docs:
            print("no more records")
            break

        all_records.extend(docs)
        total_available = data.get("@odata.count", "?")
        print(f"got {len(docs)} (total so far: {len(all_records)}, available: {total_available})")

        skip += len(docs)

        if len(docs) < top:
            break

    return all_records


def save_as_csv(records, output_path):
    """Save records as CSV compatible with main.py --input_type file."""
    if not records:
        print("No records to save")
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for rec in records:
            # Clean up Azure Search metadata fields
            clean = {k: rec.get(k, "") for k in FIELDS}
            writer.writerow(clean)

    print(f"Saved {len(records)} records to {output_path}")


def save_as_json(records, output_path):
    """Save records as JSON."""
    clean_records = []
    for rec in records:
        clean = {k: rec.get(k, "") for k in FIELDS}
        clean_records.append(clean)

    with open(output_path, 'w') as f:
        json.dump(clean_records, f, indent=2)

    print(f"Saved {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fast transcript fetcher via REST API")
    parser.add_argument("--max_records", type=int, default=5000, help="Max records to fetch (default: 5000)")
    parser.add_argument("--start_date", default="2026-04-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", default="2026-05-15", help="End date (YYYY-MM-DD)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Records per API call (default: 1000)")
    args = parser.parse_args()

    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching up to {args.max_records} transcripts")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Endpoint: {SEARCH_ENDPOINT}")
    print(f"Index: {INDEX_NAME}")
    print()

    records = fetch_records(args.start_date, args.end_date, args.max_records, args.batch_size)

    if records:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(output_dir, f"transcripts_raw_{timestamp}.csv")
        save_as_csv(records, csv_path)

        print(f"\nNow run the pipeline on this file:")
        print(f"  python main.py --dry_run --input_type=\"file\" --source_data_name=\"{csv_path}\" --destination_data_name=\"data/output/qna_output_{timestamp}.csv\" --project=\"MIRA\" --file_input=\"conversations\" --lookup_data_name=\"transcripts-qna-subtopic-groupings-test-1\"")
    else:
        print("No records fetched.")


if __name__ == "__main__":
    main()
