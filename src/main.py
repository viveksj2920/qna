import os
import time
import argparse
from qna_batch_processor import BatchProcessor
try:
    from index import IndexProcessor
except ImportError:
    IndexProcessor = None
from datetime import datetime, timedelta
import pandas as pd
from utils.logger_config import logger
import logging  
logging.getLogger("mlflow").setLevel(logging.ERROR)

def main():
    """
    Main function to execute the pipeline.
    """
    parser = argparse.ArgumentParser(description="Batch QnA processing")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--input_type", required=True, help="input_type could be index or csv file")
    parser.add_argument("--source_data_name", required=True, help="source_data_name can take source index name or input csv file path")
    parser.add_argument("--lookup_data_name", required=False, default="", help="lookup_data_name can take subtopic lookup index name or csv file path")
    parser.add_argument("--destination_data_name", required=False, default="", help="destination_data_name can take destination index name or csv file path")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--start_date", required=False, help="Start date (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--end_date", required=False, help="End date (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--file_input", required=False, help="conversations or questions")
    parser.add_argument("--scheduled", action="store_true", help="Run in scheduled mode to process last hour's data")
    parser.add_argument("--topic_filter", required=False, default="", help="Only process questions matching this topic (e.g., 'dental', 'enrollment'). Leave empty for all topics.")
    args = parser.parse_args()

    if args.input_type=="index":

        index_dict = {
            "dry_run": args.dry_run,
            "input_type": args.input_type,
            "source_index": args.source_data_name,
            "destination_index": args.destination_data_name,
            "lookup_index": args.lookup_data_name,
            "project": args.project,
            "topic_filter": args.topic_filter.lower().strip() if args.topic_filter else ""
        }

        # Handle date ranges based on CLI arguments
        if args.scheduled:
            # Get current time and round down to the previous hour
            now = datetime.now()
            current_hour = now.replace(minute=0, second=0, microsecond=0)
            previous_hour = current_hour - timedelta(hours=1)
            
            # Format dates for the filter
            start_date = previous_hour.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_date = current_hour.strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info(f"Scheduled mode: Processing data from {start_date} to {end_date}")
        elif args.start_date and args.end_date:
            start_date = args.start_date
            end_date = args.end_date
            logger.info(f"Using specified date range: {start_date} to {end_date}")
        else:
            # Default: Set tomorrow's date as end date and start date as 2 days before
            start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
            end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
            logger.info(f"Using default date range: {start_date} to {end_date}")
        
        datetime_object = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
        start_date_formatted = datetime_object.strftime('%Y-%m-%dT%H:%M:%SZ')

        datetime_object = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
        # Make end date inclusive if not in scheduled mode
        if not args.scheduled:
            datetime_object = datetime_object + timedelta(days=1)
        end_date_formatted = datetime_object.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Set filter to extract records in the date range and not processed yet
        date_filter = f"StartTime ge {start_date_formatted} and StartTime lt {end_date_formatted} and metadata_processed_time ne null and Text ne ''"

        # create an index processor object
        index_processor = IndexProcessor(index_name=index_dict['source_index'])
        project = index_dict['project']
        logger.info(f"Processing input_type: {index_dict['input_type']}")
        logger.info(f"Processing project: {index_dict['project']}")
        logger.info(f"Using date filter: {date_filter}")

        df = index_processor.fetch_records(date_filter, project)

        logger.info(f"Total records processing: {len(df)}")
        logger.info(f"Columns in the DataFrame: {df.columns.tolist()}")

        if not df.empty:
            logger.info("Batch processing started")
            start_time = time.time()
            batch_processor = BatchProcessor(dataframe=df, input_dict=index_dict)
            batch_processor.process()
            end_time = time.time()
            logger.info("batch QnA metadata processed")
            logger.info(f"Total run time: {(end_time - start_time) / 60:.2f} minutes")
        else:
            logger.info("No records found to process.")

    elif args.input_type == "file":

        file_dict = {
            "input_type": args.input_type,
            "source_csv": args.source_data_name,
            "destination_csv": args.destination_data_name,
            "project": args.project,
            "file_input": args.file_input,
            "topic_filter": args.topic_filter.lower().strip() if args.topic_filter else ""
        }

        logger.info(f"Processing input_type: {file_dict['input_type']}")
        logger.info(f"Processing project: {file_dict['project']}")

        df = pd.read_csv(file_dict['source_csv'])
        logger.info(f"Total records processing: {len(df)}")
        logger.info(f"Columns in the DataFrame: {df.columns.tolist()}")

        if not df.empty:
            logger.info("Batch processing started")
            start_time = time.time()
            batch_processor = BatchProcessor(dataframe=df, input_dict=file_dict)
            batch_processor.process()
            end_time = time.time()
            logger.info("batch QnA metadata processed")
            logger.info(f"Total run time: {(end_time - start_time) / 60:.2f} minutes")
        else:
            logger.info("No records found to process.")


    else:
        logger.info("Invalid input type. Please use 'index' or 'csv'.")


if __name__ == "__main__":
    main()