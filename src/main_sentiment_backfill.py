import argparse
from utils.logger_config import logger
from datetime import datetime, timedelta
from qna_batch_processor_sentiment_backfill import BatchProcessor
from index_sentiment_backfill import IndexProcessor
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", type=str, required=True)
    parser.add_argument("--source_data_name", type=str, required=True)
    parser.add_argument("--destination_data_name", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--start_date", required=False)
    parser.add_argument("--end_date", required=False)
    parser.add_argument("--file_input", required=False)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--sentiment_filter", required=False)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logger.info("Starting Sentiment Backfill Pipeline")
    logger.info(f"Input Type: {args.input_type}")
    logger.info(f"source Index name: {args.source_data_name}")
    logger.info(f"Destination Index name: {args.destination_data_name}")
    logger.info(f"project: {args.project}")
    logger.info(f"start_date: {args.start_date}")
    logger.info(f"end_date: {args.end_date}")
    logger.info(f"batch_size: {args.batch_size}")
    logger.info(f"sentiment_filter: {args.sentiment_filter}")
    logger.info(f"dry_run: {args.dry_run}")

    if args.dry_run:
            logger.info("DRY RUN MODE: No changes will be made to the data")

    if args.input_type == "index":
        process_index_sentiment_backfill(args)
    elif args.input_type == "file":
        process_file_sentiment_backfill(args)
    else:
        logger.error("Invalid input type. Please use 'index' or 'file'.")
        return

def process_index_sentiment_backfill(args):
    """Process sentiment backfill for index-based data"""
    
    index_dict = {
        "input_type": args.input_type,
        "source_index": args.source_data_name,
        "destination_index": args.destination_data_name,
        "project": args.project,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run
    }

    # Handle date range
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
        logger.info(f"Date range: {start_date} to {end_date}")
    else:
        # Default: last 7 days for sentiment backfill
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
        logger.info(f"Default date range: {start_date} to {end_date}")

    # Format dates
    datetime_object = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
    start_date_formatted = datetime_object.strftime('%Y-%m-%dT%H:%M:%SZ')

    datetime_object = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
    end_date_object = datetime_object
    end_date_formatted = end_date_object.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Build filter based on sentiment status
    date_filter = f"StartTime ge {start_date_formatted} and StartTime lt {end_date_formatted}"
    
    if args.sentiment_filter == "missing":
        # Only process records without sentiment data
        date_filter += " and (sentiment_level_1 eq null or sentiment_level_1 eq '')"
        logger.info("Processing only records missing sentiment data")
    elif args.sentiment_filter == "all":
        logger.info("Processing all records in date range")
    else:
        # Default: missing sentiment data
        date_filter += " and (sentiment_level_1 eq null or sentiment_level_1 eq '')"
        logger.info("Default: Processing only records missing sentiment data")

    logger.info(f"Applied filter: {date_filter}")

    # Create index processor
    index_processor = IndexProcessor(index_name=index_dict['source_index'])
    project = index_dict['project']

    # Fetch records
    df = index_processor.fetch_records_for_sentiment(date_filter, project,index_dict['batch_size'])
    logger.info(f"Total records fetched: {len(df):,}")
    
    if df.empty:
        logger.info("No records found to process.")
        return

    logger.info(f"Columns in the DataFrame: {df.columns.tolist()}")

    # Validate required columns
    required_columns = ['question'] if project == "MIRA" else ['question']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return

    # Process records in batches
    logger.info("Starting batch sentiment processing")
    start_time = time.time()
    
    try:
        batch_processor = BatchProcessor(dataframe=df, input_dict=index_dict)
        results = batch_processor.process()
        
        end_time = time.time()
        processing_time = (end_time - start_time) / 60
        
        logger.info("Sentiment backfill processing completed")
        logger.info(f"Total processing time: {processing_time:.2f} minutes")
        logger.info(f"Average time per record: {(processing_time * 60 / len(df)):.2f} seconds")
        
        if results:
            logger.info(f"Successfully processed: {results.get('processed', 0)} records")
            logger.info(f"Failed: {results.get('failed', 0)} records")
            logger.info(f"Skipped: {results.get('skipped', 0)} records")
            
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        raise

if __name__ == "__main__":
    main()