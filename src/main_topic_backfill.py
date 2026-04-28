import os
import time
import argparse
from qna_batch_processor_topic_backfill import BatchProcessor
from index_topic_backfill import IndexProcessor
from datetime import datetime, timedelta
import pandas as pd
from utils.logger_config import logger

def main():
    """
    Main function to execute the pipeline.
    """
    parser = argparse.ArgumentParser(description="Batch QnA processing")
    parser.add_argument("--input_type", required=True, help="input_type could be index or csv file")
    parser.add_argument("--source_data_name", required=True, help="source_data_name can take source index name or input csv file path")
    parser.add_argument("--lookup_data_name", required=False, help="lookup_data_name for subtopic groupings index")
    parser.add_argument("--destination_data_name", required=True, help="destination_data_name can take destination index name or csv file path")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--start_date", required=False, help="Start date (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--end_date", required=False, help="End date (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--file_input", required=False, help="conversations or questions")
    args = parser.parse_args()

    if args.input_type=="index":

        index_dict = {
            "input_type": args.input_type,
            "source_index": args.source_data_name,
            "lookup_index": args.lookup_data_name,
            "destination_index": args.destination_data_name,
            "project": args.project
        }

        # Accept command line arguments for start date and end date
        if args.start_date and args.end_date:
            start_date = args.start_date
            end_date = args.end_date
            logger.info(f"Start date: {start_date}, End date: {end_date}")
        else:
            # Set tomorrow's date as end date including time as T00:00:00Z and start date as 2 days before including time as T00:00:00Z
            start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
            end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
            logger.info(f"Default Start date: {start_date}, Default End date: {end_date}")
        
        datetime_object = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
        start_date_formatted = datetime_object.strftime('%Y-%m-%dT%H:%M:%SZ')

        datetime_object = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
        # make the end date inclusive by adding 1 day
        end_date_object = datetime_object + timedelta(days=1)
        end_date_formatted = end_date_object.strftime('%Y-%m-%dT%H:%M:%SZ')


        csv_file_path = 'data/input/source_topics.csv'
        df_topics = pd.read_csv(csv_file_path)
        unique_topics = df_topics['source_topics'].unique()
        filter_parts = [f"topic eq '{topic}'" for topic in unique_topics]
        topics_filter = " or ".join(filter_parts)
        date_filter = f"StartTime ge {start_date_formatted} and StartTime lt {end_date_formatted} and ({topics_filter})"
        print(f"date and topics filter: {date_filter}")

        # create an index processor object
        index_processor = IndexProcessor(index_name=index_dict['source_index'])
        project = index_dict['project']
        logger.info(f"Processing input_type: {index_dict['input_type']}")
        logger.info(f"Processing project: {index_dict['project']}")

        df = index_processor.fetch_records(date_filter, project)
        df = df[['id', 'Ucid', 'question']]

        #print unique count of Ucid in the dataframe
        unique_ucid_count = df['Ucid'].nunique()
        logger.info(f"Unique Ucid count in the input DataFrame: {unique_ucid_count}")

        logger.info(f"Total records processing: {len(df)}")
        logger.info(f"Columns in the input DataFrame: {df.columns.tolist()}")

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

    else:
        logger.info("Invalid input type. Please use 'index'.")


if __name__ == "__main__":
    main()