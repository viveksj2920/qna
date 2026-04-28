#!/usr/bin/env python3
"""
Sentiment Batch Processor
Handles batch processing of sentiment extraction for QnA records.
Optimized for concurrent processing with error handling and progress tracking.
"""

import pandas as pd
import concurrent.futures
import time
from datetime import datetime
from qna_extractor_sentiment_backfill import qna_extractor
from utils.logger_config import logger

class BatchProcessor:
    """
    Processes batches of QnA records for sentiment extraction.
    Supports both index and file-based processing with concurrent execution.
    """

    def __init__(self, dataframe, input_dict):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("The input should be a Pandas DataFrame")

        self.dataframe = dataframe
        self.input_dict = input_dict
        self.batch_size = input_dict.get('batch_size', 10000)
        self.dry_run = input_dict.get('dry_run', False)
        self.project = input_dict.get('project')
        
        # Performance tracking
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.start_time = None

    def process(self):
        """
        Main processing method that orchestrates the sentiment extraction.
        Returns: Dictionary with processing statistics
        """
        self.start_time = time.time()
        total_records = len(self.dataframe)
        
        logger.info(f"Starting sentiment batch processing for {total_records:,} records")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Project: {self.project}")
        logger.info(f"Dry run mode: {self.dry_run}")

        if self.input_dict['input_type'] == 'index':
            return self._process_index_records()
        elif self.input_dict['input_type'] == 'file':
            return self._process_file_records()
        else:
            raise ValueError("Unsupported input type")

    def _process_index_records(self):
        """Process records from Azure Search index"""
        #logger.info("Processing index records for sentiment extraction")
        
        qna = qna_extractor()
        
        try:
            # Process records in batches
            batch_results = []
            total_batches = (len(self.dataframe) + self.batch_size - 1) // self.batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min((batch_num + 1) * self.batch_size, len(self.dataframe))
                batch_df = self.dataframe.iloc[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_num + 1}/{total_batches} "
                           f"(records {start_idx + 1}-{end_idx})")
                
                batch_result = self._process_sentiment_batch(batch_df, qna)
                #batch_results.extend(batch_result)
                
                # Log progress
                self._log_progress(batch_num + 1, total_batches)
            
                # Update index with results if not dry run
                if not self.dry_run and batch_result:
                    self._update_index_with_sentiments(batch_result)
                
                #return 
                self._get_processing_stats()
            
        except Exception as e:
            logger.error(f"Error during index processing: {e}")
            raise

    def _process_file_records(self):
        """Process records from CSV file"""
        logger.info("Processing file records for sentiment extraction")
        
        qna = qna_extractor()
        
        try:
            # Add sentiment columns if they don't exist
            if 'sentiment_level_1' not in self.dataframe.columns:
                self.dataframe['sentiment_level_1'] = None
            if 'sentiment_level_2' not in self.dataframe.columns:
                self.dataframe['sentiment_level_2'] = None
            
            # Process records with concurrent execution
            total_batches = (len(self.dataframe) + self.batch_size - 1) // self.batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min((batch_num + 1) * self.batch_size, len(self.dataframe))
                batch_df = self.dataframe.iloc[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_num + 1}/{total_batches} "
                           f"(records {start_idx + 1}-{end_idx})")
                
                self._process_file_batch_sentiments(batch_df, start_idx)
                
                # Log progress
                self._log_progress(batch_num + 1, total_batches)
            
            # Save results if not dry run
            if not self.dry_run:
                self._save_file_results()
            
            return self._get_processing_stats()
            
        except Exception as e:
            logger.error(f"Error during file processing: {e}")
            raise

    def _process_sentiment_batch(self, batch_df, qna_extractor_instance):
        """Process a batch of records for sentiment extraction"""
        batch_results = []
        
        # Use concurrent processing for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
            future_to_row = {
                executor.submit(self._extract_sentiment_for_record, row, idx): (idx, row)
                for idx, row in batch_df.iterrows()
            }
            
            for future in concurrent.futures.as_completed(future_to_row):
                idx, row = future_to_row[future]
                try:
                    result = future.result(timeout=60)  # 60 seconds timeout per record
                    if result:
                        batch_results.append(result)
                        self.processed_count += 1
                    else:
                        self.skipped_count += 1
                        
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Timeout processing record {idx}")
                    self.failed_count += 1
                except Exception as e:
                    logger.error(f"Error processing record {idx}: {e}")
                    self.failed_count += 1
        
        return batch_results

    def _extract_sentiment_for_record(self, row, idx):
        """Extract sentiment for a single record"""
        try:
            question = row.get('question', '')
            if not question or pd.isna(question):
                logger.warning(f"Empty question for record {idx}")
                return None
            
            # Check if sentiment already exists (unless we're reprocessing all)
            existing_sentiment = row.get('sentiment_level_1')
            if existing_sentiment and not pd.isna(existing_sentiment) and existing_sentiment != '':
                # Skip if sentiment already exists and we're only processing missing ones
                filter_mode = self.input_dict.get('sentiment_filter', 'missing')
                if filter_mode == 'missing':
                    return None
            
            # Extract sentiments
            extractor = qna_extractor(question=question, project=self.project)
            
            # Level 1 sentiment
            sentiment_1_json = extractor.extract_sentiment_level_one()
            if not sentiment_1_json:
                logger.warning(f"Failed to extract level 1 sentiment for record {idx}")
                return None
            
            # Parse level 1 result
            import json
            try:
                cleaned_sentiment_1 = sentiment_1_json.strip().replace("```json", "").replace("```", "").strip()
                sentiment_1_data = json.loads(cleaned_sentiment_1)
                sentiment_level_1 = sentiment_1_data.get('sentiment_level_1')
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing level 1 sentiment JSON for record {idx}: {e}")
                return None
            
            # Level 2 sentiment
            sentiment_level_2 = None
            if sentiment_level_1:
                sentiment_2_json = extractor.extract_sentiment_level_two(sentiment_level_1)
                if sentiment_2_json:
                    try:
                        cleaned_sentiment_2 = sentiment_2_json.strip().replace("```json", "").replace("```", "").strip()
                        sentiment_2_data = json.loads(cleaned_sentiment_2)
                        sentiment_level_2 = sentiment_2_data.get('sentiment_level_2')
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing level 2 sentiment JSON for record {idx}: {e}")
            
            # Prepare result for index update
            result = {
                'id': row.get('id'),  # Required for index updates
                'sentiment_level_1': sentiment_level_1,
                'sentiment_level_2': sentiment_level_2,
                '@search.action': 'mergeOrUpload'
            }
            
            # Add other required fields based on project
            if self.project == "MIRA":
                result.update({
                    'Ucid': row.get('Ucid'),
                    'question': row.get('question'),
                    'answer': row.get('answer'),
                    'StartTime': row.get('StartTime'),
                    'Is_Digital': row.get('Is_Digital'),
                    'Is_Enrollment': row.get('Is_Enrollment')
                })
            elif self.project == "PCL":
                result.update({
                    'Ucid': row.get('Ucid'),
                    'question': row.get('question'),
                    'answer': row.get('answer'),
                    'StartTime': row.get('StartTime')
                })
            
            logger.debug(f"Extracted sentiments for record {idx}: L1={sentiment_level_1}, L2={sentiment_level_2}")
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error extracting sentiment for record {idx}: {e}")
            return None

    def _process_file_batch_sentiments(self, batch_df, start_idx):
        """Process sentiment extraction for file-based batch"""
        for relative_idx, (idx, row) in enumerate(batch_df.iterrows()):
            try:
                question = row.get('question', '')
                if not question or pd.isna(question):
                    continue
                
                # Extract sentiments
                extractor = qna_extractor(question=question, project=self.project)
                
                # Level 1 sentiment
                sentiment_1_json = extractor.extract_sentiment_level_one()
                if sentiment_1_json:
                    import json
                    try:
                        cleaned_sentiment_1 = sentiment_1_json.strip().replace("```json", "").replace("```", "").strip()
                        sentiment_1_data = json.loads(cleaned_sentiment_1)
                        sentiment_level_1 = sentiment_1_data.get('sentiment_level_1')
                        
                        # Update DataFrame
                        self.dataframe.at[idx, 'sentiment_level_1'] = sentiment_level_1
                        
                        # Level 2 sentiment
                        if sentiment_level_1:
                            sentiment_2_json = extractor.extract_sentiment_level_two(sentiment_level_1)
                            if sentiment_2_json:
                                try:
                                    cleaned_sentiment_2 = sentiment_2_json.strip().replace("```json", "").replace("```", "").strip()
                                    sentiment_2_data = json.loads(cleaned_sentiment_2)
                                    sentiment_level_2 = sentiment_2_data.get('sentiment_level_2')
                                    self.dataframe.at[idx, 'sentiment_level_2'] = sentiment_level_2
                                except json.JSONDecodeError:
                                    pass
                        
                        self.processed_count += 1
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing sentiment JSON for file record {idx}: {e}")
                        self.failed_count += 1
                else:
                    self.failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing file record {idx}: {e}")
                self.failed_count += 1

    def _update_index_with_sentiments(self, results):
        """Update Azure Search index with sentiment results"""
        try:
            from index_sentiment_backfill import IndexProcessor
            
            index_processor = IndexProcessor(index_name=self.input_dict['destination_index'])
            
            logger.info(f"Updating index with {len(results):,} sentiment results")
            
            # Process in chunks to avoid overwhelming the index
            chunk_size = 100
            #for i in range(0, len(results), chunk_size):
                #chunk = results[i:i + chunk_size]
            index_processor.update_index(results, key_field_name="id")
            #logger.debug(f"Updated index chunk {i//chunk_size + 1}")
            logger.info("Index update completed successfully")
            
        except Exception as e:
            logger.error(f"Error updating index: {e}")
            raise

    def _save_file_results(self):
        """Save file processing results to CSV"""
        try:
            output_path = self.input_dict['destination_csv']
            self.dataframe.to_csv(output_path, index=False)
            logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving file results: {e}")
            raise

    def _log_progress(self, current_batch, total_batches):
        """Log processing progress"""
        # Handle case where start_time might not be set (e.g., in tests)
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
        else:
            elapsed_time = 0.0
            
        progress_pct = (current_batch / total_batches) * 100
        
        logger.info(f"Progress: {current_batch}/{total_batches} batches ({progress_pct:.1f}%) - "
                   f"Processed: {self.processed_count:,}, Failed: {self.failed_count:,}, "
                   f"Skipped: {self.skipped_count:,}, Elapsed: {elapsed_time/60:.1f}m")

    def _get_processing_stats(self):
        """Get processing statistics"""
        # Handle case where start_time might not be set (e.g., in tests)
        if self.start_time is not None:
            total_time = time.time() - self.start_time
        else:
            total_time = 0.0
        
        stats = {
            'processed': self.processed_count,
            'failed': self.failed_count,
            'skipped': self.skipped_count,
            'total_time_minutes': total_time / 60,
            'records_per_minute': self.processed_count / (total_time / 60) if total_time > 0 else 0
        }
        
        logger.info(f"Processing completed - Processed: {stats['processed']:,}, "
                   f"Failed: {stats['failed']:,}, Skipped: {stats['skipped']:,}, "
                   f"Rate: {stats['records_per_minute']:.1f} records/min")
        
        return stats