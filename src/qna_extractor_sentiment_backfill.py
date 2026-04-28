#!/usr/bin/env python3
"""
Sentiment QnA Extractor - Backfill Version
Specialized version of qna_extractor optimized for sentiment backfill operations.
Focuses on sentiment extraction with enhanced error handling and performance optimizations.
"""

# Standard Library Imports
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import time

# Third-Party Imports
from azure.core.exceptions import HttpResponseError
from dotenv import load_dotenv
from openai import AzureOpenAI
import pandas as pd

# Local Application Imports
from index_sentiment_backfill import IndexProcessor
from skillsets.vectorizer import AzureOpenAIVectorizer

# Import the Prompts
from prompts.prompt_config import (
    qna_extraction_prompt, 
    prompt_topic_format, 
    topic_extraction_prompt, 
    prompt_sub_topic_format, 
    sub_topic_extraction_prompt, 
    is_useful_question_extraction_prompt,
    sentiment_level_1_prompt,
    sentiment_level_2_prompt
)

# Configure the LLM
from llm.llm_config import chat_completion

from utils.helper import load_project_config, write_to_csv, clean_topic
from utils.logger_config import logger
import config

class qna_extractor:
    """
    Enhanced QnA extractor specialized for sentiment analysis backfill operations.
    Optimized for batch processing with improved error handling and performance.
    """

    def __init__(self, text=None, question=None, topic=None, project=None):
        self.text = text
        self.qna = {}
        self.question = question
        self.topic = topic
        self.project = project

    def clean_text(self):
        """Clean and normalize text for processing"""
        text = self.text

        # Remove unwanted characters and extra spaces
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'\*+', '****', text)  # Normalize redacted PII to a consistent format
        text = re.sub(r'[^\w\s,.!?\'-]', '', text)  # Remove any non-alphanumeric characters except punctuation

        # Remove back ticks and new lines and quotes and curly braces
        text = text.replace("`", " ")
        text = text.replace("\n", " ")
        text = text.replace("'", " ")
        text = text.replace('"', " ")
        text = text.replace("{", " ")

        # Ensure proper formatting
        text = text.strip()  # Remove leading and trailing spaces
        
        return text

    def extract_qna(self):
        """Extract Q&A pairs from conversation text"""
        if self.text:
            extractor = qna_extractor(self.text)
            chunk = extractor.clean_text()
            try:
                prompt = qna_extraction_prompt(chunk)
            except KeyError as e:
                logger.error(f"Error formatting Question Extraction prompt: Missing key {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error while formatting prompt: {e}")
                raise

            message = [{"role": "user", "content": prompt}]
            #logger.debug(f"QnA Extraction Prompt:\n{prompt}\n")

            self.qna = chat_completion(messages=message, max_tokens=2000, temperature=1e-9, task_type="qna_extraction")

        return self.qna

    def extract_sentiment_level_one(self):
        """
        Extract high-level sentiment (Positive, Neutral, Negative) for a question.
        Optimized for backfill operations with enhanced error handling.
        Returns: JSON string with sentiment_level_1
        """
        sentiment_level_1 = None
        if self.question:
            try:
                # Load sentiment categories from config
                sentiment_config = load_project_config(self.project, "sentiment_extraction")
                level_1_categories = sentiment_config.get("level_1", {})
                
                prompt = sentiment_level_1_prompt(self.question, level_1_categories)
                
                message = [{"role": "user", "content": prompt}]
                
                sentiment_level_1 = chat_completion(
                    messages=message, 
                    max_tokens=500, 
                    temperature=1e-9, 
                    task_type="sentiment_level_1_extraction"
                )
                
                #logger.debug(f"Level 1 sentiment extracted for question: '{self.question[:50]}...'")
                
            except Exception as e:
                logger.error(f"Error in extract_sentiment_level_one: {e}")
                return None
            
        return sentiment_level_1

    def extract_sentiment_level_two(self, sentiment_level_1):
        """
        Extract detailed sentiment subcategory based on level 1 sentiment.
        Optimized for backfill operations with enhanced error handling.
        Args:
            sentiment_level_1: The high-level sentiment (Positive, Neutral, Negative)
        Returns: JSON string with sentiment_level_2
        """
        sentiment_level_2 = None
        if self.question and sentiment_level_1:
            try:
                # Load sentiment categories from config
                sentiment_config = load_project_config(self.project, "sentiment_extraction")
                level_2_categories = sentiment_config.get("level_2", {}).get(sentiment_level_1.lower(), {})
                
                prompt = sentiment_level_2_prompt(
                    self.question, 
                    sentiment_level_1, 
                    level_2_categories
                )
                
                message = [{"role": "user", "content": prompt}]
                
                sentiment_level_2 = chat_completion(
                    messages=message, 
                    max_tokens=500, 
                    temperature=1e-9, 
                    task_type="sentiment_level_2_extraction"
                )
                
                #logger.debug(f"Level 2 sentiment extracted: {sentiment_level_1} -> {sentiment_level_2}")
                
            except Exception as e:
                logger.error(f"Error in extract_sentiment_level_two: {e}")
                return None
            
        return sentiment_level_2

    def extract_topic(self):
        """Extract topic from question (included for compatibility)"""
        if self.question:
            question = self.question
            topic_descriptions_json = load_project_config(self.project, "topic_extraction")
            try:
                topic_descriptions = prompt_topic_format(self.project, topic_descriptions_json)
                prompt = topic_extraction_prompt(question, topic_descriptions)

                cleaned_lines = [line.lstrip() for line in prompt.splitlines()]
                prompt = "\n".join(cleaned_lines)

            except KeyError as e:
                logger.error(f"Error formatting prompt: Missing key {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error while formatting prompt: {e}")
                raise

            message = [{"role": "user", "content": prompt}]
            self.topic = chat_completion(messages=message, max_tokens=2000, temperature=1e-9, task_type="topic_extraction") 

        return self.topic

    def extract_subtopic(self):
        """Extract subtopic from question and topic (included for compatibility)"""
        sub_topic_prompt_dict = None
        sub_topic = None
        if self.question and self.topic:
            topic_cleaned = clean_topic(self.project, self.topic)
            sub_topic_prompt_dict = load_project_config(self.project, "subtopic_extraction")
            
            sub_topic_prompt = sub_topic_prompt_dict[topic_cleaned]
            sub_topic_prompt = prompt_sub_topic_format(self.project, sub_topic_prompt)
            
            prompt = sub_topic_extraction_prompt(question=self.question, topic=self.topic, sub_topic_descriptions=sub_topic_prompt)

            cleaned_lines = [line.lstrip() for line in prompt.splitlines()]
            prompt = "\n".join(cleaned_lines)
            
            message = [{"role": "user", "content": prompt}]
    
            sub_topic = chat_completion(messages=message, max_tokens=2000, temperature=1e-9, task_type="sub_topic_extraction")
            
        return sub_topic

    def extract_is_useful(self):
        """Extract usefulness classification (included for compatibility)"""
        is_useful = None
        if self.question:
            question_json_data = load_project_config(self.project, "useful_questions")
            prompt = is_useful_question_extraction_prompt(self.question, question_json_data)

            message = [{"role": "user", "content": prompt}]
            is_useful = chat_completion(messages=message, max_tokens=2000, temperature=1e-9, task_type="is_useful_extraction")

        return is_useful

    def process_sentiment_batch_records(self, records_batch, input_dict):
        """
        Process a batch of records for sentiment extraction.
        Optimized for concurrent processing with proper error handling.
        
        Args:
            records_batch: List of record dictionaries
            input_dict: Configuration dictionary
        
        Returns:
            List of processed records with sentiment data
        """
        processed_records = []
        max_workers = min(5, len(records_batch))  # Limit concurrent workers
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all records for processing
            future_to_record = {
                executor.submit(self._process_single_record_sentiment, record, input_dict): record
                for record in records_batch
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    result = future.result(timeout=120)  # 2 minute timeout per record
                    if result:
                        processed_records.append(result)
                        
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Timeout processing record: {record.get('id', 'unknown')}")
                except Exception as e:
                    logger.error(f"Error processing record {record.get('id', 'unknown')}: {e}")
        
        return processed_records

    def _process_single_record_sentiment(self, record, input_dict):
        """
        Process a single record for sentiment extraction.
        
        Args:
            record: Record dictionary containing question and metadata
            input_dict: Configuration dictionary
        
        Returns:
            Updated record with sentiment information
        """
        try:
            question = record.get('question', '')
            if not question or pd.isna(question):
                logger.warning(f"No question found in record: {record.get('id', 'unknown')}")
                return None
            
            project = input_dict.get('project')
            
            # Create extractor instance for this record
            extractor = qna_extractor(question=question, project=project)
            
            # Extract level 1 sentiment
            sentiment_1_json = extractor.extract_sentiment_level_one()
            if not sentiment_1_json:
                logger.warning(f"Failed to extract level 1 sentiment for record: {record.get('id', 'unknown')}")
                return None
            
            # Parse level 1 result
            try:
                cleaned_sentiment_1 = sentiment_1_json.strip().replace("```json", "").replace("```", "").strip()
                cleaned_sentiment_1 = cleaned_sentiment_1.lstrip('\n').rstrip('\n')
                sentiment_1_data = json.loads(cleaned_sentiment_1)
                sentiment_level_1 = sentiment_1_data.get('sentiment_level_1')
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing level 1 sentiment JSON for record {record.get('id', 'unknown')}: {e}")
                return None
            
            # Extract level 2 sentiment
            sentiment_level_2 = None
            if sentiment_level_1:
                sentiment_2_json = extractor.extract_sentiment_level_two(sentiment_level_1)
                if sentiment_2_json:
                    try:
                        cleaned_sentiment_2 = sentiment_2_json.strip().replace("```json", "").replace("```", "").strip()
                        cleaned_sentiment_2 = cleaned_sentiment_2.lstrip('\n').rstrip('\n')
                        sentiment_2_data = json.loads(cleaned_sentiment_2)
                        sentiment_level_2 = sentiment_2_data.get('sentiment_level_2')
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing level 2 sentiment JSON for record {record.get('id', 'unknown')}: {e}")
            
            # Update record with sentiment information
            updated_record = record.copy()
            updated_record['sentiment_level_1'] = sentiment_level_1
            updated_record['sentiment_level_2'] = sentiment_level_2
            updated_record['@search.action'] = 'mergeOrUpload'
            
            # Add processing timestamp
            now = datetime.datetime.utcnow()
            updated_record['sentiment_processed_at'] = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            
            logger.debug(f"Successfully processed sentiment for record {record.get('id', 'unknown')}: "
                        f"L1={sentiment_level_1}, L2={sentiment_level_2}")
            
            return updated_record
            
        except Exception as e:
            logger.error(f"Unexpected error processing record {record.get('id', 'unknown')}: {e}")
            return None

    def extract_batch_sentiments(self, dataframe, input_dict):
        """
        Extract sentiments for a batch of records from dataframe.
        Optimized for large-scale backfill operations.
        
        Args:
            dataframe: Pandas DataFrame containing records to process
            input_dict: Configuration dictionary
        
        Returns:
            DataFrame with sentiment information added
        """
        logger.info(f"Starting batch sentiment extraction for {len(dataframe):,} records")
        
        # Convert dataframe to records for processing
        records = dataframe.to_dict('records')
        
        # Process in smaller batches to manage memory and API limits
        batch_size = input_dict.get('batch_size', 100)
        all_processed_records = []
        
        total_batches = (len(records) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(records))
            batch_records = records[start_idx:end_idx]
            
            logger.info(f"Processing sentiment batch {batch_num + 1}/{total_batches} "
                       f"(records {start_idx + 1}-{end_idx})")
            
            # Process this batch
            processed_batch = self.process_sentiment_batch_records(batch_records, input_dict)
            all_processed_records.extend(processed_batch)
            
            # Add small delay between batches to avoid overwhelming the API
            if batch_num < total_batches - 1:
                time.sleep(1)
        
        logger.info(f"Completed batch sentiment extraction. "
                   f"Processed {len(all_processed_records):,} out of {len(records):,} records")
        
        # Convert back to DataFrame
        if all_processed_records:
            result_df = pd.DataFrame(all_processed_records)
            return result_df
        else:
            # Return original dataframe with empty sentiment columns if no records were processed
            result_df = dataframe.copy()
            result_df['sentiment_level_1'] = None
            result_df['sentiment_level_2'] = None
            return result_df