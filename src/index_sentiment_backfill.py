#!/usr/bin/env python3
"""
Sentiment Index Processor
Specialized index handler for sentiment backfill operations.
Optimized for fetching records needing sentiment analysis and updating them efficiently.
"""

import re
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from urllib.request import Request, urlopen
from azure.core.exceptions import HttpResponseError
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents.indexes import SearchIndexClient
from azure_search.index_handler import AzureSearchIndexUtility
from utils.logger_config import logger

class IndexProcessor:
    """
    Specialized index processor for sentiment backfill operations.
    Handles fetching records that need sentiment analysis and updating them with results.
    """
    
    def __init__(self, index_name=""):
        self.index_name = index_name
        self.azure_search = AzureSearchIndexUtility(index_name=self.index_name)
        logger.info(f"Initialized IndexProcessor for index: {index_name}")

    def fetch_records_for_sentiment(self, date_filter, project, limit=None):
        """
        Fetch records that need sentiment processing.
        Optimized to retrieve only necessary fields for sentiment extraction.
        
        Args:
            date_filter: OData filter string for date range and sentiment status
            project: Project name (MIRA or PCL)
            limit: Optional limit on number of records to fetch
        
        Returns:
            DataFrame containing records that need sentiment processing
        """
        df = pd.DataFrame()
        try:
            if project == "MIRA":
                # Fields needed for MIRA sentiment processing
                fields = [
                    "id", "Ucid", "question", "answer", "StartTime", 
                    "Is_Digital", "Is_Enrollment"
                ]
                logger.info(f"Fetching QNA records with fields: {fields}")
                documents = self.azure_search.search(
                    filter=date_filter, 
                    select=fields,
                    #top=limit
                )
                df = pd.DataFrame(documents)
                
            elif project == "PCL":
                # Fields needed for PCL sentiment processing ,"sentiment_level_1", "sentiment_level_2"
                fields = [
                    "id", "Ucid", "question", "answer", "StartTime"
                ]
                
                logger.info(f"Fetching PCL records with fields: {fields}")
                documents = self.azure_search.search(
                    filter=date_filter, 
                    select=fields,
                    #top=limit
                )
                df = pd.DataFrame(documents)
                
            else:
                raise ValueError("Unsupported project type. Supported projects are: MIRA, PCL.")
            
            logger.info(f"Successfully fetched {len(df):,} records for sentiment processing")
            
            # Log sentiment status distribution
            if not df.empty and 'sentiment_level_1' in df.columns:
                sentiment_stats = self._analyze_sentiment_status(df)
                logger.info(f"Sentiment status distribution: {sentiment_stats}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching records for sentiment processing: {e}")
            raise

    def fetch_records_missing_sentiment(self, project, days_back=7, batch_size=1000):
        """
        Fetch records that are missing sentiment data.
        Useful for targeted sentiment backfill operations.
        
        Args:
            project: Project name (MIRA or PCL)
            days_back: Number of days to look back for records
            batch_size: Maximum number of records to fetch
        
        Returns:
            DataFrame containing records missing sentiment data
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Build filter for records missing sentiment
        date_filter = f"StartTime ge {start_date_str} and StartTime lt {end_date_str}"
        sentiment_filter = "(sentiment_level_1 eq null or sentiment_level_1 eq '')"
        combined_filter = f"{date_filter} and {sentiment_filter}"
        
        logger.info(f"Fetching records missing sentiment from last {days_back} days")
        logger.info(f"Filter: {combined_filter}")
        
        return self.fetch_records_for_sentiment(combined_filter, project, limit=batch_size)

    def fetch_sample_records_for_testing(self, project, sample_size=10):
        """
        Fetch a small sample of records for testing sentiment extraction.
        
        Args:
            project: Project name (MIRA or PCL)
            sample_size: Number of records to fetch for testing
        
        Returns:
            DataFrame containing sample records
        """
        # Get recent records with questions
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        date_filter = f"StartTime ge {start_date_str} and StartTime lt {end_date_str}"
        question_filter = "question ne null and question ne ''"
        combined_filter = f"{date_filter} and {question_filter}"
        
        logger.info(f"Fetching {sample_size} sample records for testing")
        
        return self.fetch_records_for_sentiment(combined_filter, project, limit=sample_size)

    def update_index(self, result_data, key_field_name="id", semantic_content_field="question"):
        """
        Update the index with sentiment results.
        Optimized for batch updates with error handling.
        
        Args:
            result_data: List of documents to update or single document
            key_field_name: Name of the key field for updates
            semantic_content_field: Field to use for semantic search
        """
        try:
            if not result_data:
                logger.warning("No data provided for index update")
                return
            
            # Ensure result_data is a list
            if not isinstance(result_data, list):
                result_data = [result_data]
            
            logger.info(f"Updating index with {len(result_data):,} sentiment records")
            
            # Process in chunks to avoid overwhelming the index
            chunk_size = 5000
            total_chunks = (len(result_data) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(result_data), chunk_size):
                chunk = result_data[i:i + chunk_size]
                chunk_num = (i // chunk_size) + 1
                
                try:
                    self.azure_search.push_to_index(
                        chunk, 
                        key_field_name=key_field_name, 
                        semantic_content_field=semantic_content_field
                    )
                    logger.debug(f"Successfully updated chunk {chunk_num}/{total_chunks}")
                    
                    # Small delay between chunks to avoid rate limiting
                    if chunk_num < total_chunks:
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Error updating index chunk {chunk_num}: {e}")
                    # Continue with other chunks even if one fails
                    continue
            
            logger.info("Index update completed successfully")
            
        except Exception as e:
            logger.error(f"Error during index update: {e}")
            raise

    def update_sentiment_batch(self, sentiment_results):
        """
        Specialized method for updating sentiment results in batches.
        Includes validation and error recovery.
        
        Args:
            sentiment_results: List of records with sentiment data
        """
        if not sentiment_results:
            logger.warning("No sentiment results to update")
            return
        
        logger.info(f"Updating {len(sentiment_results):,} records with sentiment data")
        
        # Validate sentiment results
        validated_results = self._validate_sentiment_results(sentiment_results)
        
        if validated_results:
            self.update_index(validated_results, key_field_name="id")
            logger.info(f"Successfully updated {len(validated_results):,} records with sentiment data")
        else:
            logger.warning("No valid sentiment results to update")

    def get_sentiment_statistics(self, project, days_back=30):
        """
        Get statistics about sentiment data coverage in the index.
        
        Args:
            project: Project name (MIRA or PCL)
            days_back: Number of days to analyze
        
        Returns:
            Dictionary with sentiment statistics
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            date_filter = f"StartTime ge {start_date_str} and StartTime lt {end_date_str}"
            
            # Fetch records for analysis
            fields = ["id", "sentiment_level_1", "sentiment_level_2", "StartTime"]
            documents = self.azure_search.search(
                filter=date_filter,
                select=fields,
                top=10000  # Limit for analysis
            )
            
            df = pd.DataFrame(documents)
            
            if df.empty:
                return {"error": "No records found for analysis"}
            
            # Calculate statistics
            total_records = len(df)
            has_sentiment_l1 = df['sentiment_level_1'].notna().sum()
            has_sentiment_l2 = df['sentiment_level_2'].notna().sum()
            
            stats = {
                "total_records": total_records,
                "records_with_sentiment_l1": int(has_sentiment_l1),
                "records_with_sentiment_l2": int(has_sentiment_l2),
                "sentiment_l1_coverage": round((has_sentiment_l1 / total_records) * 100, 2),
                "sentiment_l2_coverage": round((has_sentiment_l2 / total_records) * 100, 2),
                "records_missing_sentiment": int(total_records - has_sentiment_l1),
                "analysis_period_days": days_back
            }
            
            # Sentiment distribution
            if has_sentiment_l1 > 0:
                sentiment_dist = df['sentiment_level_1'].value_counts().to_dict()
                stats["sentiment_distribution"] = sentiment_dist
            
            logger.info(f"Sentiment statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating sentiment statistics: {e}")
            return {"error": str(e)}

    def _analyze_sentiment_status(self, df):
        """Analyze the sentiment status of fetched records"""
        if 'sentiment_level_1' not in df.columns:
            return {"status": "sentiment columns not present"}
        
        total = len(df)
        has_l1 = df['sentiment_level_1'].notna().sum()
        has_l2 = df['sentiment_level_2'].notna().sum() if 'sentiment_level_2' in df.columns else 0
        missing = total - has_l1
        
        return {
            "total": total,
            "has_sentiment_l1": int(has_l1),
            "has_sentiment_l2": int(has_l2),
            "missing_sentiment": int(missing),
            "coverage_percent": round((has_l1 / total) * 100, 1) if total > 0 else 0
        }

    def _validate_sentiment_results(self, sentiment_results):
        """Validate sentiment results before updating index"""
        validated = []
        
        for result in sentiment_results:
            # Check required fields
            if 'id' not in result:
                logger.warning("Skipping result without id field")
                continue
            
            # Check sentiment data
            if 'sentiment_level_1' not in result and 'sentiment_level_2' not in result:
                logger.warning(f"Skipping result {result['id']} without sentiment data")
                continue
            
            # Validate sentiment values
            valid_l1_values = ["Positive", "Neutral", "Negative"]
            if 'sentiment_level_1' in result:
                if result['sentiment_level_1'] not in valid_l1_values and result['sentiment_level_1'] is not None:
                    logger.warning(f"Invalid sentiment_level_1 value for {result['id']}: {result['sentiment_level_1']}")
                    continue
            
            # Ensure @search.action is set
            if '@search.action' not in result:
                result['@search.action'] = 'mergeOrUpload'
            
            validated.append(result)
        
        logger.info(f"Validated {len(validated)} out of {len(sentiment_results)} sentiment results")
        return validated