# Standard Library Imports
import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
import re
import time

# Third-Party Imports
import pandas as pd
from dotenv import load_dotenv

# Local Application Imports
import config
try:
    from index import IndexProcessor
    from skillsets.vectorizer import AzureOpenAIVectorizer
except ImportError:
    IndexProcessor = None
    AzureOpenAIVectorizer = None

# configure the llm
from llm.llm_config import chat_completion
from utils.helper import load_project_config, write_to_csv, clean_topic
from utils.logger_config import logger

import httpx
import asyncio
import nest_asyncio
import aiohttp
from openai import AsyncAzureOpenAI
# nest_asyncio.apply()
import traceback

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

class AdaptiveRateLimiter:
    """Rate limiter that adapts to API responses"""
    
    def __init__(self, initial_qps=10, max_qps=80, backoff_factor=0.8):
        self.current_qps = initial_qps
        self.max_qps = max_qps
        self.backoff_factor = backoff_factor
        self.last_error_time = 0
        self.lock = asyncio.Lock()
        self.last_request_time = 0
        
    async def acquire(self):
        """Wait until we can make another request"""
        async with self.lock:
            now = time.time()
            
            # Gradually increase QPS if no recent errors
            if now - self.last_error_time > 30 and self.current_qps < self.max_qps:
                self.current_qps = min(self.current_qps * 1.05, self.max_qps)
            
            # Calculate wait time to maintain current QPS
            min_interval = 1.0 / self.current_qps
            elapsed = now - self.last_request_time
            wait_time = max(0, min_interval - elapsed)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                
            self.last_request_time = time.time()
            
    async def on_error(self, status_code):
        """Handle rate limit errors by reducing QPS"""
        if status_code == 429:
            async with self.lock:
                self.last_error_time = time.time()
                self.current_qps *= self.backoff_factor
                logger.info(f"Rate limited, reducing QPS to {self.current_qps:.2f}")

class qna_extractor:

    def __init__(self, text=None, question=None, topic=None, project=None):
        self.text = text
        self.qna = {}
        self.question = question
        self.topic = topic
        self.project = project  # Add index_dict to self
        self.http_session = None

    async def get_session(self):
        if self.http_session is None or hasattr(self.http_session, 'is_closed') and self.http_session.is_closed:
            # Create httpx client instead of aiohttp session
            self.http_session = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=50
                )
            )
        
        return self.http_session

    def clean_text(self):

        text = self.text

        # Remove unwanted characters and extra spaces
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'\*+', '****', text)  # Normalize redacted PII to a consistent format
        text = re.sub(r'[^\w\s,.!?\'-]', '', text)  # Remove any non-alphanumeric characters except punctuation

        # remove back ticks and new lines and quotes and curly braces
        text = text.replace("`", " ")
        text = text.replace("\n", " ")
        text = text.replace("'", " ")
        text = text.replace('"', " ")
        text = text.replace("{", " ")

        # Ensure proper formatting
        text = text.strip()  # Remove leading and trailing spaces
        
        return text

    def extract_qna(self):
        if self.text:
            extractor = qna_extractor(text=self.text)
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

            # Debugging
            logger.debug(f"QnA Extraction Prompt:\n{prompt}\n")

            self.qna = chat_completion(messages=message, max_tokens=2000, temperature=1e-9, task_type="qna_extraction")

        return self.qna
    
    def extract_topic(self):
        # Code that extracts QnA metadata from single text
        if self.question:
            question = self.question
            # Select the appropriate json based on the project
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
        # Code that extracts QnA metadata from single text
        sub_topic_prompt_dict = None
        sub_topic = None
        if self.question and self.topic:
            topic_cleaned = clean_topic(self.project, self.topic)
            sub_topic_prompt_dict = load_project_config(self.project, "new_subtopic_extraction")
            
            sub_topic_prompt = sub_topic_prompt_dict[topic_cleaned]
            sub_topic_prompt = prompt_sub_topic_format(self.project, sub_topic_prompt)
            
            prompt = sub_topic_extraction_prompt(question=self.question, topic=self.topic, sub_topic_descriptions = sub_topic_prompt)

            cleaned_lines = [line.lstrip() for line in prompt.splitlines()]
            prompt = "\n".join(cleaned_lines)
            
            message = [{"role": "user", "content": prompt}]
    
            sub_topic = chat_completion(messages=message, max_tokens=2000, temperature=1e-9, task_type="sub_topic_extraction")
            
        return sub_topic

    def extract_is_useful(self):
        is_useful = None
        if self.question:

            question_json_data = load_project_config(self.project, "useful_questions")
            prompt = is_useful_question_extraction_prompt(self.question, question_json_data)

            message = [{"role": "user", "content": prompt}]

            is_useful = chat_completion(messages=message, max_tokens=2000, temperature=1e-9, task_type="is_useful_extraction")

        return is_useful
    
    def extract_sentiment_level_one(self):
        """
        Extract high-level sentiment (Positive, Neutral, Negative) for a question.
        Returns: JSON string with sentiment_level_1
        """
        sentiment_level_1 = None
        if self.question:
            sentiment_config =  load_project_config(self.project, "sentiment_extraction")
            level_1_categories = sentiment_config.get("level_1", {})

            prompt = sentiment_level_1_prompt(self.question, level_1_categories)
            
            message = [{"role": "user", "content": prompt}]
            
            sentiment_level_1 = chat_completion(
                messages=message, 
                max_tokens=500, 
                temperature=1e-9, 
                task_type="sentiment_level_1_extraction"
            )
            
        return sentiment_level_1

    def extract_sentiment_level_two(self, sentiment_level_1):
        """
        Extract detailed sentiment subcategory based on level 1 sentiment.
        Args:
            sentiment_level_1: The high-level sentiment (Positive, Neutral, Negative)
        Returns: JSON string with sentiment_level_2
        """
        sentiment_level_2 = None
        if self.question and sentiment_level_1:
            sentiment_config =  load_project_config(self.project, "sentiment_extraction")
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
            
        return sentiment_level_2
    
    def process_index_row(self, index, row, input_dict):

        max_retries = 5
        text = row['Text']
        ucid = row['Ucid']
        project = input_dict["project"]

        logger.info(f"Processing row {index} with UCID: {ucid}")
                
        for attempt in range(max_retries):
            try:
                # Extract the necessary fields
                ucid = row['Ucid']
                start_time = row['StartTime']
                is_digital = None
                is_enrollment = None
                
                if project == "MIRA":
                    is_digital = row['Is_Digital']
                    is_enrollment = row['Is_Enrollment']
                    plan_name = row['plan_name']
                    drugs = row['drugs']
                    providers = row['providers']
                    zip = row['zip']
                    county_processed = row['county_processed']
                    state_processed = row['state_processed']
                    region_processed = row['region_processed']
                    subregion_processed = row['subregion_processed']

                # Perform qna extraction
                extractor_qna = qna_extractor(text=text, project=project)
                qna_data_json = extractor_qna.extract_qna()
                
                expanded_qna = []

                if qna_data_json:
                    # Clean the response before parsing
                    raw_response = qna_data_json
                    cleaned_response = raw_response.strip()
                    cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
                    # Optionally remove leading/trailing newlines
                    cleaned_response = cleaned_response.lstrip('\n').rstrip('\n')
                    
                    try:
                        qna_data = json.loads(cleaned_response)
                    except Exception as json_err:
                        logger.error(f"JSON parsing error for UCID {ucid}: {json_err}")
                        logger.error(f"Raw response:\n{raw_response}")
                        raise

                    for question_dict in qna_data['question_and_answer']:
                        question_1 = question_dict['question']
                        answer_1 = question_dict['answer']
                        
                        vectorizer = AzureOpenAIVectorizer()
                        vector_question = vectorizer.vectorize_text(question_1)
                        id_hash = hashlib.sha256(ucid.encode() + question_1.encode()).hexdigest()
                        
                        # Instantiate doc using question and topic.
                        if project == "MIRA":
                            expanded_qna_1 = {
                                "id": id_hash,
                                "Ucid": ucid,
                                "Is_Digital": is_digital,
                                "Is_Enrollment": is_enrollment,
                                "StartTime": start_time,
                                "question": question_1,
                                "answer": answer_1,
                                "vector": vector_question,
                                "plan_name": plan_name,
                                "drugs": drugs,
                                "providers": providers,
                                "zip": zip,
                                "county_processed": county_processed,
                                "state_processed": state_processed,
                                "region_processed": region_processed,
                                "subregion_processed": subregion_processed,
                                "@search.action": "mergeOrUpload"
                            }
                        elif project == "PCL":
                            expanded_qna_1 = {
                                "id": id_hash,
                                "Ucid": ucid,
                                "StartTime": start_time,
                                "question": question_1,
                                "answer": answer_1,
                                "sales_market": row.get("sales_market", ""),
                                "business_market": row.get("business_market", ""),
                                "region": row.get("region", ""),
                                "subregion": row.get("subregion",""),
                                "state": row.get("state",""),
                                "vector": vector_question,
                                "@search.action": "mergeOrUpload"
                            }
                        else:
                            logger.error(f"Unexpected project type: {project}")
                            raise ValueError("Unsupported project type. Supported projects are: MIRA, PCL.")
                        
                        # Perform topic extraction and add topic to doc if available
                        extractor = qna_extractor(question=question_1, project=project)
                        qna_topic_data_json = extractor.extract_topic()

                        if qna_topic_data_json:
                            try:
                                qna_topic_data = json.loads(qna_topic_data_json)

                                expanded_qna_1["topic"] = qna_topic_data['topic']
                                
                            except Exception as e:
                                logger.error(f"Error processing topic for UCID {ucid}: {e}")

                        # Perform subtopic extraction and add subtopic to doc if available
                        extractor = qna_extractor(question=question_1, topic=expanded_qna_1["topic"], project=project)
                        qna_subtopic_data_json = extractor.extract_subtopic()
                        if qna_subtopic_data_json:
                            try:
                                cleaned_json = qna_subtopic_data_json.strip().replace("```json", "").replace("```", "").strip()
                                qna_subtopic_data = json.loads(cleaned_json)

                                if isinstance(qna_subtopic_data.get('sub_topic', []), str):
                                    qna_subtopic_data['sub_topic'] = [qna_subtopic_data['sub_topic']]

                                predefined_subtopics = qna_subtopic_data.get('sub_topic', [])

                                if project == "MIRA":
                                    expanded_qna_1["sub_topic"] = predefined_subtopics
                                    expanded_qna_1["grouped_sub_topic"] = predefined_subtopics
                                elif project == "PCL":
                                    expanded_qna_1["sub_topic"] = predefined_subtopics
                                    expanded_qna_1["grouped_sub_topic"] = predefined_subtopics
                                else:
                                    logger.error(f"Unexpected project type: {project}")
                                    raise ValueError("Unsupported project type. Supported projects are: MIRA, PCL.")

                            except Exception as e:
                                logger.error(f"Error processing subtopics for UCID {ucid}: {e}")

                        # Perform is_useful extraction and add is_useful to doc if available
                        qna_is_useful = extractor.extract_is_useful()
                        if qna_is_useful:
                            try:
                                if isinstance(qna_is_useful, str) and qna_is_useful.lower() in ["true", "false"]:
                                    expanded_qna_1["is_useful"] = qna_is_useful.lower() == "true"
                            except Exception as e:
                                logger.error(f"Error processing is_useful for UCID {ucid}: {e}")

                        # Append the processed Q&A to the list.
                        expanded_qna.append(expanded_qna_1)

                    now = datetime.datetime.utcnow()
                    formatted_time = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

                    expanded_qna_processed = {
                            "Ucid": ucid,
                            "qa_processed_time": formatted_time,
                            "project": project,
                            "@search.action": "mergeOrUpload"
                        }
                    return expanded_qna, expanded_qna_processed
                
                # Upon the final retry, log the error and move on.
                if attempt == max_retries - 1:
                    logger.error("No more attempts")
                    with open('error.txt', 'a') as f:
                        f.write(f"Row UCID: {row['Ucid']}, LLM response type mismatch: {qna_data_json}\n")
                    logger.error(f"Moving on to the next UCID after {max_retries} attempts.")
                    break

            except Exception as e:
                logger.error(f"Error processing for Ucid {row['Ucid']} on attempt {attempt+1}: {e}")

                # If a rate-limit error is encountered, pause briefly.
                if "RateLimitError" in str(e):
                    logger.info("Rate limit exceeded. Waiting for 1 min before retrying.")
                    time.sleep(60)

                # Upon the final retry, log the error and move on.
                if attempt == max_retries - 1:
                    with open('error.txt', 'a') as f:
                        f.write(f"Row UCID: {row['Ucid']}, Error: {e}\n")
                    logger.error(f"Moving on to the next UCID after {max_retries} attempts.")
                    break

    def process_file_conversation_row(self, index, row, input_dict):

        max_retries = 5
        text_1 = row['Text']
        project_1 = input_dict["project"]
        ucid = row.get('Ucid', None)

        logger.info(f"Processing row {index} with UCID: {ucid}")
    
        for attempt in range(max_retries):
            try:
                # Perform qna extraction.
                extractor_qna = qna_extractor(text=text_1, project=project_1)
                qna_data_json = extractor_qna.extract_qna()
                
                if qna_data_json:
                    # Clean the response before parsing
                    raw_response = qna_data_json
                    cleaned_response = raw_response.strip()
                    cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
                    # Optionally remove leading/trailing newlines
                    cleaned_response = cleaned_response.lstrip('\n').rstrip('\n')
                    
                    try:
                        qna_data = json.loads(cleaned_response)
                    except Exception as json_err:
                        logger.error(f"JSON parsing error for Row {index + 1}: {json_err}")
                        logger.error(f"Raw response:\n{raw_response}")
                        raise

                    df = pd.DataFrame(columns=['question'])
                    expanded_qna = []

                    for question_dict in qna_data['question_and_answer']:

                        question_1 = question_dict['question']
                        
                        # Instantiate doc using question and topic.
                        if project_1 == "MIRA":
                            expanded_qna_1 = {
                                "question": question_1,
                            }
                        elif project_1 == "PCL":
                            expanded_qna_1 = {
                                "question": question_1,
                            }
                        else:
                            logger.error(f"Unexpected project type: {project_1}")
                            raise ValueError("Unsupported project type. Supported projects are: MIRA, PCL.")

                        # Append the processed Q&A to the list.
                        expanded_qna.append(expanded_qna_1)

                    df_1 = pd.DataFrame(expanded_qna)
                    df = pd.concat([df, df_1], ignore_index=True)

                    logger.info(f"number of questions extracted: {len(df)}")

                    df_expanded_qna = pd.DataFrame(columns=['Ucid', 'Text', 'question', 'topic', 'sub_topic', 'is_useful'])

                    for index_q, row_q in df.iterrows():

                        df_expanded_qna_1 = self.process_file_question_row(index_q, row_q, input_dict)

                        df_expanded_qna_1['Text'] = text_1
                        df_expanded_qna_1['Ucid'] = ucid
                        df_expanded_qna_1 = df_expanded_qna_1[['Ucid', 'Text', 'question', 'topic', 'sub_topic', 'is_useful']]

                        df_expanded_qna = pd.concat([df_expanded_qna, df_expanded_qna_1], ignore_index=True)

                    return df_expanded_qna
                
                # Upon the final retry, log the error and move on.
                if attempt == max_retries - 1:
                    logger.error("No more attempts")
                    with open('error.txt', 'a') as f:
                        f.write(f"Row {index + 1}, LLM response type mismatch: {qna_data_json}\n")
                    logger.error(f"Moving on to the next Row after {max_retries} attempts.")
                    break

            except Exception as e:
                logger.error(f"Error processing for Row {index + 1} on attempt {attempt+1}: {e}")

                # If a rate-limit error is encountered, pause briefly.
                if "RateLimitError" in str(e):
                    logger.info("Rate limit exceeded. Waiting for 1 min before retrying.")
                    time.sleep(60)

                # Upon the final retry, log the error and move on.
                if attempt == max_retries - 1:
                    with open('error.txt', 'a') as f:
                        f.write(f"Row {index + 1}, Error: {e}\n")
                    logger.error(f"Moving on to the next Row after {max_retries} attempts.")
                    break

    def process_file_question_row(self, index, row, input_dict):

        max_retries = 5
        project_1 = input_dict["project"]
        expanded_qna_1 = {}
    
        for attempt in range(max_retries):
            try:
                question_1 = row['question']
                
                extractor = qna_extractor(question=question_1, project=project_1)
                qna_topic_data_json = extractor.extract_topic()

                if qna_topic_data_json:
                    try:
                        qna_topic_data = json.loads(qna_topic_data_json)

                        if isinstance(qna_topic_data['topic'], str):
                            qna_topic_data['topic'] = qna_topic_data['topic']

                        if isinstance(qna_topic_data['topic'], list):
                            qna_topic_data['topic'] = qna_topic_data['topic'][0]

                        expanded_qna_1["question"] = question_1
                        expanded_qna_1["topic"] = qna_topic_data['topic']

                    except Exception as e:
                        logger.error(f"Error processing topic for Row {index + 1}: {e}")

                    # If topic_filter is set, skip subtopic extraction for non-matching topics
                    topic_filter = input_dict.get("topic_filter", "")
                    if topic_filter and expanded_qna_1.get("topic", "").lower() != topic_filter:
                        logger.info(f"Skipping Row {index + 1}: topic '{expanded_qna_1.get('topic')}' does not match filter '{topic_filter}'")
                        expanded_qna_1["sub_topic"] = []
                        expanded_qna_1["is_useful"] = False
                        df_expanded_qna_1 = pd.DataFrame([expanded_qna_1])
                        df_expanded_qna_1['Text'] = "NA"
                        if 'Ucid' in row:
                            df_expanded_qna_1['Ucid'] = row['Ucid']
                            df_expanded_qna_1 = df_expanded_qna_1[['Ucid', 'Text', 'question', 'topic', 'sub_topic', 'is_useful']]
                        else:
                            df_expanded_qna_1 = df_expanded_qna_1[['Text', 'question', 'topic', 'sub_topic', 'is_useful']]
                        return df_expanded_qna_1

                    # Perform subtopic extraction and add subtopic to doc if available.
                    extractor = qna_extractor(question=question_1, topic=expanded_qna_1["topic"], project=project_1)
                    qna_subtopic_data_json = extractor.extract_subtopic()
                    if qna_subtopic_data_json:
                        try:
                            qna_subtopic_data = json.loads(qna_subtopic_data_json)

                            if isinstance(qna_subtopic_data['sub_topic'], str):
                                qna_subtopic_data['sub_topic'] = [qna_subtopic_data['sub_topic']]

                            predefined_subtopics = qna_subtopic_data['sub_topic']

                            expanded_qna_1["sub_topic"] = predefined_subtopics
                        except Exception as e:
                            logger.error(f"Error processing subtopic for Row {index + 1}: {e}")

                    # Perform is_useful extraction and add is_useful to doc if available.
                    qna_is_useful = extractor.extract_is_useful()
                    if qna_is_useful:
                        try:
                            if isinstance(qna_is_useful, str) and qna_is_useful.lower() in ["true", "false"]:
                                expanded_qna_1["is_useful"] = qna_is_useful.lower() == "true"
                        except Exception as e:
                            logger.error(f"Error processing is_useful for Row {index + 1}: {e}")

                    df_expanded_qna_1 = pd.DataFrame([expanded_qna_1])
                    df_expanded_qna_1['Text'] = "NA"
                    # Propagate Ucid if present in row
                    if 'Ucid' in row:
                        df_expanded_qna_1['Ucid'] = row['Ucid']
                        df_expanded_qna_1 = df_expanded_qna_1[['Ucid', 'Text', 'question', 'topic', 'sub_topic', 'is_useful']]
                    else:
                        df_expanded_qna_1 = df_expanded_qna_1[['Text', 'question', 'topic', 'sub_topic', 'is_useful']]

                    return df_expanded_qna_1
                
                # Upon the final retry, log the error and move on.
                if attempt == max_retries - 1:
                    logger.error("No more attempts")
                    with open('error.txt', 'a') as f:
                        f.write(f"Row {index + 1}, LLM response type mismatch: {qna_topic_data_json}\n")
                    logger.error(f"Moving on to the next Row after {max_retries} attempts.")
                    break

            except Exception as e:
                logger.error(f"Error processing for Row {index + 1} on attempt {attempt+1}: {e}")

                # If a rate-limit error is encountered, pause briefly.
                if "RateLimitError" in str(e):
                    logger.info("Rate limit exceeded. Waiting for 1 min before retrying.")
                    time.sleep(60)

                # Upon the final retry, log the error and move on.
                if attempt == max_retries - 1:
                    with open('error.txt', 'a') as f:
                        f.write(f"Row {index + 1}, Error: {e}\n")
                    logger.error(f"Moving on to the next Row after {max_retries} attempts.")
                    break

    async def extract_batch_async(self, df, input_dict, max_concurrent=60):
        """Async version of extract_batch using asyncio for concurrent processing"""
        
        processed_rows = 0
        try:
            all_expanded_qna = []
            all_expanded_qna_processed = []
            total_rows = len(df)
            
            rate_limiter = AdaptiveRateLimiter(initial_qps=60, max_qps=120)
            
            # Create a semaphore to limit concurrent tasks
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = []
            if input_dict['dry_run']:
                time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                all_expanded_qna_processed_batch_filename = f"data/output/{input_dict['source_index']}_{time_stamp}.json"
                all_expanded_qna_batch_filename = f"data/output/{input_dict['destination_index']}_{time_stamp}.json"
            # Create tasks for all rows
            for index, row in df.iterrows():
                text_str = row.get("Text", "")
                if text_str is None or text_str.strip() == "":
                    logger.info(f"Skipping row {index} due to empty text.")
                    continue
                
                task = self.process_index_row_async(
                    semaphore,
                    rate_limiter,
                    index,
                    row,
                    input_dict
                )
                tasks.append(task)
            
            # Process tasks in batches and collect results
            for completed_tasks in asyncio.as_completed(tasks):
                try:
                    result = await completed_tasks
                    if result:
                        expanded_qna, expanded_qna_processed = result
                        all_expanded_qna.extend(expanded_qna)
                        all_expanded_qna_processed.append(expanded_qna_processed)
                        
                        processed_rows += 1
                        percentage_processed = (processed_rows / total_rows) * 100
                        logger.info(f"Processed {processed_rows}/{total_rows} rows ({percentage_processed:.2f}%)")
                        
                        # Upload in batches of 100
                        if len(all_expanded_qna) >= 100:
                            batch = all_expanded_qna[:100]
                            filtered_batch = [doc for doc in batch if doc is not None]
                            if input_dict['dry_run']:
                                logger.info(f"Dry run enabled - skipping upload of batch to {input_dict['destination_index']}")
                                # Save filtered batch to file for review
                                with open(all_expanded_qna_batch_filename, 'w') as f:
                                    json.dump(filtered_batch, f, indent=2)
                                logger.info(f"Dry run: saved batch of {len(filtered_batch)} documents to {all_expanded_qna_batch_filename}")
                            else:
                                await self.upload_batch_async(
                                    filtered_batch, 
                                    input_dict["destination_index"], 
                                    "id", 
                                    "question"
                                )
                            all_expanded_qna = all_expanded_qna[100:]
                        
                        if len(all_expanded_qna_processed) >= 100:
                            batch = all_expanded_qna_processed[:100]
                            filtered_batch = [doc for doc in batch if doc is not None]
                            if input_dict['dry_run']:
                                logger.info(f"Dry run enabled - skipping upload of batch to {input_dict['source_index']}")
                                # Save filtered batch to file for review
                                with open(all_expanded_qna_processed_batch_filename, 'w') as f:
                                    json.dump(filtered_batch, f, indent=2)
                                logger.info(f"Dry run: saved batch of {len(filtered_batch)} documents to {all_expanded_qna_processed_batch_filename}")
                            else:
                                await self.upload_batch_async(
                                    filtered_batch, 
                                    input_dict["source_index"], 
                                    "Ucid"
                                )
                            all_expanded_qna_processed = all_expanded_qna_processed[100:]
                
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
            
            # Upload any remaining items
            if all_expanded_qna:
                if input_dict['dry_run']:
                    logger.info(f"Dry run enabled - skipping upload of batch to {input_dict['destination_index']}")
                    # Save filtered batch to file for review
                    with open(all_expanded_qna_batch_filename, 'a') as f:
                        json.dump(filtered_batch, f, indent=2)
                    logger.info(f"Dry run: saved batch of {len(filtered_batch)} documents to {all_expanded_qna_batch_filename}")
                else:
                    await self.upload_batch_async(
                        all_expanded_qna, 
                        input_dict["destination_index"], 
                        "id", 
                        "question"
                    )
            
            if all_expanded_qna_processed:
                if input_dict['dry_run']:
                    logger.info(f"Dry run enabled - skipping upload of batch to {input_dict['source_index']}")
                    # Save filtered batch to file for review
                    with open(all_expanded_qna_processed_batch_filename, 'a') as f:
                        json.dump(all_expanded_qna_processed, f, indent=2)
                    logger.info(f"Dry run: saved batch of {len(all_expanded_qna_processed)} documents to {all_expanded_qna_processed_batch_filename}")
                else:
                    await self.upload_batch_async(
                        all_expanded_qna_processed, 
                        input_dict["source_index"], 
                        "Ucid"
                    )
            
        finally:
            # Close the session when done
            if hasattr(self, 'http_session') and self.http_session:
                await self.http_session.aclose()
        
        return processed_rows

    async def process_index_row_async(self, semaphore, rate_limiter, index, row, input_dict):
        """Async version of process_index_row"""

        async with semaphore:  # Limit concurrent processing
            max_retries = 5
            text = row['Text']
            ucid = row['Ucid']
            project = input_dict["project"]

            if text is None or text.strip() == "":
                logger.info(f"Skipping row {index} with UCID: {ucid} due to empty text.")
                return None
            
            logger.info(f"Processing row {index} with UCID: {ucid}")
            
            for attempt in range(max_retries):
                try:
                    # Wait for rate limiter
                    await rate_limiter.acquire()
                    
                    # Extract QnA data
                    qna_data_json = await self.extract_qna_async(text, project)
                    
                    # Continue with processing as in the original function
                    expanded_qna = []
                    
                    if qna_data_json:
                        # Process QnA results
                        cleaned_response = qna_data_json.strip().replace("```json", "").replace("```", "").strip()
                        cleaned_response = cleaned_response.lstrip('\n').rstrip('\n')
                        
                        try:
                            qna_data = json.loads(cleaned_response)
                        except Exception as json_err:
                            logger.error(f"JSON parsing error for UCID {ucid}: {json_err}")
                            raise
                        
                        # Process each question in parallel
                        question_tasks = []
                        for question_dict in qna_data['question_and_answer']:
                            question_1 = question_dict['question']
                            answer_1 = question_dict['answer']
                            
                            # Create the basic QnA entry
                            id_hash = hashlib.sha256(ucid.encode() + question_1.encode()).hexdigest()
                            
                            if project == "MIRA":
                                expanded_qna_1 = {
                                    "id": id_hash,
                                    "Ucid": ucid,
                                    "Is_Digital": row.get('Is_Digital'),
                                    "Is_Enrollment": row.get('Is_Enrollment'),
                                    "StartTime": row.get('StartTime'),
                                    "question": question_1,
                                    "answer": answer_1,
                                    "plan_name": row.get('plan_name'),
                                    "drugs": row.get('drugs'),
                                    "providers": row.get('providers'),
                                    "zip": row.get('zip'),
                                    "county_processed": row.get('county_processed'),
                                    "state_processed": row.get('state_processed'),
                                    "region_processed": row.get('region_processed'),
                                    "subregion_processed": row.get('subregion_processed'),
                                    "@search.action": "mergeOrUpload"
                                }
                            elif project == "PCL":
                                expanded_qna_1 = {
                                    "id": id_hash,
                                    "Ucid": ucid,
                                    "StartTime": row.get('StartTime'),
                                    "question": question_1,
                                    "answer": answer_1,
                                    "sales_market": row.get("sales_market", ""),
                                    "business_market": row.get("business_market", ""),
                                    "region": row.get("region", ""),
                                    "subregion": row.get("subregion",""),
                                    "state": row.get("state",""),
                                    "@search.action": "mergeOrUpload"
                                }
                            else:
                                logger.error(f"Unexpected project type: {project}")
                                raise ValueError("Unsupported project type")
                            
                            # Process topic, subtopic, and is_useful in parallel
                            task = self.process_question_metadata_async(
                                question_1, 
                                project, 
                                expanded_qna_1, 
                                rate_limiter
                            )
                            question_tasks.append(task)
                        
                        # Wait for all question processing to complete
                        processed_questions = await asyncio.gather(*question_tasks, return_exceptions=True)
                        
                        # Filter out exceptions and add successful results to expanded_qna
                        for result in processed_questions:
                            if not isinstance(result, Exception):
                                expanded_qna.append(result)
                        
                        # Create processed record
                        now = datetime.datetime.utcnow()
                        formatted_time = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                        
                        expanded_qna_processed = {
                            "Ucid": ucid,
                            "qa_processed_time": formatted_time,
                            "project": project,
                            "@search.action": "mergeOrUpload"
                        }
                        
                        return expanded_qna, expanded_qna_processed
                    
                    if attempt == max_retries - 1:
                        logger.error("No more attempts")
                        with open('error.txt', 'a') as f:
                            f.write(f"Row UCID: {row['Ucid']}, LLM response type mismatch: {qna_data_json}\n")
                        break
                
                except Exception as e:
                    logger.error(f"Error processing for Ucid {row['Ucid']} on attempt {attempt+1}: {e}")
                    traceback.print_exc()
                    
                    # If rate limit error, adjust the limiter
                    if "RateLimitError" in str(e) or "429" in str(e):
                        await rate_limiter.on_error(429)
                        logger.info("Rate limit exceeded. Waiting before retrying.")
                        await asyncio.sleep(min(2 ** attempt, 60))  # Exponential backoff
                    
                    if attempt == max_retries - 1:
                        with open('error.txt', 'a') as f:
                            f.write(f"Row UCID: {row['Ucid']}, Error: {e}\n")
                        break

    async def process_question_metadata_async(self, question, project, expanded_qna_entry, rate_limiter):
        """Process topic, subtopic, is_useful, and sentiment in parallel for a question"""
        
        # Wait for rate limiter before starting
        await rate_limiter.acquire()
        
        # Start topic extraction
        topic_task = self.extract_topic_async(question, project, rate_limiter)
        
        # We can run is_useful and sentiment_level_1 in parallel with topic
        is_useful_task = self.extract_is_useful_async(question, project, rate_limiter)
        
        # Wait for topic result
        topic_result = await topic_task
        
        # Parse topic result
        if topic_result:
            try:
                cleaned_topic = topic_result.strip().replace("```json", "").replace("```", "").strip()
                cleaned_topic = cleaned_topic.lstrip('\n').rstrip('\n')
                
                topic_data = json.loads(cleaned_topic)

                expanded_qna_entry["topic"] = topic_data['topic']
                
                # Now we can start subtopic which depends on topic
                subtopic_task = self.extract_subtopic_async(
                    question, 
                    expanded_qna_entry["topic"], 
                    project, 
                    rate_limiter
                )
                
                # Wait for remaining tasks
                is_useful_result = await is_useful_task
                subtopic_result = await subtopic_task
                
                # Process subtopic result
                if subtopic_result:
                    logger.info(f"Raw subtopic result for question '{question}': {subtopic_result}")
                    try:
                        cleaned_subtopic = subtopic_result.strip().replace("```json", "").replace("```", "").strip()
                        cleaned_subtopic = cleaned_subtopic.lstrip('\n').rstrip('\n')

                        subtopic_data = json.loads(cleaned_subtopic)

                        if isinstance(subtopic_data.get('sub_topic', []), str):
                            subtopic_data['sub_topic'] = [subtopic_data['sub_topic']]

                        predefined_subtopics = subtopic_data.get('sub_topic', [])

                        expanded_qna_entry["sub_topic"] = predefined_subtopics
                        expanded_qna_entry["grouped_sub_topic"] = predefined_subtopics
                    except Exception as e:
                        logger.error(f"Error processing subtopic: {e}")
                
                # Process is_useful result
                if is_useful_result:
                    try:
                        if isinstance(is_useful_result, str) and is_useful_result.lower() in ["true", "false"]:
                            expanded_qna_entry["is_useful"] = is_useful_result.lower() == "true"
                    except Exception as e:
                        logger.error(f"Error processing is_useful: {e}")
                     
                # Vectorize in the async function
                expanded_qna_entry["vector"] = await self.vectorize_text_async(question)
                
                return expanded_qna_entry
                
            except Exception as e:
                logger.error(f"Error processing topic: {e} {topic_result, question}")
                raise
        
        return None
    
    # Convert the vectorizer to async to avoid blocking
    async def vectorize_text_async(self, text):
        """Async version of vectorize_text"""
        # Run the vectorizer in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        vectorizer = AzureOpenAIVectorizer()
        return await loop.run_in_executor(None, vectorizer.vectorize_text, text)

    
    async def extract_subtopic_async(self, question, topic, project, rate_limiter=None):
        """Async version of extract_subtopic"""
        if question and topic:
            # Acquire rate limiter if provided
            if rate_limiter:
                await rate_limiter.acquire()
                
            topic_cleaned = clean_topic(project, topic)
            sub_topic_prompt_dict = load_project_config(project, "new_subtopic_extraction")
            
            sub_topic_prompt = sub_topic_prompt_dict[topic_cleaned]
            sub_topic_prompt = prompt_sub_topic_format(project, sub_topic_prompt)
            
            prompt = sub_topic_extraction_prompt(question=question, topic=topic, sub_topic_descriptions=sub_topic_prompt)
            
            cleaned_lines = [line.lstrip() for line in prompt.splitlines()]
            prompt = "\n".join(cleaned_lines)
            
            message = [{"role": "user", "content": prompt}]
            
            return await self.chat_completion_async(messages=message, max_tokens=2000, temperature=1e-9)
        return None

    async def extract_is_useful_async(self, question, project, rate_limiter=None):
        """Async version of extract_is_useful"""
        if question:
            # Acquire rate limiter if provided
            if rate_limiter:
                await rate_limiter.acquire()
                
            question_json_data = load_project_config(project, "useful_questions")
            prompt = is_useful_question_extraction_prompt(question, question_json_data)
            
            message = [{"role": "user", "content": prompt}]
            
            return await self.chat_completion_async(messages=message, max_tokens=2000, temperature=1e-9)
        return None


    async def extract_qna_async(self, text, project):
        """Async version of extract_qna"""
        if text:
            extractor = qna_extractor(text)
            chunk = extractor.clean_text()
            prompt = qna_extraction_prompt(chunk)
            message = [{"role": "user", "content": prompt}]
            
            return await self.chat_completion_async(messages=message, max_tokens=2000, temperature=1e-9)

    async def extract_topic_async(self, question, project, rate_limiter=None):
        """Async version of extract_topic"""
        if question:
            # Acquire rate limiter if provided
            if rate_limiter:
                await rate_limiter.acquire()
                
            topic_descriptions_json = load_project_config(project, "topic_extraction")
            topic_descriptions = prompt_topic_format(project, topic_descriptions_json)
            prompt = topic_extraction_prompt(question, topic_descriptions)
            
            cleaned_lines = [line.lstrip() for line in prompt.splitlines()]
            prompt = "\n".join(cleaned_lines)
            
            message = [{"role": "user", "content": prompt}]
            
            return await self.chat_completion_async(messages=message, max_tokens=2000, temperature=1e-9)

    async def extract_sentiment_level_one_async(self, question, project, rate_limiter=None):
        """Async version of extract_sentiment_level_one"""
        if question:
            # Acquire rate limiter if provided
            if rate_limiter:
                await rate_limiter.acquire()
                
            sentiment_config = load_project_config(project, "sentiment_extraction")
            level_1_categories = sentiment_config.get("level_1", {})

            prompt = sentiment_level_1_prompt(question, level_1_categories)
            
            message = [{"role": "user", "content": prompt}]
            
            return await self.chat_completion_async(
                messages=message, 
                max_tokens=500, 
                temperature=1e-9,
                task_type="sentiment_level_1_extraction"
            )
        return None

    async def extract_sentiment_level_two_async(self, question, sentiment_level_1, project, rate_limiter=None):
        """Async version of extract_sentiment_level_two"""
        if question and sentiment_level_1:
            # Acquire rate limiter if provided
            if rate_limiter:
                await rate_limiter.acquire()
                
            sentiment_config = load_project_config(project, "sentiment_extraction")
            level_2_categories = sentiment_config.get("level_2", {}).get(sentiment_level_1.lower(), {})

            prompt = sentiment_level_2_prompt(
                question, 
                sentiment_level_1, 
                level_2_categories
            )
            
            message = [{"role": "user", "content": prompt}]
            
            return await self.chat_completion_async(
                messages=message, 
                max_tokens=500, 
                temperature=1e-9,
                task_type="sentiment_level_2_extraction"
            )
        return None

    async def chat_completion_async(self, messages, max_tokens=2000, temperature=1e-9, task_type="default"):
        """Async version of chat_completion"""
        try:
            session = await self.get_session()
     
            client = AsyncAzureOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_version=config.AZURE_OPENAI_API_VERSION,
                api_key=config.AZURE_OPENAI_KEY,
                http_client=session
            )
            
            response = await client.chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in async chat completion for {task_type}: {e}")
            raise


    async def upload_batch_async(self, batch, index_name, key_field_name, semantic_content_field=None):
        """
        Async version of batch upload to Azure Cognitive Search index
        
        Args:
            batch: List of documents to upload
            index_name: Name of the destination index
            key_field_name: Primary key field name in the index
            semantic_content_field: Optional field name for semantic indexing
        """
        try:
            index_processor = IndexProcessor(index_name=index_name)
            
            # Create a wrapper around the synchronous update_index method
            # Use a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            if semantic_content_field:
                await loop.run_in_executor(
                    None, 
                    lambda: index_processor.update_index(batch, key_field_name, semantic_content_field)
                )
            else:
                await loop.run_in_executor(
                    None,
                    lambda: index_processor.update_index(batch, key_field_name)
                )
            
            logger.info(f"Async uploaded batch of {len(batch)} documents to index {index_name}")
            
        except Exception as e:
            logger.error(f"Error in async upload to index {index_name}: {e}")
            # Save failed batch for later retry
            failed_file = f"failed_upload_{index_name}_{time.time()}.json"
            with open(failed_file, 'w') as f:
                json.dump(batch, f)
            logger.error(f"Failed batch saved to {failed_file}")
            raise


    def extract_batch(self, df, input_dict, max_workers=50):

        if input_dict['input_type'] == "index":

            return asyncio.run(self.extract_batch_async(df, input_dict, max_workers))

        elif input_dict['input_type'] == "file":

            all_expanded_qna = []
            total_rows = len(df)
            processed_rows = 0

            # Map variable values to functions  
            file_input_function_map = {  
                'conversations': self.process_file_conversation_row,  
                'questions': self.process_file_question_row
            }

            file_input_column_map = {  
                'conversations': 'Text',
                'questions': 'question'
            }

            # Call the selected function  
            process_function = file_input_function_map.get(input_dict['file_input'])
            process_column = file_input_column_map.get(input_dict['file_input'])

            # Use ThreadPoolExecutor to process rows concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for index, row in df.iterrows():
                    # Submit each row for parallel processing.
                    futures.append(executor.submit(process_function, index, row, input_dict))  # Remove index_dict param
                    # if the text is none or blank, skip the row
                    text_str = row.get(process_column, "")
                    if text_str is None or str(text_str).strip() == "":
                        logger.info(f"Skipping row {index} due to empty text.")
                        continue

                result_all = pd.DataFrame(columns=['Ucid', 'Text', 'question', 'topic', 'sub_topic', 'is_useful']) 

                # Collect results from all submitted tasks.
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if not result.empty:
                        result_all = pd.concat([result_all, result], ignore_index=True)
                        processed_rows += 1
                        percentage_processed = (processed_rows / total_rows) * 100
                        logger.info(f"Processed {processed_rows}/{total_rows} rows ({percentage_processed:.2f}%)")

            if process_column == 'question':
                if 'Ucid' in result_all.columns:
                    result_all = result_all[['Ucid', 'question', 'topic', 'sub_topic', 'is_useful']]
                else:
                    result_all = result_all[['question', 'topic', 'sub_topic', 'is_useful']]
            if process_column == 'Text':
                if 'Ucid' in result_all.columns:
                    result_all = result_all[['Ucid', 'Text', 'question', 'topic', 'sub_topic', 'is_useful']]
                else:
                    result_all = result_all[['Text', 'question', 'topic', 'sub_topic', 'is_useful']]
                process_column = 'conversation'

            write_to_csv(result_all, input_dict['destination_csv'], index=False)

            logger.info(f"Total rows in dataframe: {len(result_all)}")

        else:
            logger.error("Invalid input type. Please use 'index' or 'csv'.")
            raise ValueError("Unsupported input type. Supported types are: index, csv.")

