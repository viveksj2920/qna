# Standard Library Imports
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import time

# Third-Party Imports
import pandas as pd
from dotenv import load_dotenv

# Local Application Imports
import config
from index_topic_backfill import IndexProcessor
from skillsets.vectorizer import AzureOpenAIVectorizer

# configure the llm
from llm.llm_config import chat_completion
from utils.helper import load_project_config, clean_topic, clean_subtopics
from utils.logger_config import logger

import httpx
import asyncio
import nest_asyncio
import aiohttp
from openai import AsyncAzureOpenAI
nest_asyncio.apply()

# Import the Prompts
from prompts.prompt_config import (
    qna_extraction_prompt,
    prompt_topic_format,
    topic_extraction_prompt,
    prompt_sub_topic_format,
    sub_topic_extraction_prompt,
    subtopic_grouping_prompt,
    subtopic_labeling_prompt
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
            sub_topic_prompt_dict = load_project_config(self.project, "subtopic_extraction")
            
            sub_topic_prompt = sub_topic_prompt_dict[topic_cleaned]
            sub_topic_prompt = prompt_sub_topic_format(self.project, sub_topic_prompt)
            
            prompt = sub_topic_extraction_prompt(question=self.question, topic=self.topic, sub_topic_descriptions = sub_topic_prompt)

            cleaned_lines = [line.lstrip() for line in prompt.splitlines()]
            prompt = "\n".join(cleaned_lines)
            
            message = [{"role": "user", "content": prompt}]
    
            sub_topic = chat_completion(messages=message, max_tokens=2000, temperature=1e-9, task_type="sub_topic_extraction")
            
        return sub_topic


    
    def keyword_search(self, subtopic, index_name):
        lookup_index_processor = IndexProcessor(index_name=index_name)
        results = lookup_index_processor.azure_search.search_clients[lookup_index_processor.azure_search.search_region].search(
            filter=f"subtopic eq '{subtopic}'",
            select=["subtopic", "grouped_subtopic"]
        )

        results = [res for res in results]

        if len(results) == 0:
            return None
        
        return results[0]["grouped_subtopic"]

    def semantic_hybrid_search(self, subtopic, index_name):
        lookup_index_processor = IndexProcessor(index_name=index_name)
        results = lookup_index_processor.azure_search.search_clients[lookup_index_processor.azure_search.search_region].search(
            search_text=subtopic,
            search_fields=["subtopic"],
            select=["subtopic", "grouped_subtopic"],
            semantic_configuration_name="my-semantic-config"
        )
        
        threshold = 1.75
        filtered_results = []
        for result in results:
            # Check if the result has semantic reranker score
            if "@search.rerankerScore" in result and result["@search.rerankerScore"] >= threshold:
                filtered_results.append(result)
        
        if not filtered_results:
            return None

        return filtered_results[0]["grouped_subtopic"]
    
    def group_subtopics(self, subtopics):
        subtopics = ", ".join([f'"{topic}"' for topic in subtopics])

        prompt = subtopic_grouping_prompt(subtopics)

        message = [{"role": "user", "content": prompt}]

        groupings = chat_completion(messages=message, max_tokens=2000, temperature=1e-9, task_type="subtopic_grouping")

        return groupings
    
    def process_index_row(self, index, row, input_dict, qnas_with_ungrouped_subtopics, all_ungrouped_subtopics):

        max_retries = 5
        record_id = row['id']
        question = row['question']
        ucid = row['Ucid']
        project = input_dict["project"]

        logger.info(f"Processing row {index} with ID: {record_id}, UCID: {ucid}")
    
        for attempt in range(max_retries):
            try:
                # Process the single question directly
                expanded_qna = []

                # Create the basic Q&A entry using the provided data
                expanded_qna_1 = {
                    "id": record_id,
                    "Ucid": ucid,
                    "question": question,
                    "@search.action": "mergeOrUpload"
                }
                
                # Perform topic extraction and add topic to doc if available
                extractor = qna_extractor(question=question, project=project)
                qna_topic_data_json = extractor.extract_topic()

                if qna_topic_data_json:
                    try:
                        qna_topic_data = json.loads(qna_topic_data_json)
                        expanded_qna_1["topic"] = qna_topic_data['topic']
                    except Exception as e:
                        logger.error(f"Error processing topic for ID {record_id}: {e}")

                # Perform subtopic extraction and add subtopic to doc if available
                extractor = qna_extractor(question=question, topic=expanded_qna_1.get("topic"), project=project)
                qna_subtopic_data_json = extractor.extract_subtopic()
                if qna_subtopic_data_json:
                    try:
                        qna_subtopic_data = json.loads(qna_subtopic_data_json)

                        if isinstance(qna_subtopic_data['sub_topic'], str):
                            qna_subtopic_data['sub_topic'] = [qna_subtopic_data['sub_topic']]

                        expanded_qna_1["sub_topic"] = qna_subtopic_data['sub_topic']

                        if project == "MIRA":
                            subtopics = clean_subtopics(project, qna_subtopic_data['sub_topic'])
                            expanded_qna_1["useful_sub_topics"] = subtopics
                            grouped_subtopics = set()
                            ungrouped_subtopics = []
                            for subtopic in subtopics:
                                # Do a keyword search first
                                label = extractor.keyword_search(subtopic, input_dict["lookup_index"])
                                if label:
                                    logger.info(f"label found from keyword search: {label}")
                                    grouped_subtopics.add(label)
                                    continue
                                
                                # If not found, do a semantic search
                                label = extractor.semantic_hybrid_search(subtopic, input_dict["lookup_index"])
                                if label:
                                    logger.info(f"label found from semantic search: {label}")
                                    grouped_subtopics.add(label)
                                    # Insert into lookup index
                                    insert = {
                                        "id": hashlib.sha256(subtopic.encode()).hexdigest(),
                                        "subtopic": subtopic,
                                        "grouped_subtopic": label
                                    }
                                    lookup_index_processor = IndexProcessor(index_name=input_dict["lookup_index"])
                                    lookup_index_processor.update_index(
                                        [insert],
                                        key_field_name="id",
                                        semantic_content_field="subtopic"
                                    )
                                    continue

                                # If still not found, save for later grouping
                                logger.info(f"Label not found, saving subtopic for later grouping: {subtopic}")
                                ungrouped_subtopics.append(subtopic)
                                all_ungrouped_subtopics.append(subtopic)

                            # If no more subtopics need to be grouped, then processing is finished
                            if not ungrouped_subtopics:
                                expanded_qna_1["grouped_sub_topic"] = list(grouped_subtopics)
                            else:
                                qnas_with_ungrouped_subtopics.append({"id": record_id, "grouped_sub_topic": grouped_subtopics, "ungrouped_sub_topic": ungrouped_subtopics})
                        elif project == "PCL":
                            expanded_qna_1["grouped_sub_topic"] = qna_subtopic_data['sub_topic']
                        else:
                            logger.error(f"Unexpected project type: {project}")
                            raise ValueError("Unsupported project type. Supported projects are: MIRA, PCL.")

                    except Exception as e:
                        logger.error(f"Error processing subtopics for ID {record_id}: {e}")



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
                
            except Exception as e:
                logger.error(f"Error processing for ID {record_id} on attempt {attempt+1}: {e}")

                # If a rate-limit error is encountered, pause briefly.
                if "RateLimitError" in str(e):
                    logger.info("Rate limit exceeded. Waiting for 1 min before retrying.")
                    time.sleep(60)

                # Upon the final retry, log the error and move on.
                if attempt == max_retries - 1:
                    with open('error.txt', 'a') as f:
                        f.write(f"Row ID: {record_id}, Error: {e}\n")
                    logger.error(f"Moving on to the next ID after {max_retries} attempts.")
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

                    df_expanded_qna = pd.DataFrame(columns=['id', 'question', 'topic', 'sub_topic', 'useful_sub_topics', 'grouped_sub_topic'])

                    for index_q, row_q in df.iterrows():

                        df_expanded_qna_1 = self.process_file_question_row(index_q, row_q, input_dict)

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
                # Read both 'id' and 'question' from the DataFrame row
                question_1 = row['question']
                record_id = row.get('id', None)
                
                extractor = qna_extractor(question=question_1, project=project_1)
                qna_topic_data_json = extractor.extract_topic()

                if qna_topic_data_json:
                    try:
                        qna_topic_data = json.loads(qna_topic_data_json)

                        if isinstance(qna_topic_data['topic'], str):
                            qna_topic_data['topic'] = qna_topic_data['topic']

                        if isinstance(qna_topic_data['topic'], list):
                            qna_topic_data['topic'] = qna_topic_data['topic'][0]

                        # Add both id and question to the output
                        expanded_qna_1["id"] = record_id
                        expanded_qna_1["question"] = question_1
                        expanded_qna_1["topic"] = qna_topic_data['topic']
                        
                    except Exception as e:
                        logger.error(f"Error processing topic for Row {index + 1}: {e}")

                    # Perform subtopic extraction and add subtopic to doc if available.
                    extractor = qna_extractor(question=question_1, topic=expanded_qna_1["topic"], project=project_1)
                    qna_subtopic_data_json = extractor.extract_subtopic()
                    if qna_subtopic_data_json:
                        try:
                            qna_subtopic_data = json.loads(qna_subtopic_data_json)

                            if isinstance(qna_subtopic_data['sub_topic'], str):
                                qna_subtopic_data['sub_topic'] = [qna_subtopic_data['sub_topic']]

                            expanded_qna_1["sub_topic"] = qna_subtopic_data['sub_topic']
                            
                            # Process subtopics to create useful_sub_topics and grouped_sub_topic
                            subtopics = clean_subtopics(project_1, qna_subtopic_data['sub_topic'])
                            expanded_qna_1["useful_sub_topics"] = subtopics
                            
                            # Process subtopic grouping
                            grouped_subtopics = []
                            if input_dict.get("lookup_index"):
                                for subtopic in subtopics:
                                    # Try keyword search first
                                    label = self.keyword_search(subtopic, input_dict["lookup_index"])
                                    if label:
                                        grouped_subtopics.append(label)
                                    else:
                                        # Try semantic search if keyword search fails
                                        label = self.semantic_hybrid_search(subtopic, input_dict["lookup_index"])
                                        if label:
                                            grouped_subtopics.append(label)
                                        else:
                                            # If no grouping found, use the original subtopic
                                            grouped_subtopics.append(subtopic)
                            else:
                                # If no lookup index provided, use original subtopics
                                grouped_subtopics = subtopics
                                
                            expanded_qna_1["grouped_sub_topic"] = list(set(grouped_subtopics))
                            
                        except Exception as e:
                            logger.error(f"Error processing subtopic for Row {index + 1}: {e}")



                    df_expanded_qna_1 = pd.DataFrame([expanded_qna_1])
                    
                    # Return only the required fields: id, question, topic, sub_topic, useful_sub_topics, grouped_sub_topic
                    required_columns = ['id', 'question', 'topic', 'sub_topic', 'useful_sub_topics', 'grouped_sub_topic']
                    df_expanded_qna_1 = df_expanded_qna_1[required_columns]

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

    def process_qna_with_id_row(self, index, row, input_dict):
        """
        Process rows that already have id and question fields.
        Extracts topic, subtopic, useful_sub_topics, and grouped_sub_topic.
        """
        max_retries = 5
        project_1 = input_dict["project"]
        
        for attempt in range(max_retries):
            try:
                # Read both 'id' and 'question' from the DataFrame row
                question_1 = row['question']
                record_id = row['id']
                
                # Initialize the result dictionary
                expanded_qna_1 = {
                    "id": record_id,
                    "question": question_1,
                    "topic": None,
                    "sub_topic": None,
                    "useful_sub_topics": None,
                    "grouped_sub_topic": None
                }
                
                # Extract topic
                extractor = qna_extractor(question=question_1, project=project_1)
                qna_topic_data_json = extractor.extract_topic()

                if qna_topic_data_json:
                    try:
                        qna_topic_data = json.loads(qna_topic_data_json)

                        if isinstance(qna_topic_data['topic'], str):
                            qna_topic_data['topic'] = qna_topic_data['topic']

                        if isinstance(qna_topic_data['topic'], list):
                            qna_topic_data['topic'] = qna_topic_data['topic'][0]

                        expanded_qna_1["topic"] = qna_topic_data['topic']
                        
                    except Exception as e:
                        logger.error(f"Error processing topic for Row {index + 1}: {e}")

                    # Perform subtopic extraction
                    extractor = qna_extractor(question=question_1, topic=expanded_qna_1["topic"], project=project_1)
                    qna_subtopic_data_json = extractor.extract_subtopic()
                    if qna_subtopic_data_json:
                        try:
                            qna_subtopic_data = json.loads(qna_subtopic_data_json)

                            if isinstance(qna_subtopic_data['sub_topic'], str):
                                qna_subtopic_data['sub_topic'] = [qna_subtopic_data['sub_topic']]

                            expanded_qna_1["sub_topic"] = qna_subtopic_data['sub_topic']
                            
                            # Process subtopics to create useful_sub_topics and grouped_sub_topic
                            subtopics = clean_subtopics(project_1, qna_subtopic_data['sub_topic'])
                            expanded_qna_1["useful_sub_topics"] = subtopics
                            
                            # Process subtopic grouping
                            grouped_subtopics = []
                            if input_dict.get("lookup_index"):
                                for subtopic in subtopics:
                                    # Try keyword search first
                                    label = self.keyword_search(subtopic, input_dict["lookup_index"])
                                    if label:
                                        grouped_subtopics.append(label)
                                    else:
                                        # Try semantic search if keyword search fails
                                        label = self.semantic_hybrid_search(subtopic, input_dict["lookup_index"])
                                        if label:
                                            grouped_subtopics.append(label)
                                        else:
                                            # If no grouping found, use the original subtopic
                                            grouped_subtopics.append(subtopic)
                            else:
                                # If no lookup index provided, use original subtopics
                                grouped_subtopics = subtopics
                                
                            expanded_qna_1["grouped_sub_topic"] = list(set(grouped_subtopics))
                            
                        except Exception as e:
                            logger.error(f"Error processing subtopic for Row {index + 1}: {e}")

                # Create DataFrame with required columns
                df_expanded_qna_1 = pd.DataFrame([expanded_qna_1])
                required_columns = ['id', 'question', 'topic', 'sub_topic', 'useful_sub_topics', 'grouped_sub_topic']
                df_expanded_qna_1 = df_expanded_qna_1[required_columns]

                return df_expanded_qna_1
                
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
                    
        # Return empty dataframe if all attempts failed
        return pd.DataFrame(columns=['id', 'question', 'topic', 'sub_topic', 'useful_sub_topics', 'grouped_sub_topic'])

    async def extract_batch_async(self, df, input_dict, max_concurrent=60):
        """Async version of extract_batch using asyncio for concurrent processing"""
        
        processed_rows = 0
        try:
            all_expanded_qna = []
            all_expanded_qna_processed = []
            total_rows = len(df)
            
            # Add these for subtopic grouping
            qnas_with_ungrouped_subtopics = []
            all_ungrouped_subtopics = []

            rate_limiter = AdaptiveRateLimiter(initial_qps=40, max_qps=80)
            
            # Create a semaphore to limit concurrent tasks
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = []
            
            # Create tasks for all rows
            for index, row in df.iterrows():
                question_str = row.get("question", "")
                if question_str is None or question_str.strip() == "":
                    logger.info(f"Skipping row {index} due to empty question.")
                    continue
                
                # Add the task with the semaphore and the subtopic grouping collections
                task = self.process_index_row_async(
                    semaphore, 
                    rate_limiter, 
                    index, 
                    row, 
                    input_dict, 
                    qnas_with_ungrouped_subtopics, 
                    all_ungrouped_subtopics
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
                            await self.upload_batch_async(
                                filtered_batch, 
                                input_dict["destination_index"], 
                                "id", 
                                "question"
                            )
                            all_expanded_qna = all_expanded_qna[100:]
                        
                        
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
            
            # Upload any remaining items
            if all_expanded_qna:
                await self.upload_batch_async(
                    all_expanded_qna, 
                    input_dict["destination_index"], 
                    "id", 
                    "question"
                )
            
            # After processing all rows, handle the ungrouped subtopics for MIRA project
            if input_dict["project"] == "MIRA" and all_ungrouped_subtopics:
                await self.process_ungrouped_subtopics_async(
                    all_ungrouped_subtopics,
                    qnas_with_ungrouped_subtopics,
                    input_dict
                )
            
        finally:
            # Close the session when done
            if hasattr(self, 'http_session') and self.http_session:
                await self.http_session.aclose()
        
        return processed_rows

    async def process_ungrouped_subtopics_async(self, all_ungrouped_subtopics, qnas_with_ungrouped_subtopics, input_dict):
        """Process ungrouped subtopics asynchronously"""
        logger.info(f"Processing {len(all_ungrouped_subtopics)} ungrouped subtopics")

        # logger.info(f"Ungrouped subtopics: {all_ungrouped_subtopics}")
        
        try:
            # Group all ungrouped subtopics
            batch_size = 100
            grouped_subtopics = []
            
            for i in range(0, len(all_ungrouped_subtopics), batch_size):
                batch = all_ungrouped_subtopics[i:i + batch_size]
                max_retries = 5
                
                for attempt in range(1, max_retries + 1):
                    try:
                        # Run in a thread to avoid blocking
                        loop = asyncio.get_event_loop()
                        batch_res = await loop.run_in_executor(
                            None,
                            lambda: self.group_subtopics(batch)
                        )
                        
                        if batch_res:
                            batch_res = json.loads(batch_res)["categories"]
                            grouped_subtopics.extend(batch_res)
                            break
                            
                    except Exception as e:
                        logger.error(f"Error during async grouping attempt {attempt} for batch {i//batch_size + 1}: {e}")
                        if attempt < max_retries:
                            await asyncio.sleep(60)
                        else:
                            logger.error(f"Max retries reached for batch {i//batch_size + 1}. Moving to next batch.")
            
            # Label grouped subtopics
            labeled_groupings = []
            for group in grouped_subtopics:
                # Run in a thread to avoid blocking
                loop = asyncio.get_event_loop()
                label = await loop.run_in_executor(
                    None,
                    lambda: self.label_group(group)
                )
                labeled_groupings.append({"subtopics": group, "label": label})
            
            new_grouped_lookup = {
                subtopic: {
                    "subtopic": subtopic,
                    "grouped_subtopic": group["label"],
                    "vector": await self.vectorize_text_async(subtopic),
                    "id": hashlib.sha256(subtopic.encode()).hexdigest()
                }
                for group in labeled_groupings
                for subtopic in group["subtopics"]
                if group["label"]  # Skip if no label was generated
            }

            # logger.info(f"New grouped lookup: {new_grouped_lookup}")
            
            # Upload new grouped subtopics to lookup index
            if new_grouped_lookup:
                await self.upload_batch_async(
                    list(new_grouped_lookup.values()),
                    input_dict["lookup_index"],
                    "id",
                    "subtopic"
                )
            
            # Reprocess QnA documents with previously ungrouped subtopics
            processed_qnas = []
            for qna in qnas_with_ungrouped_subtopics:
                # Loop through the ungrouped subtopics and try to find their new grouping
                for subtopic in qna["ungrouped_sub_topic"]:
                    if subtopic in new_grouped_lookup:
                        qna["grouped_sub_topic"].add(new_grouped_lookup[subtopic]["grouped_subtopic"])
                    else:
                        qna["grouped_sub_topic"].add(subtopic)
                
                # Once all the subtopics for this QnA are now grouped, add to processed QnAs
                processed_qnas.append({
                    "id": qna["id"],
                    "grouped_sub_topic": list(qna["grouped_sub_topic"])
                })
            
            # Upload the processed QnAs with grouped subtopics
            if processed_qnas:
                await self.upload_batch_async(
                    processed_qnas,
                    input_dict["destination_index"],
                    "id"
                )
                
        except Exception as e:
            logger.error(f"Error in async ungrouped subtopics processing: {e}")


    async def process_index_row_async(self, semaphore, rate_limiter, index, row, input_dict, qnas_with_ungrouped_subtopics=None, all_ungrouped_subtopics=None):
        """Async version of process_index_row that processes a single question with subtopic grouping"""
        
        if qnas_with_ungrouped_subtopics is None:
            qnas_with_ungrouped_subtopics = []
        if all_ungrouped_subtopics is None:
            all_ungrouped_subtopics = []
        
        async with semaphore:  # Limit concurrent processing
            max_retries = 5
            record_id = row['id']
            question = row['question']
            ucid = row['Ucid']
            project = input_dict["project"]

            if question is None or question.strip() == "":
                logger.info(f"Skipping row {index} with ID: {record_id} due to empty question.")
                return None
            
            logger.info(f"Processing row {index} with ID: {record_id}, UCID: {ucid}")
            
            for attempt in range(max_retries):
                try:
                    # Wait for rate limiter
                    await rate_limiter.acquire()
                    
                    # Process the single question directly
                    expanded_qna = []
                    
                    # Create the basic QnA entry using the provided data
                    expanded_qna_1 = {
                        "id": record_id,
                        "Ucid": ucid,
                        "question": question,
                        "@search.action": "mergeOrUpload"
                    }
                    
                    # Process topic and subtopic in parallel
                    task = self.process_question_metadata_async(
                        question, 
                        project, 
                        expanded_qna_1, 
                        rate_limiter
                    )
                    
                    # Wait for processing to complete
                    result = await task
                    
                    if result is not None:
                        expanded_qna.append(result)
                        
                        # Add subtopic grouping logic for MIRA project
                        if project == "MIRA" and "sub_topic" in result:
                            await self.process_subtopic_grouping_async(
                                result,
                                qnas_with_ungrouped_subtopics,
                                all_ungrouped_subtopics,
                                input_dict["lookup_index"],
                                project
                            )
                        elif project == "PCL" and "sub_topic" in result:
                            result["grouped_sub_topic"] = result['sub_topic']
                    
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
                    
                except Exception as e:
                    logger.error(f"Error processing for ID {record_id} on attempt {attempt+1}: {e}")
                    
                    # If rate limit error, adjust the limiter
                    if "RateLimitError" in str(e) or "429" in str(e):
                        await rate_limiter.on_error(429)
                        logger.info("Rate limit exceeded. Waiting before retrying.")
                        await asyncio.sleep(min(2 ** attempt, 60))  # Exponential backoff
                    
                    if attempt == max_retries - 1:
                        with open('error.txt', 'a') as f:
                            f.write(f"Row ID: {record_id}, Error: {e}\n")
                        break
                        
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


    async def process_question_metadata_async(self, question, project, expanded_qna_entry, rate_limiter):
        """Process topic and subtopic for a question"""
        
        # Wait for rate limiter before starting
        await rate_limiter.acquire()
        
        # Start topic extraction
        topic_task = self.extract_topic_async(question, project, rate_limiter)
        

        
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
                
                # Wait for subtopic task
                subtopic_result = await subtopic_task
                
                # Process subtopic result
                if subtopic_result:
                    try:
                        cleaned_subtopic = subtopic_result.strip().replace("```json", "").replace("```", "").strip()
                        cleaned_subtopic = cleaned_subtopic.lstrip('\n').rstrip('\n')
                        
                        subtopic_data = json.loads(cleaned_subtopic)

                        if isinstance(subtopic_data['sub_topic'], str):
                            subtopic_data['sub_topic'] = [subtopic_data['sub_topic']]
                        expanded_qna_entry["sub_topic"] = subtopic_data['sub_topic']
                    except Exception as e:
                        logger.error(f"Error processing subtopic: {e}")
                

                return expanded_qna_entry
                
            except Exception as e:
                logger.error(f"Error processing topic: {e} {topic_result, question}")
                raise
        
        return None
    
    async def process_subtopic_grouping_async(self, expanded_qna_entry, qnas_with_ungrouped_subtopics, all_ungrouped_subtopics, lookup_index_name, project):
        """Process subtopic grouping asynchronously"""
        try:
            if "sub_topic" not in expanded_qna_entry:
                expanded_qna_entry["sub_topic"] = []
                expanded_qna_entry["useful_sub_topics"] = []
                expanded_qna_entry["grouped_sub_topic"] = []
                return
                
            # Clean the subtopics
            subtopics = clean_subtopics(project, expanded_qna_entry["sub_topic"])
            
            if not subtopics:
                expanded_qna_entry["useful_sub_topics"] = []
                expanded_qna_entry["grouped_sub_topic"] = []
                return
            
            expanded_qna_entry["useful_sub_topics"] = subtopics
                
            # Process grouping
            grouped_subtopics = set()
            ungrouped_subtopics = []
            
            for subtopic in subtopics:
                # Do a keyword search first
                label = await self.keyword_search_async(subtopic, lookup_index_name)
                
                if label:
                    grouped_subtopics.add(label)
                    continue
                
                # If not found, do a semantic search
                label = await self.semantic_hybrid_search_async(subtopic, lookup_index_name)
                
                if label:
                    grouped_subtopics.add(label)
                    # Insert into lookup index asynchronously
                    await self.insert_subtopic_lookup_async(subtopic, label, lookup_index_name)
                    continue
                
                # If still not found, save for later grouping
                ungrouped_subtopics.append(subtopic)
                all_ungrouped_subtopics.append(subtopic)
            
            # If no more subtopics need to be grouped, then processing is finished
            if not ungrouped_subtopics:
                expanded_qna_entry["grouped_sub_topic"] = list(grouped_subtopics)
            else:
                qnas_with_ungrouped_subtopics.append({
                    "id": expanded_qna_entry["id"], 
                    "grouped_sub_topic": grouped_subtopics, 
                    "ungrouped_sub_topic": ungrouped_subtopics
                })
                
        except Exception as e:
            logger.error(f"Error in subtopic grouping for {expanded_qna_entry.get('id', 'unknown')}: {e}")


    async def keyword_search_async(self, subtopic, lookup_index_name):
        """Async version of keyword search"""
        try:
            # Run the sync method in an executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.keyword_search(subtopic, lookup_index_name)
            )
            return result
        except Exception as e:
            logger.error(f"Error in async keyword search for {subtopic}: {e}")
            return None

    async def semantic_hybrid_search_async(self, subtopic, lookup_index_name):
        """Async version of semantic hybrid search"""
        try:
            # Run the sync method in an executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.semantic_hybrid_search(subtopic, lookup_index_name)
            )
            return result
        except Exception as e:
            logger.error(f"Error in async semantic search for {subtopic}: {e}")
            return None
            
    async def insert_subtopic_lookup_async(self, subtopic, label, lookup_index_name):
        """Insert a new subtopic-to-label mapping into the lookup index"""
        try:
            # Create the document
            insert = {
                "id": hashlib.sha256(subtopic.encode()).hexdigest(),
                "subtopic": subtopic,
                "grouped_subtopic": label
            }
            
            # Run the update in an executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: IndexProcessor(index_name=lookup_index_name).update_index(
                    [insert],
                    key_field_name="id", 
                    semantic_content_field="subtopic"
                )
            )
            
        except Exception as e:
            logger.error(f"Error inserting subtopic mapping for {subtopic}: {e}")

    
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
            sub_topic_prompt_dict = load_project_config(project, "subtopic_extraction")
            
            sub_topic_prompt = sub_topic_prompt_dict[topic_cleaned]
            sub_topic_prompt = prompt_sub_topic_format(project, sub_topic_prompt)
            
            prompt = sub_topic_extraction_prompt(question=question, topic=topic, sub_topic_descriptions=sub_topic_prompt)
            
            cleaned_lines = [line.lstrip() for line in prompt.splitlines()]
            prompt = "\n".join(cleaned_lines)
            
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



    async def chat_completion_async(self, messages, max_tokens=2000, temperature=1e-9, task_type="default"):
        """Async version of chat_completion"""
        try:
            session = await self.get_session()
     
            client = AsyncAzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                http_client=session
            )
            
            response = await client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
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


    def get_substring_labeling(self, subtopics):
        for subtopic in subtopics:
            if all(subtopic in other for other in subtopics):
                return subtopic
        return None

    def llm_labeling(self, subtopics):
        subtopic_list = ", ".join([f'"{subtopic}"' for subtopic in subtopics])
        
        prompt = subtopic_labeling_prompt(subtopic_list)

        message = [{"role": "user", "content": prompt}]

        labeled_groups = chat_completion(messages=message, max_tokens=2000, temperature=1e-9, task_type="subtopic_group_labeling")

        return labeled_groups

    def label_group(self, group, max_retries=5, wait_seconds=60):
        # Try substring match
        label = self.get_substring_labeling(group)
        if label:
            return label
        
        # LLM
        for attempt in range(1, max_retries + 1):
            try:
                label = self.llm_labeling(group)
                label = label.lower().replace("\"", '')
            except Exception as e:
                logger.error(f"Error during LLM labeling attempt {attempt}: {e}")
                if attempt < max_retries:
                    time.sleep(wait_seconds)
                else:
                    logger.error("Max retries reached for LLM labeling")
                    label = None
        return label

    def extract_batch(self, df, input_dict, max_workers=12):

        if input_dict['input_type'] == "index":
            return asyncio.run(self.extract_batch_async(df, input_dict, max_workers))

        else:
            logger.error("Invalid input type. Please use 'index'")
            raise ValueError("Unsupported input type. Supported types are: index")