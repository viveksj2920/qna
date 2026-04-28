# README: QnA Batch Processing Script

### Overview
The script main.py, performs an end-to-end process to fetch documents from an Azure Search index, process them using Azure OpenAI, and update the Azure Search index with new fields. The script is designed to handle large datasets by fetching documents in batches, processing each document to extract metadata, and then uploading the processed QnA data back to the Azure Search index.

### Assumptions

1. **Azure OpenAI Configuration**:
   - The script assumes that you have an Azure OpenAI deployment with the necessary API key and endpoint.
   - The api_key, endpoint, and api_version, should be correctly configured.

2. **Azure Search Configuration**:
   - The script assumes that you have an Azure Search service with the necessary API key and endpoint.
   - The search_service_name, index_name, search_api_key, and search_api_version, should be correctly configured.

3. **Index Fields**:
   - The script expects the following fields to be present in the Azure Search source index:
     - `Ucid`: Unique identifier for each document.
     - `Text`: The text content of the document to be processed.
     - `StartTime`: The StartTime content of the document to be processed.
     - `Is_Digital`: The is_digital content of the document to be processed.
   - The script will add the following new fields to the destination index schema if they do not already exist:
     - `Ucid`, `is_digital`, `question`, `answer`

### Process

1. **Fetch Documents**:
   - If input_type='index', the script fetches documents from the Azure Search index that have "qa_processed_time" as null 
   - If input_type='file', 
      - If file_input='conversations', the script fetches documents from the conversations file.
      - If file_input='questions', the script fetches documents from the questions file.

2. **Process Documents**:
   - Each document is processed using Azure OpenAI to extract metadata.

3. **Update Index Schema**:
   - The script takes the Azure Search index schema to include the new fields if they do not already exist.

4. **Upload Processed Documents**:
   - The processed documents are uploaded back to the Azure Search index one Ucid at a time.

4. **Update Metadata Index with qa_processed_time**:
   - The processed documents are uploaded back to the Azure Search index one Ucid at a time.

### Output

- **Error file**:
  - error.txt captures the Ucids that failed in the process

- **Runtime file**:
   - run_time.txt captures the run time of the process for every 500 transcriptions.

# Scripts Flow
- Fetches documents from the Azure Search index.
- Analyzes and extracts metadata using Azure OpenAI.
- Upload the QnA data back to the Azure Search index.

### Usage
1. Ensure that the Azure OpenAI and Azure Search configurations are correctly set in the .env file.

2. **Important**: All scripts should be run from the `src` directory.

3. For batch processing, use the qna_batch.py script with the following required arguments:
   - `--input_type`: type of input (must be either "index" or "file")
   - `--source_data_name`: 
      - If input_type='index', name of the source Azure Search index containing documents to process
      - If input_type='file', file path of the source conversations file or source questions file
   - `--destination_data_name`:
      - If input_type='index', name of the destination Azure Search index containing documents to process
      - If input_type='file', file path of the destination conversations file or destination questions file
   - `--project`: Project type (must be either "MIRA" or "PCL")

   Example commands (run from the src directory):
   ```
   # Navigate to src directory first
   cd src

   # For index-based processing
   python main.py --dry_run --input_type="index" --source_data_name="transcripts-mira" --lookup_data_name="transcripts-qna-subtopic-groupings-test-1" --destination_data_name="transcript-index-test-qna-v1" --project="MIRA" --start_date="2025-10-07T00:00:00Z" --end_date="2025-10-07T00:00:00Z"
   python main.py --input_type="index" --source_data_name="transcripts-mira" --lookup_data_name="transcripts-qna-subtopic-groupings" --destination_data_name="transcript-index-test-qna-v1" --project="MIRA" 
   # PCL
   python main.py 
   --dry_run 
   --input_type="index" 
   --project="PCL" 
   --source_data_name="transcripts-pcl-sentiment" 
   --destination_data_name="transcript-index-test-qna-v1"  
   --lookup_data_name="transcripts-qna-subtopic-groupings-test-1" --start_date="2025-07-31T00:00:00Z" --end_date="2025-07-31T00:00:00Z"

   # For file-based processing with conversations
   python main.py --input_type="file" --source_data_name="data/input/raw_convs.csv" --lookup_data_name="transcripts-qna-subtopic-groupings"  --destination_data_name="data/output/output_conversations.csv" --project="MIRA" --file_input="conversations"

   # For file-based processing with questions
   python main.py --input_type="file" --source_data_name="data/input/questions.csv" --lookup_data_name="transcripts-qna-subtopic-groupings" --destination_data_name="data/output/output_questions.csv" --project="MIRA" --file_input="questions"
   ```

3. Project-specific prompts:
   - The system uses different prompt templates depending on the project type:
      - Topic prompt:
         - PCL projects use templates from `topic_prompt_pcl.json`
         - MIRA projects use templates from `topic_prompt_mira.json`
      - Sub-topic prompt:
         - PCL projects use templates from `sub_topic_prompt_pcl.json`
         - MIRA projects use templates from `sub_topic_prompt_mira.json`
   - These JSON files contain topic and sub-topic classification prompts for the respective project types
   - Each project has its own extraction logic and field requirements

4. Common files:
- qna_metadata_extractor.py: This file has code logic to extract QnA metadata from the transcript using Azure OpenAI LLM and also captures errors/run_time.
- index.py: This file is used to connect, fetch and update records in Azure AI Search index


### Notes
- Replace placeholder values with your actual Azure API keys and endpoints.
- Ensure that the Azure Search index schema is compatible with the fields expected by the script.
- The script handles large datasets by processing documents in batches to avoid exceeding API limits.

### 5. Sentiment fields into Azure AI Search Index 

Two new fields are added to the search index:
- `sentiment_level_1`: String field containing high-level sentiment
- `sentiment_level_2`: String field containing detailed sentiment subcategory

### Configuration Files
- `src/data/input/sentiment_config_mira.json`: MIRA sentiment configuration
- `src/data/input/sentiment_config_pcl.json`: PCL sentiment configuration

# Sentiment Backfill 

## Overview

The Sentiment Backfill functionality provides a comprehensive solution for extracting and analyzing sentiment from existing QnA records. It processes questions in batches to classify them into sentiment categories at two levels of granularity.

#### Basic Index Backfill
```bash
python src/main_sentiment_backfill.py --input_type index --source_data_name "mira_qna_index" --destination_data_name "mira_qna_index" --project MIRA --sentiment_filter missing --batch_size 100
```

#### Basic File Backfill
```bash
python src/main_sentiment_backfill.py --input_type file --source_data_name "data/input/questions.csv" --destination_data_name "data/output/questions_with_sentiment.csv" --project PCL --batch_size 50
