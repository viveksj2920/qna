import os
from dotenv import load_dotenv
from utils.logger_config import logger

# Load environment variables from the root folder outside src
dotenv_path=os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path, override=True)

try:
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-transcripts")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    # Get the directory where this config file is located (src directory)
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input JSON File - use absolute paths relative to src directory
    PCL_TOPIC_JSON = os.getenv("PCL_TOPIC_JSON", os.path.join(_src_dir, "data/input/topic_prompt_pcl.json"))
    PCL_SUBTOPIC_JSON = os.getenv("PCL_SUBTOPIC_JSON", os.path.join(_src_dir, "data/input/sub_topic_prompt_pcl.json"))
    PCL_QUESTION_JSON = os.getenv("PCL_QUESTION_JSON", os.path.join(_src_dir, "data/input/questions_pcl.json"))
    PCL_SENTIMENT_JSON = os.getenv("PCL_SENTIMENT_JSON", os.path.join(_src_dir, "data/input/sentiment_config_pcl.json"))
    MIRA_TOPIC_JSON = os.getenv("MIRA_TOPIC_JSON", os.path.join(_src_dir, "data/input/topic_prompt_mira.json"))
    MIRA_SUBTOPIC_JSON = os.getenv("MIRA_SUBTOPIC_JSON", os.path.join(_src_dir, "data/input/sub_topic_prompt_mira.json"))
    MIRA_NEW_SUBTOPIC_JSON = os.getenv("MIRA_NEW_SUBTOPIC_JSON", os.path.join(_src_dir, "data/input/new_sub_topic_prompt_mira.json"))
    MIRA_QUESTION_JSON = os.getenv("MIRA_QUESTION_JSON", os.path.join(_src_dir, "data/input/questions_mira.json"))
    MIRA_SENTIMENT_JSON = os.getenv("MIRA_SENTIMENT_JSON", os.path.join(_src_dir, "data/input/sentiment_config_mira.json"))

    # Check for required variables
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_KEY": AZURE_OPENAI_KEY,
        "AZURE_OPENAI_DEPLOYMENT": AZURE_OPENAI_DEPLOYMENT,
        "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION
    }
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    else:
        logger.info("All required environment variables loaded successfully.")

except Exception as e:
    logger.exception(f"Error loading configuration: {e}")
    raise

    