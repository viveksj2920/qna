import json
import pandas as pd
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import config 
from utils.logger_config import logger


def read_json_file(file_path):
    """
    Reads a JSON file and returns the loaded data.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        Exception: For any other unexpected errors.
    """
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}")
        raise

# Function to write to a csv file using pandas with error handling
def write_to_csv(data, file_path, index=False):
    """
    Writes data to a CSV file.

    Args:
        data (list of dict): Data to write to the CSV file.
        file_path (str): Path to the output CSV file.

    Raises:
        Exception: For any unexpected errors during file writing.
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=index)
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to {file_path}: {e}")
        raise

# Function to load the json data based on project name
def load_project_config(project, prompt_name):
    """
    Loads the project configuration from a JSON file based on the project name and prompt type.

    Args:
        project (str): Name of the project. Supported values are "MIRA" and "PCL".
        prompt_name (str): Type of prompt. Supported values are "topic_extraction", "subtopic_extraction", and "useful_questions".

    Returns:
        dict: Project configuration data loaded from the corresponding JSON file.

    Raises:
        ValueError: If the project name or prompt name is not recognized.
    """
    # Define a mapping of project names to their respective configuration files
    project_config_map = {
        "MIRA": {
            "topic_extraction": config.MIRA_TOPIC_JSON,
            "subtopic_extraction": config.MIRA_SUBTOPIC_JSON,
            "useful_questions": config.MIRA_QUESTION_JSON,
            "sentiment_extraction": config.MIRA_SENTIMENT_JSON
        },
        "PCL": {
            "topic_extraction": config.PCL_TOPIC_JSON,
            "subtopic_extraction": config.PCL_SUBTOPIC_JSON,
            "useful_questions": config.PCL_QUESTION_JSON,
            "sentiment_extraction": config.PCL_SENTIMENT_JSON
        }
    }

    # Validate the project name
    if project not in project_config_map:
        raise ValueError(f"Unsupported project type: {project}. Supported projects are: {', '.join(project_config_map.keys())}.")

    # Validate the prompt name
    if prompt_name not in project_config_map[project]:
        raise ValueError(f"Unsupported prompt name for {project}. Supported prompts are: {', '.join(project_config_map[project].keys())}.")

    # Load the JSON data using the appropriate file path
    json_file_path = project_config_map[project][prompt_name]
    logger.debug(f"Loading configuration for project {project}, promp name {prompt_name} from {json_file_path}")
    json_data = read_json_file(json_file_path)

    return json_data

# Topic cleaning function
def clean_topic(project, topic):
    """
    Cleans and formats a topic string based on the project type.

    Args:
        project (str): Name of the project. Supported values are "MIRA" and "PCL".
        topic (str): The topic string to be cleaned.

    Returns:
        str: The cleaned and formatted topic string.

    Raises:
        ValueError: If the project name is not recognized.
    """
    if not isinstance(topic, str):
        raise ValueError("The topic must be a string.")

    if project == "PCL":
        # Convert to lowercase, replace spaces with underscores, and strip leading/trailing whitespace
        topic_cleaned = topic.lower().replace(" ", "_").strip()
    elif project == "MIRA":
        # Convert to lowercase and strip leading/trailing whitespace
        topic_cleaned = topic.lower().strip()
    else:
        raise ValueError(f"Unsupported project type: {project}. Supported projects are 'MIRA' and 'PCL'.")

    return topic_cleaned


def clean_subtopics(project, subtopics):
        
    if project == "MIRA":
        
        unuseful_subtopics = [
            'a', 'aap medicare advantage', 'aarp', 'aarp healthcare plan', 'aarp medicare', 'aarp medicare advantage', 'aarp medicare advantage essentials from united healthcare',
            'aarp medicare advantage essentials plan', 'aarp medicare advantage from uhc', 'aarp medicare advantage from united healthcare', 'aarp medicare advantage plan',
            'aarp medicare rx from united healthcare', 'aarp medicare supplement insurance plan', 'aarp plan', 'aarp plans', 'aarp supplement', 'aarp supplement and advantage plans',
            'aarp supplemental health insurance plans', 'aarp supplements', 'aarp united healthcare', 'aarp united healthcare medicare advantage', 'aarp united healthcare plan',
            'advantage medicare plan', 'advantage plan', 'advantage plans', 'at&t group plan', 'chronic plan', 'chronic plans', 'current plan', 'dual', 'dual complete',
            'dual complete coverage plan', 'dual complete plan', 'dual complete plans', 'dual medicaid medicare program', 'dual membership plan', 'dual plan', 'dual plans', 
            'hmo', 'hmo plan', 'hmo plans', 'medicaid', 'medicaid plan', 'medicaid plans', 'medicare', 'medicare advantage', 'medicare advantage plan', 'medicare advantage plans',
            'medicare advantage ppo plan', 'medicare insurance plan', 'medicare medical aarp plan', 'medicare plan', 'medicare plans', 'medicare supplement plan', 'medicare supplemental insurance',
            'n', 'na', 'nan', 'new plan', 'ppo', 'ppo plan', 'ppo plans', 'snp', 'uhc', 'uhc aarp', 'uhc aarp medicare', 'uhc aarp medicare plan', 'united', 'united care', 'united health',
            'united health plan', 'united health plans', 'united healthcare', 'united healthcare aarp', 'united healthcare advantage plan', 'united healthcare advantage ppo',
            'united healthcare dual', 'united healthcare dual complete', 'united healthcare dual complete plan', 'united healthcare dual plan', 'united healthcare dual plans', 
            'united healthcare medicare', 'united healthcare medicare advantage', 'united healthcare medicare supplement', 'united healthcare plan', 'united healthcare plans',
            'united healthcare prescription drug plan', 'united healthcare u card', 'united plan', 'united plans', 'description for others', 'description for other', 
            'united health advantage medicare advantage', 'other', 'others'
        ]

        unuseful_subtopics = [x.strip() for x in unuseful_subtopics]

        words_to_replace = "uhc, uhc aarp, uhc aarp medicare, uhc aarp medicare plan, aarp medicare, aarp medicare advantage, aarp medicare advantage plan, new plan, current plan, united healthcare, dual plan, dual plans, medicare advantage, medicare advantage plan, medicare advantage plans, medicare, medicaid, medicaid plan, medicare plan, medicaid plans, medicare plans, aarp, hmo, ppo, aarp plan, hmo plan, ppo plan, aarp plans, hmo plans, ppo plans, dual complete, dual complete plan, dual complete plans, dual, chronic plan, chronic plans, united plan, united plans, united healthcare plan, united healthcare plans, united health plan, united health plans, united healthcare medicare supplement"
        words_to_replace_2 = "NA, N, A, United Healthcare AARP, AARP United Healthcare, AARP Healthcare plan, AARP Supplemental health insurance plans, United Healthcare, United Healthcare Plan, United Healthcare Plans, AARP Medicare Advantage, AARP United Healthcare Medicare Advantage, AARP Medicare Advantage from United Healthcare, Medicare Advantage, AAP Medicare Advantage, United Healthcare dual, AARP Medicare Advantage Essentials from United Healthcare, United Care, Medicaid, SNP, Advantage Plan, AARP Medicare Advantage from UHC, United healthcare U card, United Healthcare dual complete, United Healthcare Prescription drug plan, chronic plan, medicare advantage ppo plan, at&t group plan, united health, united healthcare dual plan, united healthcare dual plans, aarp supplement, aarp supplements, aarp supplement and advantage plans, aarp medicare rx from united healthcare"

        words_to_replace = words_to_replace + ", " + words_to_replace_2
        words_to_replace = words_to_replace.lower()
        words_to_replace = words_to_replace.split(",")
        words_to_replace = [x.strip() for x in words_to_replace]

        words_to_replace = words_to_replace + unuseful_subtopics
        unuseful_subtopics = list(set(words_to_replace))

        unique_subtopics = set()
        for subtopic in subtopics:
            subtopic = subtopic.lower().replace("'", "")
            expanded_subtopics = [str(s).replace("[", "").replace("]", "").replace("'", "").strip() for s in subtopic.split(",")]
            for s in expanded_subtopics:
                if s not in unuseful_subtopics:
                    unique_subtopics.add(s)
        unique_subtopics = list(unique_subtopics)

        return unique_subtopics