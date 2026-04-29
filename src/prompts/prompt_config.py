import textwrap


def qna_extraction_prompt(chunk):
    """
    Generates the MIRA extraction prompt for metadata extraction from transcripts.
    """
    prompt = textwrap.dedent(f"""

            You are an intelligent knowledge building assistant. 
            You task is to *EXTRACT* question, answer pairs from the given call transcript.

            ### Context:

            - The transcripts are conversations between agents and customers.
            - Portions of the transcript may have personally identifiable information (PII) redacted with asterisks (e.g., ****). **This redaction does NOT affect the usability of the transcript, and should not be called out at any point.**

            
            ### Instructions:

            1. Thoroughly review each transcript.
            2. If a question has multiple answers or clarifications, combine them into a single comprehensive answer and keep it to no more than 2 sentences.
            3. If the same or a very identically similar question is repeated , dont repeat the question, just consider the answer as part of the comprehensive answer.
            4. If a question is asked in a different way, dont treat it as a new question.
            5. If a question is a follow up to a previous question, ignore that question but summarize the answer to the original question and add it as an answer to the original question.
            6. Limit the number of questions to 10 and include the most important questions first.
            7. Do not include small talk, greetings or closing remarks unless they contain specific insurance related questions.
            8. Extract the required Q&A pairs.
            9. Each question should have one and only one corresponding answer.
            10. Focus only on the questions asked by the consumers and not the questions or follow up questions asked by the agent.
            11. Each answer should correspond to the appropriate question.
            12. Format your response like the following JSON:

            ```

            {{ 
            "question_and_answer":
                [
                    {{"question":"question asked by customer in transcript", "answer":"answer from agent for respective question"}},
                    {{"question":"question asked by customer in transcript", "answer":"answer from agent for respective question"}}
                ] 
            }}

            ```

            Make sure that your responses are always valid JSON and only the required JSON response.

            Call Transcript: {chunk} """
 )
    return prompt

def prompt_topic_format(project, json_data):
    """
    Converts JSON content to a formatted text string, handling different JSON structures.
    
    Args:
        project, json_data
        
    Returns:
        str: Formatted text string
    """
    
    # Initialize the output string
    output = ""
    
    if project == "MIRA":
        for topic, description in json_data.items():
            output += f"Topic: {topic}\n"
            output += f"Description: {description}\n\n"
        
    elif project == "PCL":
        for topic, content in json_data.items():
            output += f"Topic: {topic}\n"
            output += f"Description: {content['description']}\n"
            
            if content['details']:
                output += "Details:\n"
                for detail in content['details']:
                    output += f"-{detail}\n"
            
            output += "\n"  # Add a blank line between topics
    
    else:
        # Generic handling for other JSON formats
        output = "Unsupported JSON format\n"
    
    return output

def topic_extraction_prompt(question, topic_descriptions):
    """
    Generates the MIRA extraction prompt for metadata extraction from transcripts.
    """
    prompt = textwrap.dedent(f"""

            You are an intelligent knowledge building assistant. 
            You task is to assign only one topic for a given question asked by customer in transcript.

            ### Context:

            - The question is asked are conversations between agents and customers.
            - Portions of the transcript may have personally identifiable information (PII) redacted with asterisks (e.g., ****). **This redaction does NOT affect the usability of the transcript, and should not be called out at any point.**

            
            ## Classify given question into a topic using below topic descriptions: 
            
            ## Topics and Descriptions
                             
            {topic_descriptions} 

            
            ### Instructions:

            1. Each topic should be relevant to the given question.
            2. Classify each question into only one best possible corresponding topic based on the detailed descriptions provided.
            3. Format your response like the following JSON:

            ```

            {{ "question": [string], "topic": [string] }}

            ```

            Make sure that your responses are always valid JSON and only the required JSON response.

            Question: {question} """
 )
    return prompt

def prompt_sub_topic_format(project, sub_topic_config):
    """
    Formats subtopic config for inclusion in the LLM prompt.

    For MIRA: expects new format {"subtopics": [...], "open_ended": [...]}
    For PCL: expects old format {"subtopic_name": {"description": ..., "examples": [...]}}
    """
    formatted_output = ""

    if project == "MIRA":
        subtopics = sub_topic_config.get("subtopics", [])
        open_ended = sub_topic_config.get("open_ended", [])

        formatted_output += "PREDEFINED SUBTOPICS (you MUST select from this list only):\n"
        for i, name in enumerate(subtopics, 1):
            formatted_output += f"  {i}. {name}\n"

        if open_ended:
            formatted_output += "\nENTITY EXTRACTION (extract these named entities if mentioned in the question):\n"
            for entry in open_ended:
                formatted_output += f"  - {entry['type']}: {entry['instruction']}\n"

    elif project == "PCL":
        for sub_topic in sub_topic_config:
            topic_prompt = sub_topic_config[sub_topic]
            topic_description = topic_prompt["description"]

            formatted_output += f"Sub topic: {sub_topic}\n"
            formatted_output += f"Description: {topic_description}\n"
            formatted_output += "\nreturn the output as a list of entities.\n"

            if "examples" in topic_prompt:
                examples_list = topic_prompt["examples"]
                formatted_output += "\nexample questions:\n\n"
                for question in examples_list:
                    formatted_output += f"question: {question}\n"

            if "explanation" in topic_prompt:
                explanation = topic_prompt["explanation"]
                formatted_output += f"\nexplanation: {explanation}\n\n"

            formatted_output += "----------------------------------------\n\n"

    return formatted_output

def sub_topic_extraction_prompt(question, topic, sub_topic_descriptions):
    """
    Generates the subtopic extraction prompt. Instructs the LLM to select from
    predefined subtopics and extract named entities when applicable.
    """
    prompt = textwrap.dedent(f"""
            You are an intelligent knowledge building assistant.
            Your task is to classify a customer question into applicable sub-topics
            AND extract specific named entities when applicable.

            ### Context:

            - The question comes from a conversation between an agent and a customer
              at a Medicare/healthcare company.
            - PII may be redacted with asterisks (****). This does not affect analysis.

            ## Topic: {topic}

            ## Available Sub-Topics and Entity Extraction Rules:

            {sub_topic_descriptions}

            ### Instructions:

            1. Select ALL applicable sub-topics from the PREDEFINED SUBTOPICS list above.
            2. You MUST ONLY select sub-topic names that appear exactly in the predefined list. Do NOT invent, rephrase, or modify sub-topic names.
            3. A question may belong to multiple sub-topics if more than one clearly applies.
            4. Be STRICT: only select a sub-topic if the question is genuinely about that specific sub-topic. Do NOT select a sub-topic just because it is loosely related.
            5. If none of the predefined sub-topics clearly fit the question, return an empty list [] for sub_topic. An empty list is preferred over a poor match.
            6. For ENTITY EXTRACTION fields (if listed above), extract the named entity from the question following the specific instruction for each type.
               - If no entity of that type is mentioned in the question, set its value to null.
            6. For drug names: extract ONLY the base/generic medication name. No dosages, frequencies, forms, brand qualifiers, or parentheses. Example: "zolpidem 10mg tablet" → extract "zolpidem".
            7. For plan names: extract the plan name as mentioned by the customer. Example: "AARP Medicare Advantage Choice Plan 1" → extract "AARP Medicare Advantage Choice Plan 1".
            8. For pharmacy/hospital/facility names: extract just the name as mentioned.

            ### Response Format (valid JSON only):

            ```
            {{
                "question": [string],
                "topic": [string],
                "sub_topic": ["sub_topic_1", "sub_topic_2"],
                "entities": {{}}
            }}
            ```

            The "entities" object should only include keys for entity types listed in the ENTITY EXTRACTION rules above. If no entity extraction rules are listed for this topic, return an empty object for "entities".

            Make sure that your response is always valid JSON and contains only the JSON response.

            Question: {question} """
 )
    return prompt

def questions_prompt_format(questions):
    """
    Converts a list of questions into a formatted text string.
    
    Args:
        questions (list): List of question strings.
    
    Returns:
        str: Formatted text string
    """
    
    # Join the questions into a single string with each question on a new line
    formatted_questions = "\n    ".join(f"- {question}" for question in questions)
    
    return formatted_questions

def is_useful_question_extraction_prompt(question, questions_json):
    """
    Generates the MIRA extraction prompt for metadata extraction from transcripts.
    """
    un_useful_questions = questions_prompt_format(questions_json["UnusefulQuestions"])
    useful_questions = questions_prompt_format(questions_json["UsefulQuestions"])
    prompt = textwrap.dedent(f"""

    You are an AI language model tasked with analyzing questions extracted from call transcripts between agents and individuals at a healthcare company, specifically related to Medicare and Retirement.
    Your objective is to determine whether each question is useful or not useful for understanding customer pain points or needs. 
    You are given a sample list of common questions that are considered unuseful. More detailed instructions are below.

    ### Instructions:
    1. Review the question carefully and pay close attention to the specifics of the question.
    2. If the question matches or is semantically similar to any question in the unuseful questions list, return false.
    3. If the question does not match any unuseful question, determine whether it is useful based on the context above.
    4. Return true if the question is useful, otherwise return false.

    ### List of Unuseful Questions:
    {un_useful_questions}

    ### Sample Useful Questions:
    {useful_questions}

    ### Input:
    - Question: {question}

    ### Formatting the Output:
    - Return a string (true or false) where true indicates the question is useful and false indicates it is not useful. No explanation is needed.
    
    """
 )
    return prompt

def test_prompts():
    """
    Unit test for prompt generation functions.
    """
    # Test qna_extraction_prompt
    chunk = "Customer: What is the status of my claim? Agent: Your claim is being processed."
    qna_prompt = qna_extraction_prompt(chunk)

    assert "Call Transcript:" in qna_prompt, "qna_extraction_prompt failed"
    assert "You task is to *EXTRACT* question, answer pairs" in qna_prompt, "qna_extraction_prompt content mismatch"

    # Test topic_extraction_prompt
    question = "What is the status of my claim?"
    topic_descriptions = "Claims: Questions related to claim status, filing, or processing.\nBilling: Questions related to billing or payments."
    topic_prompt = topic_extraction_prompt(question, topic_descriptions)
    assert "Classify given question into a topic using below topic descriptions" in topic_prompt, "topic_extraction_prompt content mismatch"
    assert "Claims: Questions related to claim status" in topic_prompt, "topic_extraction_prompt topic descriptions mismatch"

    # Test sub_topic_extraction_prompt
    topic = "Claims"
    sub_topic_descriptions = "PREDEFINED SUBTOPICS (you MUST select from this list only):\n  1. claim_status\n  2. claim_filing\n"
    sub_topic_prompt = sub_topic_extraction_prompt(question, topic, sub_topic_descriptions)
    assert "classify a customer question into applicable sub-topics" in sub_topic_prompt, "sub_topic_extraction_prompt content mismatch"
    assert "claim_status" in sub_topic_prompt, "sub_topic_extraction_prompt sub-topic descriptions mismatch"

    # Test is_useful_question_extraction_prompt
    useful_question = "Will I receive a new card for the plan?"
    questions_json = [
        "How are you?",
        "Can you hear me?",
        "Is this call being recorded?"
    ]
    is_useful_prompt = is_useful_question_extraction_prompt(useful_question, questions_json)
    assert "determine whether each question is useful or not useful" in is_useful_prompt, "is_useful_question_extraction_prompt content mismatch"
    assert "List of Unuseful Questions:" in is_useful_prompt, "is_useful_question_extraction_prompt unuseful questions list missing"
    assert useful_question in is_useful_prompt, "is_useful_question_extraction_prompt input question missing"

    print("All prompt tests passed!")

def sentiment_level_1_prompt(question, level_1_categories):
    """
    Generate prompt for level 1 sentiment extraction
    """
    categories_text = "\n".join([f"- {category}: {description}" for category, description in level_1_categories.items()])
    
    prompt = textwrap.dedent(f"""
    You are an intelligent sentiment analysis assistant. Your task is to analyze the sentiment of customer questions and classify them into high-level sentiment categories.

    ### Instructions:

    1. Carefully analyze the emotional tone and context of the given question.
    2. Consider the customer's intent and emotional state expressed in the question.
    3. Classify the question into one of the available sentiment categories.
    4. Choose the most appropriate category based on the overall emotional tone.
    5. Respond with valid JSON format only.

    ### Available Sentiment Categories:
    {categories_text}

    ### Question to Analyze:
    "{question}"

    ### Response Format:
    {{"sentiment_level_1": "category_name"}}

    Provide only the JSON response with the most appropriate sentiment category.
    """)
    return prompt

def sentiment_level_2_prompt(question, sentiment_level_1, level_2_categories):
    """
    Generate prompt for level 2 sentiment extraction
    """
    categories_text = "\n".join([f"- {subcategory}: {description}" for subcategory, description in level_2_categories.items()])
    
    prompt = textwrap.dedent(f"""
    You are an intelligent sentiment analysis assistant. The customer question has been classified as "{sentiment_level_1}" sentiment. Now classify it into a more specific subcategory.

    ### Instructions:

    1. Analyze the specific emotional nuance within the "{sentiment_level_1}" sentiment category.
    2. Consider the customer's specific emotional state and intent.
    3. Choose the most appropriate subcategory that best describes the specific sentiment.
    4. If none of the specific subcategories fit perfectly, choose "Other".
    5. Respond with valid JSON format only.

    ### Available Subcategories for {sentiment_level_1}:
    {categories_text}

    ### Question to Analyze:
    "{question}"

    ### Response Format:
    {{"sentiment_level_2": "subcategory_name"}}

    Provide only the JSON response with the most appropriate sentiment subcategory.
    """)
    return prompt

if __name__ == "__main__":
    test_prompts()