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

def prompt_sub_topic_format(project, sub_topic_prompt):

    # Initialize a string variable to hold the formatted output  
    formatted_output = ""

    if project == "MIRA":
        count = 1
        for keys, value in sub_topic_prompt.items():
            formatted_output += f"{count}. {keys}: {value['description']}\n\n"  
            formatted_output += "examples:\n\n"
            for example in value["examples"]:
                formatted_output += f"question: {example}\n"
            count += 1

    if project == "PCL":

        for sub_topic in sub_topic_prompt:

            topic_prompt = sub_topic_prompt[sub_topic]
            topic_description = topic_prompt["description"]  

            # Format the output for this topic  
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
    Generates the MIRA extraction prompt for metadata extraction from transcripts.
    """
    prompt = textwrap.dedent(f"""

            You are an intelligent knowledge building assistant. 
            You task is to identify all applicable sub-topics for a given question asked by customer in transcript.

            ### Context:

            - The question is asked are conversations between agents and customers.
            - Portions of the transcript may have personally identifiable information (PII) redacted with asterisks (e.g., ****). **This redaction does NOT affect the usability of the transcript, and should not be called out at any point.**

            
            ## Classify given question into all applicable sub-topics using below sub-topic descriptions for the given topic.
            
            ## Topic: {topic}

            ## Sub-Topics and Descriptions
                             
            {sub_topic_descriptions} 

            
            ### Instructions:

            1. Select every sub-topic that is relevant to the given question and consistent with the provided topic.
            2. A question may belong to multiple sub-topics if more than one description clearly applies.
            3. If none of the provided sub-topics fit the question, return an empty list for sub_topic.
            4. Format your response like the following JSON:

            ```

            {{ "question": [string], "topic": [string], "sub_topic": ["sub_topic_1", "sub_topic_2"] }}

            ```

            Make sure that your responses are always valid JSON and only the required JSON response.

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

def subtopic_grouping_prompt(subtopic_list):
    """
    Generates the MIRA extraction prompt for metadata extraction from transcripts.
    """
    prompt = textwrap.dedent(f"""
        You are given a list of topics. Your task is to group them based on functional and contextual semantic similarity.

        ### Grouping Rules:
        - Each group MUST contain at least 2 topics.
        - Topics in a group must be interchangeable or strongly related in meaning and usage.
        - Do NOT group topics that:
            - Differ in purpose or domain (e.g., 'medical card' vs. 'credit card')
            - Are loosely related or only tangentially connected
            - Are proper nouns, including:
                - Specific names of people (e.g., Dr. Smith)
                - Plan names (e.g., United Healthcare Dual Complete)
                - Drug or medication names (e.g., eliquis, lorazepam, gabapentin)
                - Geographic locations (e.g., Los Angeles, Chicago)
                - Organization or brand names (e.g., Walgreens, CVS, AARP)

        ### Exceptions:
        - You may group variations of the same proper noun if they clearly refer to the same entity.
            - Example: "United Healthcare", "United Health", "United Healthcare Medicare" may be grouped.
            - Example: "zofran", "zofran 15 mg", "ondansetron" may be grouped as they refer to the same medication.
        - Do NOT group different entities even if they belong to the same category.
            - Example: "Walgreens" and "CVS" must NOT be grouped.
            - Example: "Los Angeles" and "Chicago" must NOT be grouped.
        - Do not group different medications, even if they treat similar conditions or belong to the same drug class. Only group medication variants if they clearly refer to the same drug.
            - Example: "zofran" and "sumatriptan 15 mg" must NOT be grouped.

        ### Output Format:
        Return the result in valid JSON format:
        {{
            "categories": [["topic1", "topic2"], ["topic3", "topic4", "topic5"], ...]
        }}

        ### Examples of Correct Groupings:
        Example 1:
        Topics: "united health", "united healthcare", "united healthcare medicare", "united healthcare dual complete"
        Output: {{"categories": [["united health", "united healthcare", "united healthcare medicare"]]}}

        Example 2:
        Topics: "zofran", "zofran 15 mg", "ondansetron", "gabapentin", "lorazepam"
        Output: {{"categories": [["zofran", "zofran 15 mg", "ondansetron"]]}}

        Example 3:
        Topics: "walgreens", "cvs", "los angeles", "chicago"
        Output: {{"categories": []}}  # No groups formed as topics are distinct proper nouns

        Example 4:
        Topics: "replacement card", "new card replacement", "medical card", "card", "new card"
        Output: {{"categories": [["medical card", "card", "new card"], ["replacement card", "new card replacement"]]}}

        Topics: {subtopic_list}
        """
    )
    return prompt

def subtopic_labeling_prompt(subtopic_list):
    """
    Generates the MIRA extraction prompt for metadata extraction from transcripts.
    """
    prompt = textwrap.dedent(f"""
        Your task is to generate a topic from the given set of related topics that fully represents all the information in all the given topics. 
        If one topic is broader than the others, select that one.
        Ensure that the newly generated topic is accurate and complete based only on the given topics. Restrict the new topic to two words or less and return only the new topic without any additional text or punctuation.

        Below are some examples of good topics generated from a set of related topics:
        Example 1:
        Given Topics: "regular card", "current humana card", "dual complete card", "extra benefits card", "new card replacement", "replacement card", "physical id card", "replacement cards", "new card"
        Generated Topic: "card"

        Example 2:
        Given Topics: "address update", "account update"
        Generated Topic: "account update"

        Example 3:
        Given Topics: "plan start date", "plan start", "plan effective date"
        Generated Topic: "plan effective date"

        Example 4:
        Given Topics: "plan status", "coverage status", "activation status", "insurance status", "policy status", "membership status", "enrollment status", "reinstatement status", "approval status"
        Generated Topic: "status"

        Topics: {subtopic_list}
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
    sub_topic_descriptions = "Status: Questions about the current status of a claim.\nFiling: Questions about how to file a claim."
    sub_topic_prompt = sub_topic_extraction_prompt(question, topic, sub_topic_descriptions)
    assert "Classify given question into a sub-topic using below sub-topic descriptions" in sub_topic_prompt, "sub_topic_extraction_prompt content mismatch"
    assert "Status: Questions about the current status of a claim" in sub_topic_prompt, "sub_topic_extraction_prompt sub-topic descriptions mismatch"

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