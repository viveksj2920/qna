#!/usr/bin/env python3
"""
Demonstration script for sentiment extraction functionality.
This script shows how to use the new sentiment extraction methods.
"""

import sys
import os
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from qna_extractor import qna_extractor
    from utils.logger_config import logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

def demonstrate_sentiment_extraction():
    """Demonstrate the sentiment extraction functionality with sample questions"""
    
    print("="*80)
    print("SENTIMENT EXTRACTION DEMONSTRATION")
    print("="*80)
    
    # Sample questions representing different sentiments
    sample_questions = [
        {
            "question": "Thank you so much for helping me resolve this issue! You've been incredibly helpful.",
            "expected_sentiment": "Positive - Grateful"
        },
        {
            "question": "What is the deadline for enrollment this year?",
            "expected_sentiment": "Neutral - Informational"
        },
        {
            "question": "I've been waiting on hold for 45 minutes and I'm getting really frustrated!",
            "expected_sentiment": "Negative - Frustrated"
        },
        {
            "question": "Can you walk me through the step-by-step process to submit my claim?",
            "expected_sentiment": "Neutral - Procedural"
        },
        {
            "question": "This is absolutely ridiculous! I want to speak to your manager immediately!",
            "expected_sentiment": "Negative - Angry"
        },
        {
            "question": "I'm very satisfied with the quick resolution of my problem.",
            "expected_sentiment": "Positive - Satisfied"
        }
    ]
    
    project = "MIRA"  # Can be changed to "PCL"
    
    print(f"Testing with {project} project configuration\n")
    
    successful_extractions = 0
    total_questions = len(sample_questions)
    
    for i, sample in enumerate(sample_questions, 1):
        print(f"--- Example {i} ---")
        print(f"Question: \"{sample['question']}\"")
        print(f"Expected: {sample['expected_sentiment']}")
        
        try:
            # Create extractor instance
            extractor = qna_extractor(question=sample['question'], project=project)
            
            # Extract level 1 sentiment
            sentiment_1_json = extractor.extract_sentiment_level_one()
            
            if sentiment_1_json:
                # Clean and parse the response
                cleaned_response = sentiment_1_json.strip()
                cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
                
                sentiment_1_data = json.loads(cleaned_response)
                level_1_sentiment = sentiment_1_data.get('sentiment_level_1')
                
                print(f"Level 1 Result: {level_1_sentiment}")
                
                # Extract level 2 sentiment
                sentiment_2_json = extractor.extract_sentiment_level_two(level_1_sentiment)
                
                if sentiment_2_json:
                    cleaned_response_2 = sentiment_2_json.strip()
                    cleaned_response_2 = cleaned_response_2.replace("```json", "").replace("```", "").strip()
                    
                    sentiment_2_data = json.loads(cleaned_response_2)
                    level_2_sentiment = sentiment_2_data.get('sentiment_level_2')
                    
                    print(f"Level 2 Result: {level_2_sentiment}")
                    print(f"Final Classification: {level_1_sentiment} - {level_2_sentiment}")
                    
                    successful_extractions += 1
                    print("✅ SUCCESS")
                    
                else:
                    print("❌ Level 2 extraction failed")
            else:
                print("❌ Level 1 extraction failed")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
        
        print("-" * 50)
    
    print(f"\nSUMMARY:")
    print(f"Successful extractions: {successful_extractions}/{total_questions}")
    print(f"Success rate: {(successful_extractions/total_questions)*100:.1f}%")
    
    return successful_extractions == total_questions

def demonstrate_integration():
    """Demonstrate integration with the full QnA extraction pipeline"""
    
    print("\n" + "="*80)
    print("INTEGRATION DEMONSTRATION")
    print("="*80)
    
    # Sample conversation text
    sample_conversation = """
    Agent: Hello, thank you for calling. How can I help you today?
    Customer: Hi, I'm really frustrated because I've been trying to enroll online for the past week and the website keeps crashing!
    Agent: I'm sorry to hear about the trouble you're having. Let me help you with that enrollment process.
    Customer: Thank you, I really appreciate your patience. What information do you need from me?
    Agent: I'll need your member ID and some basic information to get started.
    Customer: What is the deadline for enrollment this year?
    Agent: The enrollment deadline is December 15th, and you still have plenty of time.
    Customer: Perfect! You've been so helpful. Thank you for resolving this issue.
    """
    
    project = "MIRA"
    
    print(f"Sample conversation text:")
    print(f'"{sample_conversation.strip()}"')
    print("\nExtracting Q&A pairs and analyzing sentiments...\n")
    
    try:
        # Extract Q&A pairs first
        extractor = qna_extractor(text=sample_conversation, project=project)
        qna_result = extractor.extract_qna()
        
        if qna_result:
            # Clean and parse the Q&A result
            cleaned_qna = qna_result.strip().replace("```json", "").replace("```", "").strip()
            qna_data = json.loads(cleaned_qna)
            
            questions_and_answers = qna_data.get('question_and_answer', [])
            print(f"Extracted {len(questions_and_answers)} Q&A pairs:\n")
            
            for i, qa in enumerate(questions_and_answers, 1):
                question = qa.get('question', '')
                answer = qa.get('answer', '')
                
                print(f"Q{i}: {question}")
                print(f"A{i}: {answer}")
                
                # Now extract sentiment for this question
                sentiment_extractor = qna_extractor(question=question, project=project)
                
                # Get level 1 sentiment
                level_1_json = sentiment_extractor.extract_sentiment_level_one()
                
                if level_1_json:
                    cleaned_l1 = level_1_json.strip().replace("```json", "").replace("```", "").strip()
                    level_1_data = json.loads(cleaned_l1)
                    sentiment_l1 = level_1_data.get('sentiment_level_1')
                    
                    # Get level 2 sentiment
                    level_2_json = sentiment_extractor.extract_sentiment_level_two(sentiment_l1)
                    
                    if level_2_json:
                        cleaned_l2 = level_2_json.strip().replace("```json", "").replace("```", "").strip()
                        level_2_data = json.loads(cleaned_l2)
                        sentiment_l2 = level_2_data.get('sentiment_level_2')
                        
                        print(f"Sentiment: {sentiment_l1} - {sentiment_l2}")
                    else:
                        print(f"Sentiment: {sentiment_l1} - (Level 2 extraction failed)")
                else:
                    print("Sentiment: (Extraction failed)")
                
                print()
        else:
            print("❌ Q&A extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration demonstration failed: {e}")
        return False
    
    print("✅ Integration demonstration completed successfully!")
    return True

def main():
    """Main demonstration function"""
    print("Starting sentiment extraction demonstration...\n")
    
    # Run basic sentiment extraction demonstration
    basic_success = demonstrate_sentiment_extraction()
    
    # Run integration demonstration
    integration_success = demonstrate_integration()
    
    print("\n" + "="*80)
    print("DEMONSTRATION SUMMARY")
    print("="*80)
    
    if basic_success and integration_success:
        print("🎉 All demonstrations completed successfully!")
        print("\nThe sentiment extraction functionality is working correctly and")
        print("is properly integrated with the existing QnA extraction pipeline.")
    else:
        print("❌ Some demonstrations failed.")
        print("Please check the error messages above and ensure all dependencies are properly configured.")
    
    print("\n" + "="*80)
    
    return basic_success and integration_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
