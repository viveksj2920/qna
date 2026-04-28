#!/usr/bin/env python3
"""
Comprehensive testing script for sentiment extraction functionality.
This script tests both sentiment level 1 and sentiment level 2 extraction.
"""

import unittest
import json
import sys
import os
import pandas as pd
from io import StringIO
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from qna_extractor import qna_extractor
    from utils.helper import load_project_config
    from utils.logger_config import logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the correct directory")
    sys.exit(1)

class TestSentimentExtraction(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.project = "MIRA"  # Can be changed to "PCL" for testing PCL configuration
        
        # Test questions with known sentiments - covering various scenarios
        self.test_questions = {
            # Positive sentiment tests
            "positive_satisfied": "Thank you so much for resolving my issue quickly! Everything is working perfectly now.",
            "positive_grateful": "I really appreciate your help with my enrollment. You've been extremely helpful.",
            "positive_other": "This service is amazing! I'm so happy with the results.",
            
            # Neutral sentiment tests  
            "neutral_informational": "What is the deadline for enrollment this year?",
            "neutral_procedural": "Can you walk me through the steps to submit my claim online?",
            "neutral_other": "I need to update my address information in the system.",
            
            # Negative sentiment tests
            "negative_frustrated": "I've been waiting on hold for 30 minutes, this is really frustrating!",
            "negative_angry": "This is completely unacceptable! I demand to speak to a manager right now!",
            "negative_other": "I'm disappointed with the service quality lately."
        }
        
        self.expected_results = {
            "positive_satisfied": {"level_1": "Positive", "level_2": "Satisfied"},
            "positive_grateful": {"level_1": "Positive", "level_2": "Grateful"},
            "positive_other": {"level_1": "Positive", "level_2": "Other"},
            "neutral_informational": {"level_1": "Neutral", "level_2": "Informational"},
            "neutral_procedural": {"level_1": "Neutral", "level_2": "Procedural"},
            "neutral_other": {"level_1": "Neutral", "level_2": "Other"},
            "negative_frustrated": {"level_1": "Negative", "level_2": "Frustrated"},
            "negative_angry": {"level_1": "Negative", "level_2": "Angry"},
            "negative_other": {"level_1": "Negative", "level_2": "Other"}
        }

    def test_sentiment_configuration_loading(self):
        """Test that sentiment configuration loads correctly"""
        print("\n=== Testing Configuration Loading ===")
        
        config = load_project_config(self.project, "sentiment_extraction")
        
        self.assertIsNotNone(config, "Sentiment config should not be None")
        self.assertIn("level_1", config, "Config should contain level_1")
        self.assertIn("level_2", config, "Config should contain level_2")
        
        # Check level 1 categories
        level_1 = config["level_1"]
        expected_level_1_categories = ["Positive", "Neutral", "Negative"]
        for category in expected_level_1_categories:
            self.assertIn(category, level_1, f"Level 1 should contain {category}")
        
        # Check level 2 categories
        level_2 = config["level_2"]
        for sentiment in ["positive", "neutral", "negative"]:
            self.assertIn(sentiment, level_2, f"Level 2 should contain {sentiment}")
        
        print("Configuration loaded successfully")
        print(f"Level 1 categories: {list(level_1.keys())}")
        print(f"Level 2 structure: {list(level_2.keys())}")

    def test_sentiment_level_1_extraction(self):
        """Test level 1 sentiment extraction"""
        print("\n=== Testing Sentiment Level 1 Extraction ===")
        
        passed_tests = 0
        total_tests = len(self.test_questions)
        
        for test_case, question in self.test_questions.items():
            with self.subTest(test_case=test_case):
                try:
                    extractor = qna_extractor(question=question, project=self.project)
                    result_json = extractor.extract_sentiment_level_one()
                    
                    self.assertIsNotNone(result_json, f"Level 1 extraction failed for {test_case}")
                    
                    # Clean and parse JSON result
                    cleaned_result = result_json.strip().replace("```json", "").replace("```", "").strip()
                    result = json.loads(cleaned_result)
                    sentiment_level_1 = result.get('sentiment_level_1')
                    
                    expected_level_1 = self.expected_results[test_case]["level_1"]
                    
                    print(f"Question: {question}")
                    print(f"Expected: {expected_level_1}, Got: {sentiment_level_1}")
                    
                    if sentiment_level_1 == expected_level_1:
                        passed_tests += 1
                        print("✅ PASS")
                    else:
                        print("❌ FAIL")
                        
                except Exception as e:
                    print(f"❌ ERROR in {test_case}: {e}")
                    
                print("-" * 50)
        
        accuracy = (passed_tests / total_tests) * 100
        print(f"\nLevel 1 Accuracy: {passed_tests}/{total_tests} ({accuracy:.1f}%)")
        
        # We'll be lenient and require at least 70% accuracy for tests to pass
        self.assertGreaterEqual(accuracy, 70, f"Level 1 sentiment accuracy ({accuracy:.1f}%) is below 70%")

    def test_sentiment_level_2_extraction(self):
        """Test level 2 sentiment extraction"""
        print("\n=== Testing Sentiment Level 2 Extraction ===")
        
        passed_tests = 0
        total_tests = len(self.test_questions)
        
        for test_case, question in self.test_questions.items():
            with self.subTest(test_case=test_case):
                try:
                    extractor = qna_extractor(question=question, project=self.project)
                    
                    # First get level 1
                    level_1_json = extractor.extract_sentiment_level_one()
                    cleaned_level_1 = level_1_json.strip().replace("```json", "").replace("```", "").strip()
                    level_1_data = json.loads(cleaned_level_1)
                    sentiment_level_1 = level_1_data.get('sentiment_level_1')
                    
                    # Then get level 2
                    level_2_json = extractor.extract_sentiment_level_two(sentiment_level_1)
                    
                    self.assertIsNotNone(level_2_json, f"Level 2 extraction failed for {test_case}")
                    
                    # Clean and parse JSON result
                    cleaned_level_2 = level_2_json.strip().replace("```json", "").replace("```", "").strip()
                    result = json.loads(cleaned_level_2)
                    sentiment_level_2 = result.get('sentiment_level_2')
                    
                    expected_level_2 = self.expected_results[test_case]["level_2"]
                    
                    print(f"Question: {question}")
                    print(f"Level 1: {sentiment_level_1}")
                    print(f"Expected Level 2: {expected_level_2}, Got: {sentiment_level_2}")
                    
                    if sentiment_level_2 == expected_level_2:
                        passed_tests += 1
                        print("✅ PASS")
                    else:
                        print("❌ FAIL")
                        
                except Exception as e:
                    print(f"❌ ERROR in {test_case}: {e}")
                    
                print("-" * 50)
        
        accuracy = (passed_tests / total_tests) * 100
        print(f"\nLevel 2 Accuracy: {passed_tests}/{total_tests} ({accuracy:.1f}%)")
        
        # We'll be lenient and require at least 60% accuracy for level 2 tests to pass
        self.assertGreaterEqual(accuracy, 60, f"Level 2 sentiment accuracy ({accuracy:.1f}%) is below 60%")

    def test_integration_with_qna_pipeline(self):
        """Test sentiment extraction integration with the main QnA pipeline"""
        print("\n=== Testing Integration with QnA Pipeline ===")
        
        # Create a sample row similar to what would be processed
        sample_text = """
        Agent: Hello, how can I help you today?
        Customer: I've been trying to enroll for weeks and it's really frustrating! Why is this so difficult?
        Agent: I understand your frustration. Let me help you with the enrollment process.
        Customer: Thank you, I really appreciate your patience with me.
        """
        
        # Test the QnA extraction first
        extractor = qna_extractor(text=sample_text, project=self.project)
        qna_result = extractor.extract_qna()
        
        if qna_result:
            try:
                # Clean the response
                cleaned_response = qna_result.strip().replace("```json", "").replace("```", "").strip()
                qna_data = json.loads(cleaned_response)
                
                questions = [qa['question'] for qa in qna_data.get('question_and_answer', [])]
                print(f"Extracted {len(questions)} questions from sample text")
                
                # Test sentiment extraction on the first question if available
                if questions:
                    test_question = questions[0]
                    sentiment_extractor = qna_extractor(question=test_question, project=self.project)
                    
                    sentiment_1 = sentiment_extractor.extract_sentiment_level_one()
                    self.assertIsNotNone(sentiment_1, "Sentiment level 1 should be extracted")
                    
                    # Parse level 1 result
                    cleaned_sentiment_1 = sentiment_1.strip().replace("```json", "").replace("```", "").strip()
                    sentiment_1_data = json.loads(cleaned_sentiment_1)
                    level_1_value = sentiment_1_data.get('sentiment_level_1')
                    
                    sentiment_2 = sentiment_extractor.extract_sentiment_level_two(level_1_value)
                    self.assertIsNotNone(sentiment_2, "Sentiment level 2 should be extracted")
                    
                    print(f"Successfully extracted sentiments for question: '{test_question}'")
                    print(f"Level 1: {level_1_value}")
                    
                    cleaned_sentiment_2 = sentiment_2.strip().replace("```json", "").replace("```", "").strip()
                    sentiment_2_data = json.loads(cleaned_sentiment_2)
                    level_2_value = sentiment_2_data.get('sentiment_level_2')
                    print(f"Level 2: {level_2_value}")
                    
            except Exception as e:
                self.fail(f"Integration test failed: {e}")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n=== Testing Edge Cases ===")
        
        # Test empty question
        extractor = qna_extractor(question="", project=self.project)
        result = extractor.extract_sentiment_level_one()
        self.assertIsNone(result, "Empty question should return None")
        
        # Test None question
        extractor = qna_extractor(question=None, project=self.project)
        result = extractor.extract_sentiment_level_one()
        self.assertIsNone(result, "None question should return None")
        
        # Test level 2 with None level 1
        extractor = qna_extractor(question="Test question", project=self.project)
        result = extractor.extract_sentiment_level_two(None)
        self.assertIsNone(result, "Level 2 with None level 1 should return None")
        
        # Test level 2 with invalid level 1
        result = extractor.extract_sentiment_level_two("InvalidSentiment")
        # This should handle gracefully
        
        print("Edge cases handled correctly")

    def test_both_projects_configuration(self):
        """Test that both MIRA and PCL project configurations work"""
        print("\n=== Testing Both Project Configurations ===")
        
        test_question = "What is my enrollment deadline?"
        
        for project in ["MIRA", "PCL"]:
            print(f"\nTesting {project} configuration:")
            
            try:
                # Test configuration loading
                config = load_project_config(project, "sentiment_extraction")
                self.assertIsNotNone(config, f"{project} config should load")
                
                # Test extraction
                extractor = qna_extractor(question=test_question, project=project)
                result = extractor.extract_sentiment_level_one()
                self.assertIsNotNone(result, f"{project} sentiment extraction should work")
                
                print(f"✅ {project} configuration working correctly")
                
            except Exception as e:
                self.fail(f"{project} configuration failed: {e}")

def run_sentiment_tests():
    """Run all sentiment extraction tests with detailed reporting"""
    print("="*80)
    print("SENTIMENT EXTRACTION TESTING SUITE")
    print("="*80)
    
    # Suppress some logging for cleaner test output
    logging.getLogger().setLevel(logging.WARNING)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests in logical order
    suite.addTest(TestSentimentExtraction('test_sentiment_configuration_loading'))
    suite.addTest(TestSentimentExtraction('test_both_projects_configuration'))
    suite.addTest(TestSentimentExtraction('test_edge_cases'))
    suite.addTest(TestSentimentExtraction('test_sentiment_level_1_extraction'))
    suite.addTest(TestSentimentExtraction('test_sentiment_level_2_extraction'))
    suite.addTest(TestSentimentExtraction('test_integration_with_qna_pipeline'))
    
    # Run tests with custom result handling
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    # Print detailed summary
    print(f"\n{'='*80}")
    print("TEST EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n{'='*40}")
        print("FAILURES:")
        print(f"{'='*40}")
        for test, traceback in result.failures:
            print(f"\n❌ {test}:")
            print(traceback)
    
    if result.errors:
        print(f"\n{'='*40}")
        print("ERRORS:")
        print(f"{'='*40}")
        for test, traceback in result.errors:
            print(f"\n💥 {test}:")
            print(traceback)
    
    print(f"\n{'='*80}")
    
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED! Sentiment extraction is working correctly.")
    else:
        print("❌ Some tests failed. Please review the issues above.")
    
    print(f"{'='*80}")
    
    return result.wasSuccessful()

def test_sentiment_extraction_summary():
    """Generate a summary of the sentiment extraction functionality"""
    print("\n" + "="*80)
    print("SENTIMENT EXTRACTION FUNCTIONALITY SUMMARY")
    print("="*80)
    
    summary = """
    ✅ IMPLEMENTED FEATURES:
    
    1. TWO-LEVEL SENTIMENT CLASSIFICATION:
       • Level 1: Positive, Neutral, Negative
       • Level 2: Specific subcategories for each main sentiment
    
    2. CONFIGURABLE SENTIMENT CATEGORIES:
       • JSON configuration files for MIRA and PCL projects
       • Easy to modify sentiment categories and descriptions
       • Supports different sentiment models per project
    
    3. INTEGRATED WITH EXISTING PIPELINE:
       • Added to qna_extractor class as two new methods
       • Automatically processes sentiments during QnA extraction
       • Stores results in Azure AI Search index fields
    
    4. ROBUST ERROR HANDLING:
       • Graceful handling of empty/null inputs
       • JSON parsing error recovery
       • Comprehensive logging for debugging
    
    5. REUSABLE LOGIC:
       • Can be used in both backfill scripts and hourly jobs
       • Supports both index and file-based processing
       • Works with existing concurrent processing framework
    
    📊 SENTIMENT CATEGORIES:
    
    Level 1 → Level 2:
    • Positive:
      - Satisfied: Expresses contentment with service/resolution
      - Grateful: Shows appreciation or thanks
      - Other: Any other positive sentiment
    
    • Neutral:
      - Informational: Factual statements without emotional tone
      - Procedural: Step-by-step or transactional dialogue
      - Other: Any other neutral sentiment
    
    • Negative:
      - Frustrated: Shows irritation or impatience
      - Angry: Strong dissatisfaction or aggression
      - Other: Any other negative sentiment
    
    🔧 TECHNICAL IMPLEMENTATION:
    
    • New methods: extract_sentiment_level_one(), extract_sentiment_level_two()
    • New config files: sentiment_config_mira.json, sentiment_config_pcl.json
    • New prompt functions: sentiment_level_1_prompt(), sentiment_level_2_prompt()
    • Updated Azure AI Search index with: sentiment_level_1, sentiment_level_2 fields
    • Enhanced CSV output includes sentiment columns
    
    🧪 TESTING COVERAGE:
    
    • Configuration loading and validation
    • Both project configurations (MIRA and PCL)
    • Edge cases and error handling
    • Integration with existing QnA pipeline
    • Accuracy validation for sentiment classification
    
    ✨ NEXT STEPS:
    
    1. Run the tests to validate functionality
    2. Deploy to staging environment for integration testing
    3. Monitor sentiment extraction accuracy in production
    4. Adjust sentiment categories based on real-world performance
    5. Implement sentiment statistics calculation (separate requirement)
    """
    
    print(summary)
    print("="*80)

if __name__ == "__main__":
    try:
        print("Starting sentiment extraction testing...")
        
        # Run the comprehensive test suite
        success = run_sentiment_tests()
        
        # Show the summary regardless of test results
        test_sentiment_extraction_summary()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Fatal error during testing: {e}")
        sys.exit(1)
