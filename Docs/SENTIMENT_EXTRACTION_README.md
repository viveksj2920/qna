# Sentiment Extraction Implementation

This document describes the sentiment extraction functionality that has been implemented for the QnA extraction pipeline.

## Overview

The sentiment extraction feature analyzes customer questions and classifies them into two levels of sentiment categories:

- **Level 1**: High-level sentiment (Positive, Neutral, Negative)
- **Level 2**: Specific subcategories within each main sentiment

## Implementation Details

### 1. New Methods Added to `qna_extractor` Class

#### `extract_sentiment_level_one()`
- Extracts high-level sentiment (Positive, Neutral, Negative)
- Returns JSON string with `sentiment_level_1` field
- Uses configurable sentiment categories from JSON files

#### `extract_sentiment_level_two(sentiment_level_1)`
- Extracts detailed sentiment subcategory
- Requires Level 1 sentiment as input parameter
- Returns JSON string with `sentiment_level_2` field
- Uses Level 1 result to determine appropriate subcategories

### 2. Configuration Files

#### Sentiment Categories Configuration
- `src/data/input/sentiment_config_mira.json` - MIRA project sentiment configuration
- `src/data/input/sentiment_config_pcl.json` - PCL project sentiment configuration

#### Configuration Structure
```json
{
  "level_1": {
    "Positive": "Expresses satisfaction, gratitude, or other positive emotions",
    "Neutral": "Factual, informational, or procedural without emotional tone",
    "Negative": "Shows dissatisfaction, frustration, anger, or other negative emotions"
  },
  "level_2": {
    "positive": {
      "Satisfied": "Expresses contentment with service or resolution",
      "Grateful": "Shows appreciation or thanks",
      "Other": "Any other positive sentiment"
    },
    "neutral": {
      "Informational": "Factual statements without emotional tone",
      "Procedural": "Step-by-step or transactional dialogue",
      "Other": "Any other neutral sentiment"
    },
    "negative": {
      "Frustrated": "Shows irritation or impatience",
      "Angry": "Strong dissatisfaction or aggression",
      "Other": "Any other negative sentiment"
    }
  }
}
```

### 3. Sentiment Categories

#### Level 1 → Level 2 Mapping

**Positive**
- **Satisfied**: Expresses contentment with service or resolution
- **Grateful**: Shows appreciation or thanks
- **Other**: Any other positive sentiment

**Neutral**
- **Informational**: Factual statements without emotional tone
- **Procedural**: Step-by-step or transactional dialogue
- **Other**: Any other neutral sentiment

**Negative**
- **Frustrated**: Shows irritation or impatience
- **Angry**: Strong dissatisfaction or aggression
- **Other**: Any other negative sentiment

### 4. Integration with Pipeline

The sentiment extraction is integrated into the main QnA processing pipeline:

1. **Index Processing**: Sentiment fields are automatically added to Azure AI Search index documents
2. **File Processing**: Sentiment columns are included in CSV output files
3. **Batch Processing**: Works with existing concurrent processing framework
4. **Error Handling**: Robust error handling with comprehensive logging

### 5. Azure AI Search Index Fields

Two new fields are added to the search index:
- `sentiment_level_1`: String field containing high-level sentiment
- `sentiment_level_2`: String field containing detailed sentiment subcategory

## Usage

### Basic Usage

```python
from qna_extractor import qna_extractor

# Create extractor instance
extractor = qna_extractor(question="Thank you for your help!", project="MIRA")

# Extract Level 1 sentiment
level_1_json = extractor.extract_sentiment_level_one()
level_1_data = json.loads(level_1_json)
sentiment_level_1 = level_1_data.get('sentiment_level_1')

# Extract Level 2 sentiment
level_2_json = extractor.extract_sentiment_level_two(sentiment_level_1)
level_2_data = json.loads(level_2_json)
sentiment_level_2 = level_2_data.get('sentiment_level_2')

print(f"Sentiment: {sentiment_level_1} - {sentiment_level_2}")
```

### Integration with Existing Pipeline

The sentiment extraction is automatically included when processing conversations or questions through the existing pipeline. No additional code changes are required in the main processing scripts.

## Testing

### Running Tests

```bash
# Run comprehensive test suite
python test_sentiment_extraction.py

# Run demonstration script
python demo_sentiment_extraction.py
```

### Test Coverage

The test suite includes:
- Configuration loading and validation
- Both project configurations (MIRA and PCL)
- Edge cases and error handling
- Integration with existing QnA pipeline
- Accuracy validation for sentiment classification

## Files Modified

### Core Implementation
- `src/qna_extractor.py`: Added sentiment extraction methods and integration
- `src/config.py`: Added sentiment configuration file paths
- `src/utils/helper.py`: Added sentiment extraction to configuration loading
- `src/prompts/prompt_config.py`: Added sentiment extraction prompt functions

### Configuration Files
- `src/data/input/sentiment_config_mira.json`: MIRA sentiment configuration
- `src/data/input/sentiment_config_pcl.json`: PCL sentiment configuration

### Testing and Documentation
- `test_sentiment_extraction.py`: Comprehensive test suite
- `demo_sentiment_extraction.py`: Demonstration script
- `SENTIMENT_EXTRACTION_README.md`: This documentation

## Technical Considerations

### Performance
- Sentiment extraction adds minimal overhead to existing processing
- Uses same LLM infrastructure as other extractions
- Concurrent processing maintains performance scalability

### Reliability
- Comprehensive error handling prevents pipeline failures
- JSON parsing errors are handled gracefully
- Missing sentiment data doesn't block other extractions

### Configurability
- Sentiment categories are easily modifiable via JSON files
- Different configurations for different projects
- No code changes required to adjust sentiment classifications

### Reusability
- Same logic works for both backfill scripts and hourly jobs
- Compatible with both index-based and file-based processing
- Can be used independently of other extraction features

## Future Enhancements

1. **Statistics Calculation**: Implement separate functionality to calculate sentiment statistics
2. **Custom Categories**: Add support for project-specific sentiment categories
3. **Confidence Scoring**: Add confidence scores to sentiment classifications
4. **Batch Optimization**: Optimize batch processing for large sentiment extractions
5. **Historical Analysis**: Add capability to analyze sentiment trends over time

## Support

For questions or issues related to sentiment extraction:
1. Check the test suite output for specific error messages
2. Verify configuration files are properly formatted JSON
3. Ensure Azure OpenAI credentials are correctly configured
4. Review the logs for detailed error information

## Version History

- **v1.0**: Initial implementation with two-level sentiment classification
- Support for MIRA and PCL projects
- Integration with existing QnA extraction pipeline
- Comprehensive testing suite
