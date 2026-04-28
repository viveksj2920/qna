import pandas as pd
from qna_extractor_topic_backfill import qna_extractor

# Definition of class that process batch of transcripts
class BatchProcessor:

    def __init__(self, dataframe, input_dict):
        
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("The input should be a Pandas DataFrame")

        self.dataframe = dataframe
        self.input_dict = input_dict

    def process(self):

        qna = qna_extractor()

        batch_qna = qna.extract_batch(self.dataframe, self.input_dict)
        return batch_qna
