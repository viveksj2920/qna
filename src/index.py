import re
import os
import io
import openai
import json
import requests
import pandas as pd
import numpy as np
import time
import datetime

from dotenv import load_dotenv  
load_dotenv()
from urllib.request import Request, urlopen
from azure.core.exceptions import HttpResponseError 
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents.indexes import SearchIndexClient
from azure_search.index_handler import AzureSearchIndexUtility


class IndexProcessor:
    
    def __init__(self, index_name = ""):
        self.index_name = index_name
        self.azure_search = AzureSearchIndexUtility(index_name = self.index_name)
    

    def fetch_records(self, date_filter, project, max_records=0):
        df = pd.DataFrame()
        search_kwargs = {"filter": date_filter}
        if max_records > 0:
            search_kwargs["top"] = max_records
        if project == "MIRA":
            fields = ["Ucid", "Text", "StartTime", "Is_Digital", "Is_Enrollment", "plan_name", "drugs", "providers", "zip", "county_processed", "state_processed", "region_processed", "subregion_processed"]
            search_kwargs["select"] = fields
            documents = self.azure_search.search(**search_kwargs)
            print(f"Processing fields: {fields}")
            df = pd.DataFrame(documents)
        elif project == "PCL":
            fields = ["Ucid", "Text", "StartTime", "sales_market", "business_market", "region", "subregion", "state"]
            search_kwargs["select"] = fields
            documents = self.azure_search.search(**search_kwargs)
            print(f"Processing fields: {fields}")
            df = pd.DataFrame(documents)
        else:
            raise ValueError("Unsupported project type. Supported projects are: MIRA, PCL.")
        return df


    def update_index(self, result_data, key_field_name, semantic_content_field="question"):
        self.azure_search.push_to_index(result_data, key_field_name=key_field_name, semantic_content_field=semantic_content_field)



