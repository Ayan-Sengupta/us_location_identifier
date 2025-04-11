# main.py
# This is the main entry point for the US Location Identifier 

import logging
from load_data import load_data
from process_bio import add_location_columns
import spacy
import os 

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# flag variable to skip parts of the pipeline if needed
flag = 0

def _load_and_process():
    logger.info("Starting US Location Identifier pipeline")
    # Load the data
    df = load_data('data/users_with_locations.csv.gz')
    if df.empty:
        logger.error("No data loaded. Exiting.")
        exit(1)
    logger.info(f"Loaded {len(df)} rows of data")
    # process the bio to see if location info can be extracted
    # load the spacy model we need
    # Load spaCy transformer model
    logger.info("Loading spaCy model: en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    # Add location columns to the dataframe
    logger.info("Adding spacy location columns to the dataframe")
    df = add_location_columns(df, nlp)
    logger.info("Completed adding location columns")
    # look at the shape of the new dataframe
    logger.info(f"Shape of dataframe after adding location columns: {df.shape}")
    # save to a csv file
    output_path = 'data/featureEngineered/users_with_locations_sm.csv.gz'
    df.to_csv(output_path,compression='gzip', index=False)
    logger.info(f"Saved processed data to {output_path}")

def _update_flag():
    global flag  
    # check if the file exists
    value = os.path.exists('data/featureEngineered/users_with_locations_sm.csv.gz')
    # if yes set flag to 1 
    if value:
        flag = 1
        

    
if __name__ == '__main__':
    # print the current working directory
    logger.info(f"Current working directory: {os.getcwd()}")
    _update_flag()
    if flag == 0:
        logger.info("No feature engineered data found. Loading and processing data")
        _load_and_process()
    if flag == 1:
        logger.info("Found feature engineered data. Skipping loading and processing data")
        df = load_data('data/featureEngineered/users_with_locations_sm.csv.gz')
        logger.info(f"Loaded {len(df)} rows of feature engineered data")











