# main.py
# This is the main entry point for the US Location Identifier 

import logging
from load_data import load_data
from process_bio import add_location_columns
import spacy

# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
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

