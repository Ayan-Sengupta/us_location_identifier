import pandas as pd



def _drop_preserve(df):
    """
    Drop columns that are not needed for the model and preserve the original dataframe.
    """
    # drop the columns that are not needed for the model
    df_removed_preserved = df[['reviewer_profile','bio_language','name','metadata','full_bio','spacy_location']]
    df = df.drop(columns=['bio_language','name','metadata','full_bio','spacy_location'])
