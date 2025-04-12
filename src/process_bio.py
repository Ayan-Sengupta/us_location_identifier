# constants and functions to process the full bio datat of the reviewers - to extract any info about their geographical location
import re
from tqdm import tqdm
import logging


# logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# constanst
# US cities and states for validation/lookup
_us_states = set([
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", 
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", 
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", 
    "maine", "maryland", "massachusetts", "michigan", "minnesota", 
    "mississippi", "missouri", "montana", "nebraska", "nevada", 
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina", 
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", 
    "rhode island", "south carolina", "south dakota", "tennessee", "texas", 
    "utah", "vermont", "virginia", "washington", "west virginia", 
    "wisconsin", "wyoming", "dc", "district of columbia"
])
# Add state abbreviations
_state_abbrevs = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc"
}
_us_states.update(_state_abbrevs)

# Top 100 US cities by population
_top_us_cities = {
    "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
    "san antonio", "san diego", "dallas", "san jose", "austin", "jacksonville",
    "fort worth", "columbus", "charlotte", "indianapolis", "san francisco",
    "seattle", "denver", "washington", "boston", "el paso", "nashville",
    "detroit", "oklahoma city", "portland", "las vegas", "memphis", "louisville",
    "baltimore", "milwaukee", "albuquerque", "tucson", "fresno", "sacramento",
    "kansas city", "mesa", "atlanta", "omaha", "colorado springs", "raleigh",
    "miami", "long beach", "virginia beach", "oakland", "minneapolis", "tampa",
    "tulsa", "arlington", "wichita", "cleveland", "bakersfield", "aurora", 
    "new orleans", "honolulu", "anaheim", "tampa", "pittsburgh", "cincinnati",
    "st louis", "riverside", "corpus christi", "lexington", "anchorage",
    "stockton", "toledo", "st paul", "newark", "greensboro", "buffalo", 
    "plano", "lincoln", "henderson", "fort wayne", "jersey city", "st petersburg",
    "chula vista", "orlando", "durham", "chandler", "laredo", "madison", "lubbock",
    "scottsdale", "reno", "glendale", "boise", "richmond", "baton rouge", "irvine", 
    "spokane", "tacoma", "irving", "hialeah", "fremont", "birmingham",
    "rochester", "san bernardino", "boise city"
}

state_pattern = re.compile(r"[A-Za-z\s]+,\s*([A-Z]{2})")
zip_pattern = re.compile(r"\b\d{5}(?:-\d{4})?\b")


# helper functions 
# Function to detect language as english
def _is_english(lang):
    """
    Detect if the text is in English. - no need for langdetect now
    """
    if lang == "en":
        return True
    else:
        return False
        

# Function to extract location from bio text using spaCy
def _extract_location(text, nlp_model):
    """
    Extract locations from a batch of texts using spaCy's pipe.
    """
    locations = []
    for doc in nlp_model.pipe(text, batch_size=1000):  # Adjust batch_size as needed
        locs = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        locations.append(locs[0] if locs else "nil")
    return locations
    
# Function to check if a location is in the US
def _is_us_location(location_text):
    if not location_text:
        return False
        
    location_lower = location_text.lower()
    
    # Direct state or city match
    if any(state in location_lower.split() for state in _us_states) or \
       any(city in location_lower for city in _top_us_cities):
        return True
    
    # Check for state pattern (City, ST)
    state_match = state_pattern.search( location_text)
    if state_match and state_match.group(1).lower() in _state_abbrevs:
        return True
    
    # Check for zip code pattern
    if zip_pattern.search(location_text):
        return True
    
    # Check for USA/America mentions
    us_terms = ["usa", "america", "united states", "u.s.", "u.s.a."]
    if any(term in location_lower for term in us_terms):
        return True
        
    return False

# Function to add location columns to dataframe
#  uses tqdm to show progress and the two helper functions above
def add_location_columns(df, nlp_model, batch_size=1000):
    '''
    adds to new columns to the dataframe:
    spacy_location: the location found in the bio
    spacy_us_location: 1 if the location is in the US, 0 otherwise
    '''
    #Replace NaN or non-string values in 'full_bio' with an empty string
    df['full_bio'] = df['full_bio'].fillna("").astype(str)
    # Filter out non-English bios
    # Detect language and filter for English bios -show a progress bar
    tqdm.pandas(desc="Detecting English bios")
    # Use a lambda function to apply the is_english function to each bio
    english_df = df[df['bio_language'].progress_apply(lambda x: _is_english(x))]
    # Process in batches
    total_batches = (len(english_df) + batch_size - 1) // batch_size
    spacy_locations = []

    logger.info(f"Filtered out {len(df) - len(english_df)} non-English bios\n")
    
    for i in tqdm(range(total_batches), desc="Detecting locations in", unit="batch"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(english_df))
        batch = english_df['full_bio'].iloc[start_idx:end_idx]
        
        # Extract locations in batch
        spacy_locations.extend(_extract_location(batch, nlp_model))
    
    # Add extracted locations to the DataFrame
    english_df['spacy_location'] = spacy_locations
    
    # Check if the extracted location is in the US
    english_df['spacy_us_location'] = english_df['spacy_location'].apply(lambda loc: int(_is_us_location(loc)))
    # Merge results back into the original DataFrame
    df = df.merge(english_df[['spacy_location', 'spacy_us_location']], how='left', left_index=True, right_index=True)
    df['spacy_location'] = df['spacy_location'].fillna("nil")
    df['spacy_us_location'] = df['spacy_us_location'].fillna(0)
    
    return df
