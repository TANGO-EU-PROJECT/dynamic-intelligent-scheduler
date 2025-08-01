import re

def preprocess_column_name(name):
    """
    Function that takes as input a column name and preprocesses it. 
    More specifically it replaces special characters and spaces with _ and converts to lowercase.
    """
    name = re.sub(r"[^\w\s]", '_', name)
    return name.replace(" ", "_").lower()