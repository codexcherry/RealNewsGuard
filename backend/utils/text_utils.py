import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Common clickbait phrases and patterns
CLICKBAIT_PATTERNS = [
    r"you won't believe",
    r"mind(-|\s)blowing",
    r"shocking",
    r"jaw(-|\s)dropping",
    r"unbelievable",
    r"incredible",
    r"won't believe",
    r"never guess",
    r"can't even handle",
    r"secret",
    r"top \d+",
    r"\d+ reasons",
    r"this is why",
    r"this is what happens",
    r"what happens next",
    r"when you see",
    r"must see",
    r"need to know",
    r"here's why",
    r"find out",
    r"you'll never",
    r"will make you",
]

# Sensationalist language patterns
SENSATIONALIST_PATTERNS = [
    r"breaking",
    r"urgent",
    r"alert",
    r"exclusive",
    r"bombshell",
    r"just in",
    r"scandal",
    r"controversy",
    r"explosive",
    r"shocking truth",
    r"revealed",
    r"exposes",
    r"secret",
    r"conspiracy",
]

# Suspicious claim patterns
SUSPICIOUS_CLAIM_PATTERNS = [
    r"scientists shocked",
    r"doctors hate",
    r"they don't want you to know",
    r"the government is hiding",
    r"what they don't want you to know",
    r"banned",
    r"censored",
    r"suppressed",
    r"cover(-|\s)up",
    r"mainstream media won't tell you",
    r"the truth about",
    r"cures? (cancer|covid|covid(-|\s)19|coronavirus|aids|hiv)",
    r"miracle (cure|treatment|remedy)",
    r"instant(ly)? (cure|heal)",
    r"complete(ly)? cure",
    r"medical professionals (don't want you to know|won't tell you)",
    r"big pharma",
    r"not verified",
    r"viral (on social media|online)",
    r"unknown (source|organization|researcher)",
    r"claims? (new study|research)",
    r"eat(ing)? large amounts",
]

def clean_text(text):
    """
    Clean and preprocess text by removing special characters, extra spaces, etc.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def detect_clickbait(headline):
    """
    Detect if a headline has clickbait characteristics.
    
    Args:
        headline (str): News headline
        
    Returns:
        tuple: (is_clickbait, matched_patterns)
    """
    headline = headline.lower()
    matched_patterns = []
    
    for pattern in CLICKBAIT_PATTERNS:
        if re.search(pattern, headline):
            matched_patterns.append(pattern)
    
    is_clickbait = len(matched_patterns) > 0
    
    return is_clickbait, matched_patterns

def detect_sensationalist_language(text):
    """
    Detect sensationalist language in text.
    
    Args:
        text (str): Input text
        
    Returns:
        tuple: (is_sensationalist, matched_patterns)
    """
    text = text.lower()
    matched_patterns = []
    
    for pattern in SENSATIONALIST_PATTERNS:
        if re.search(pattern, text):
            matched_patterns.append(pattern)
    
    is_sensationalist = len(matched_patterns) > 0
    
    return is_sensationalist, matched_patterns

def detect_suspicious_claims(text):
    """
    Detect suspicious claims in text.
    
    Args:
        text (str): Input text
        
    Returns:
        tuple: (has_suspicious_claims, matched_patterns)
    """
    text = text.lower()
    matched_patterns = []
    
    for pattern in SUSPICIOUS_CLAIM_PATTERNS:
        if re.search(pattern, text):
            matched_patterns.append(pattern)
    
    has_suspicious_claims = len(matched_patterns) > 0
    
    return has_suspicious_claims, matched_patterns

def compute_similarity(text1, text2):
    """
    Compute cosine similarity between two texts using TF-IDF.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    try:
        # Fit and transform the texts
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Compute cosine similarity
        similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
        
        return float(similarity)
    
    except Exception as e:
        print(f"Error computing similarity: {str(e)}")
        return 0.0

def extract_keywords(text, top_n=10):
    """
    Extract top keywords from text using TF-IDF.
    
    Args:
        text (str): Input text
        top_n (int): Number of top keywords to extract
        
    Returns:
        list: Top keywords with their scores
    """
    if not text:
        return []
    
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Fit and transform the text
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Sort keywords by score
        keywords = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return keywords[:top_n]
    
    except Exception as e:
        print(f"Error extracting keywords: {str(e)}")
        return [] 