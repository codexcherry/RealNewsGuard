import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Common clickbait phrases and patterns - expanded list
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
    r"changed forever",
    r"can't stop",
    r"epic",
    r"insane",
    r"amazing",
    r"believe your eyes",
    r"gone wrong",
    r"gone viral",
    r"never seen before",
    r"before it's deleted",
    r"before it's too late",
    r"they don't want you to see",
    r"what they found",
    r"what happened next",
    r"doctors hate",
    r"one simple trick",
    r"simple way",
    r"weird trick",
    r"weird method",
    r"secret trick",
    r"secret method",
    r"miracle",
    r"game(-|\s)changer",
    r"life(-|\s)changing",
]

# Sensationalist language patterns - expanded list
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
    r"outrage",
    r"furious",
    r"slams",
    r"destroys",
    r"obliterates",
    r"rips",
    r"tears into",
    r"blasts",
    r"erupts",
    r"meltdown",
    r"chaos",
    r"crisis",
    r"nightmare",
    r"disaster",
    r"catastrophe",
    r"terrifying",
    r"horrifying",
    r"devastating",
    r"ruins",
    r"tragic",
    r"heartbreaking",
    r"unreal",
    r"insane",
    r"disturbing",
    r"bizarre",
    r"extreme",
    r"savage",
    r"brutal",
    r"massive",
    r"huge",
    r"enormous",
    r"epic",
    r"perfect",
    r"absolutely",
    r"completely",
    r"totally",
    r"literally",
    r"actually",
    r"honestly",
    r"officially",
    r"finally",
]

# Suspicious claim patterns - expanded list
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
    r"prevents? (cancer|covid|covid(-|\s)19|coronavirus|aids|hiv)",
    r"treats? (cancer|covid|covid(-|\s)19|coronavirus|aids|hiv)",
    r"eliminates? (cancer|covid|covid(-|\s)19|coronavirus|aids|hiv)",
    r"destroys? (cancer|covid|covid(-|\s)19|coronavirus|aids|hiv)",
    r"kills? (cancer|covid|covid(-|\s)19|coronavirus|aids|hiv)",
    r"natural (cure|treatment|remedy)",
    r"ancient (cure|treatment|remedy)",
    r"secret (cure|treatment|remedy)",
    r"hidden (cure|treatment|remedy)",
    r"alternative (cure|treatment|remedy)",
    r"proven (cure|treatment|remedy)",
    r"guaranteed (cure|treatment|remedy)",
    r"(100%|completely) effective",
    r"(100%|completely) safe",
    r"no side effects",
    r"risk(-|\s)free",
    r"(doctors|scientists|experts) (are|were) (stunned|shocked|amazed|surprised)",
    r"(doctors|scientists|experts) (don't|won't) tell you",
    r"(doctors|scientists|experts) (hate|fear) this",
    r"(big pharma|the government|they) (doesn't want|don't want|won't allow)",
    r"(what|things) (they|the government|big pharma) (don't|doesn't) want you to know",
    r"(they|the government|big pharma) (is|are) hiding",
    r"(they|the government|big pharma) (is|are) lying",
    r"(they|the government|big pharma) (is|are) covering up",
    r"(they|the government|big pharma) (doesn't|don't) want you to see",
    r"(they|the government|big pharma) (doesn't|don't) want you to know",
    r"(they|the government|big pharma) (doesn't|don't) want you to find out",
    r"(they|the government|big pharma) (is|are) keeping (this|it) from you",
    r"(they|the government|big pharma) (is|are) keeping (this|it) secret",
    r"(they|the government|big pharma) (is|are) suppressing (this|it)",
    r"(they|the government|big pharma) (is|are) censoring (this|it)",
    r"(they|the government|big pharma) (is|are) banning (this|it)",
    r"(they|the government|big pharma) (doesn't|don't) want (this|it) to get out",
    r"(they|the government|big pharma) (is|are) afraid of (this|it)",
    r"(they|the government|big pharma) (is|are) terrified of (this|it)",
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
    
    # Check for pattern matches
    for pattern in CLICKBAIT_PATTERNS:
        matches = re.findall(pattern, headline)
        if matches:
            matched_patterns.extend(matches)
    
    # Check for question headlines (often clickbait)
    if re.search(r'\?$', headline) and any(word in headline for word in ['you', 'your', 'these', 'this', 'why', 'how', 'what']):
        matched_patterns.append("question_headline")
    
    # Check for all caps words (often clickbait)
    if re.search(r'\b[A-Z]{3,}\b', headline):
        matched_patterns.append("all_caps")
    
    # Check for excessive punctuation (often clickbait)
    if headline.count('!') > 1 or headline.count('?') > 1:
        matched_patterns.append("excessive_punctuation")
    
    # Check for numbers at the beginning (often listicles, which are common clickbait)
    if re.match(r'^\d+\s', headline):
        matched_patterns.append("listicle")
    
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
    
    # Check for pattern matches
    for pattern in SENSATIONALIST_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            matched_patterns.extend(matches)
    
    # Check for excessive exclamation marks
    if text.count('!') > 2:
        matched_patterns.append("excessive_exclamation")
    
    # Check for ALL CAPS sentences
    if re.search(r'[A-Z]{5,}', text):
        matched_patterns.append("all_caps")
    
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
    
    # Check for pattern matches
    for pattern in SUSPICIOUS_CLAIM_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            if isinstance(matches[0], tuple):
                matched_patterns.extend([m for m in matches if m])
            else:
                matched_patterns.extend(matches)
    
    # Check for absolute claims
    absolute_claims = [
        r"(always|never|all|none|every|everyone|nobody|guaranteed)",
        r"(100%|completely|totally|absolutely) (proven|guaranteed|effective|safe)"
    ]
    
    for pattern in absolute_claims:
        if re.search(pattern, text):
            matched_patterns.append("absolute_claim")
            break
    
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

def analyze_sentiment(text):
    """
    Analyze the sentiment of the text (positive, negative, neutral).
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Sentiment analysis results
    """
    # Simple rule-based sentiment analysis
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love',
        'happy', 'positive', 'success', 'successful', 'win', 'winning', 'won',
        'benefit', 'beneficial', 'advantage', 'advantageous', 'helpful', 'useful'
    ]
    
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'sad',
        'negative', 'fail', 'failure', 'failed', 'lose', 'losing', 'lost',
        'harm', 'harmful', 'disadvantage', 'disadvantageous', 'useless'
    ]
    
    # Tokenize and clean text
    words = word_tokenize(clean_text(text))
    
    # Count sentiment words
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Calculate sentiment score (-1 to 1)
    total_count = positive_count + negative_count
    if total_count > 0:
        sentiment_score = (positive_count - negative_count) / total_count
    else:
        sentiment_score = 0.0
    
    # Determine sentiment label
    if sentiment_score > 0.25:
        sentiment = "positive"
    elif sentiment_score < -0.25:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "score": sentiment_score,
        "positive_count": positive_count,
        "negative_count": negative_count
    } 