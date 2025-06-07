import requests
from newsapi import NewsApiClient
import os

# Initialize NewsAPI client with API key
API_KEY = "f1117364d20b40909d9ce1a53a5cb9e3"
newsapi = NewsApiClient(api_key=API_KEY)

def get_related_news(query, max_results=5):
    """
    Fetch related news articles based on the query using NewsAPI.
    
    Args:
        query (str): The search query (typically the headline)
        max_results (int): Maximum number of results to return
        
    Returns:
        list: List of related news articles
    """
    try:
        # Search for related news
        response = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='relevancy',
            page_size=max_results
        )
        
        # Extract relevant information from each article
        articles = []
        for article in response.get('articles', [])[:max_results]:
            articles.append({
                'title': article.get('title', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'description': article.get('description', '')
            })
        
        return articles
    
    except Exception as e:
        print(f"Error fetching related news: {str(e)}")
        return []

def search_news_by_keywords(keywords, days_back=7, max_results=10):
    """
    Search for news articles based on keywords over a specific time period.
    
    Args:
        keywords (list): List of keywords to search for
        days_back (int): Number of days to look back
        max_results (int): Maximum number of results to return
        
    Returns:
        list: List of news articles matching the keywords
    """
    try:
        from datetime import datetime, timedelta
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for NewsAPI
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Combine keywords for search
        query = ' OR '.join(keywords)
        
        # Search for news
        response = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='relevancy',
            page_size=max_results
        )
        
        return response.get('articles', [])
    
    except Exception as e:
        print(f"Error searching news by keywords: {str(e)}")
        return []

def search_fact_checking_sites(query):
    """
    Search fact-checking websites for information about the query
    
    Args:
        query (str): The search query
        
    Returns:
        list: List of fact-check articles related to the query
    """
    # List of fact-checking sites to search
    fact_check_sites = [
        {"name": "Snopes", "search_url": "https://www.snopes.com/?s="},
        {"name": "PolitiFact", "search_url": "https://www.politifact.com/search/?q="},
        {"name": "FactCheck.org", "search_url": "https://www.factcheck.org/?s="}
    ]
    
    results = []
    
    # In a real implementation, we would use BeautifulSoup to scrape these sites
    # For now, we'll just return the search URLs
    for site in fact_check_sites:
        results.append({
            "site": site["name"],
            "search_url": f"{site['search_url']}{query.replace(' ', '+')}",
            "status": "Search link provided (scraping not implemented)"
        })
    
    return results 