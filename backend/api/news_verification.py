import requests
from bs4 import BeautifulSoup
import re
import time

def verify_news(headline, content=None):
    """
    Verify news by checking against fact-checking websites
    
    Args:
        headline (str): The news headline
        content (str, optional): The news content
        
    Returns:
        dict: Verification results from fact-checking sites
    """
    # Use the verify_news_against_fact_checking_websites function
    return verify_news_against_fact_checking_websites(headline, content)

def simulate_similar_articles(query):
    """
    Simulate finding similar articles (in a real implementation, this would use 
    text embeddings and cosine similarity to find similar articles)
    
    Args:
        query (str): The search query
        
    Returns:
        list: List of simulated similar articles
    """
    # This is a placeholder. In a real implementation, we would:
    # 1. Convert the query to embeddings
    # 2. Compare with embeddings of articles in our database
    # 3. Return the most similar articles
    
    # For demonstration purposes only
    return [
        {
            "title": f"Similar article to '{query[:30]}...'",
            "source": "Example News",
            "url": "https://example.com/news/1",
            "similarity_score": 0.85,
            "is_fake": True
        },
        {
            "title": f"Another perspective on '{query[:20]}...'",
            "source": "Trusted Source",
            "url": "https://trusted-source.org/article/2",
            "similarity_score": 0.72,
            "is_fake": False
        }
    ]

def check_image_reuse(image_url):
    """
    Check if an image has been reused from other contexts
    
    Args:
        image_url (str): URL of the image to check
        
    Returns:
        dict: Results of the image reuse check
    """
    # In a real implementation, this would use reverse image search APIs
    # like Google Images, TinEye, or a custom solution
    
    # For demonstration purposes only
    return {
        "is_reused": False,
        "original_sources": [],
        "context_mismatch": False
    }

def verify_news_against_fact_checking_websites(headline, content):
    """
    Verify news against fact-checking websites.
    
    Args:
        headline (str): News headline
        content (str): News content
        
    Returns:
        dict: Results from fact-checking sites
    """
    # Combine results from multiple fact-checking sites
    results = {
        "snopes": check_snopes(headline),
        "politifact": check_politifact(headline),
        "factcheck_org": check_factcheck_org(headline)
    }
    
    return results

def check_snopes(query):
    """
    Check if the news has been fact-checked on Snopes.
    
    Args:
        query (str): Search query (headline)
        
    Returns:
        dict: Fact-check results from Snopes
    """
    try:
        # Format query for URL
        search_query = "+".join(query.split())
        url = f"https://www.snopes.com/?s={search_query}"
        
        # Make request with user-agent to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find fact-check articles
            articles = soup.find_all('article', class_='list-group-item')
            
            if articles:
                results = []
                for article in articles[:3]:  # Limit to top 3 results
                    title_elem = article.find('h2', class_='title')
                    rating_elem = article.find('span', class_='rating-name')
                    link_elem = article.find('a', href=True)
                    
                    if title_elem and link_elem:
                        result = {
                            "title": title_elem.text.strip(),
                            "url": link_elem['href'],
                            "rating": rating_elem.text.strip() if rating_elem else "Unknown"
                        }
                        results.append(result)
                
                return {
                    "found": True,
                    "results": results
                }
            
            return {
                "found": False,
                "message": "No fact-checks found on Snopes"
            }
        
        return {
            "found": False,
            "message": f"Error accessing Snopes: {response.status_code}"
        }
    
    except Exception as e:
        return {
            "found": False,
            "message": f"Error checking Snopes: {str(e)}"
        }

def check_politifact(query):
    """
    Check if the news has been fact-checked on PolitiFact.
    
    Args:
        query (str): Search query (headline)
        
    Returns:
        dict: Fact-check results from PolitiFact
    """
    try:
        # Format query for URL
        search_query = "+".join(query.split())
        url = f"https://www.politifact.com/search/?q={search_query}"
        
        # Make request with user-agent to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find fact-check articles
            articles = soup.find_all('li', class_='o-listicle__item')
            
            if articles:
                results = []
                for article in articles[:3]:  # Limit to top 3 results
                    title_elem = article.find('a', class_='m-statement__link')
                    rating_elem = article.find('div', class_='m-statement__meter')
                    
                    if title_elem:
                        result = {
                            "title": title_elem.text.strip(),
                            "url": f"https://www.politifact.com{title_elem['href']}",
                            "rating": rating_elem.text.strip() if rating_elem else "Unknown"
                        }
                        results.append(result)
                
                return {
                    "found": True,
                    "results": results
                }
            
            return {
                "found": False,
                "message": "No fact-checks found on PolitiFact"
            }
        
        return {
            "found": False,
            "message": f"Error accessing PolitiFact: {response.status_code}"
        }
    
    except Exception as e:
        return {
            "found": False,
            "message": f"Error checking PolitiFact: {str(e)}"
        }

def check_factcheck_org(query):
    """
    Check if the news has been fact-checked on FactCheck.org.
    
    Args:
        query (str): Search query (headline)
        
    Returns:
        dict: Fact-check results from FactCheck.org
    """
    try:
        # Format query for URL
        search_query = "+".join(query.split())
        url = f"https://www.factcheck.org/?s={search_query}"
        
        # Make request with user-agent to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find fact-check articles
            articles = soup.find_all('article')
            
            if articles:
                results = []
                for article in articles[:3]:  # Limit to top 3 results
                    title_elem = article.find('h2', class_='entry-title')
                    link_elem = title_elem.find('a') if title_elem else None
                    
                    if title_elem and link_elem:
                        result = {
                            "title": title_elem.text.strip(),
                            "url": link_elem['href'],
                            "summary": article.find('div', class_='entry-content').text.strip()[:150] + "..." if article.find('div', class_='entry-content') else ""
                        }
                        results.append(result)
                
                return {
                    "found": True,
                    "results": results
                }
            
            return {
                "found": False,
                "message": "No fact-checks found on FactCheck.org"
            }
        
        return {
            "found": False,
            "message": f"Error accessing FactCheck.org: {response.status_code}"
        }
    
    except Exception as e:
        return {
            "found": False,
            "message": f"Error checking FactCheck.org: {str(e)}"
        } 