import requests
from newsapi import NewsApiClient
import os

# Initialize NewsAPI client with API key
API_KEY = os.environ.get("NEWS_API_KEY", "your_api_key")  # Use the provided API key
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
        print(f"Error fetching related news: {e}")
        # Return a structured error response instead of empty list
        return {"status": "error", "message": f"Failed to fetch related news: {str(e)}"}

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
        print(f"Error searching news by keywords: {e}")
        return {"status": "error", "message": f"Failed to search news by keywords: {str(e)}"}

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
    {"name": "FactCheck.org", "search_url": "https://www.factcheck.org/?s="},
    {"name": "Reuters Fact Check", "search_url": "https://www.reuters.com/fact-check/search?q="},
    {"name": "AP Fact Check", "search_url": "https://apnews.com/hub/ap-fact-check?query="},
    {"name": "TruthOrFiction", "search_url": "https://www.truthorfiction.com/?s="},
    {"name": "Lead Stories", "search_url": "https://leadstories.com/?s="},
    {"name": "Full Fact (UK)", "search_url": "https://fullfact.org/search/?q="},
    {"name": "Africa Check", "search_url": "https://africacheck.org/search?query="},
    {"name": "Boom Live (India)", "search_url": "https://www.boomlive.in/search?query="},
    {"name": "Alt News (India)", "search_url": "https://www.altnews.in/?s="},
    {"name": "Factly (India)", "search_url": "https://factly.in/?s="},
    {"name": "The Logical Indian Fact Check", "search_url": "https://thelogicalindian.com/search/?q="},
    {"name": "AFP Fact Check", "search_url": "https://factcheck.afp.com/?search_api_fulltext="},
    {"name": "Poynter Institute", "search_url": "https://www.poynter.org/?s="},
    {"name": "Check Your Fact", "search_url": "https://checkyourfact.com/?s="},
    {"name": "Washington Post Fact Checker", "search_url": "https://www.washingtonpost.com/news/fact-checker/?s="},
    {"name": "NYT Fact Checks", "search_url": "https://www.nytimes.com/search?query=fact+check"},
    {"name": "CNN Facts First", "search_url": "https://edition.cnn.com/search?q=fact+check"},
    {"name": "BBC Reality Check", "search_url": "https://www.bbc.co.uk/search?q=reality+check"},
    {"name": "Deutsche Welle Fact Check", "search_url": "https://www.dw.com/en/top-stories/fact-check/s-32866"},
    {"name": "Fact Crescendo (India)", "search_url": "https://english.factcrescendo.com/?s="},
    {"name": "Newschecker (India)", "search_url": "https://newschecker.in/search?query="},
    {"name": "Vishvas News (India)", "search_url": "https://www.vishvasnews.com/en/search/"},
    {"name": "Myth Detector (Georgia)", "search_url": "https://mythdetector.ge/en/search?text="},
    {"name": "Pagella Politica (Italy)", "search_url": "https://pagellapolitica.it/"},
    {"name": "Maldita (Spain)", "search_url": "https://maldita.es/maldito-bulo/"},
    {"name": "FactCheckNI (Northern Ireland)", "search_url": "https://factcheckni.org/"},
    {"name": "Faktisk (Norway)", "search_url": "https://www.faktisk.no/"},
    {"name": "Correctiv (Germany)", "search_url": "https://correctiv.org/"},
    {"name": "Demagog (Poland)", "search_url": "https://demagog.org.pl/"},
    {"name": "Factcheck.kz (Kazakhstan)", "search_url": "https://factcheck.kz/"},
    {"name": "StopFake (Ukraine)", "search_url": "https://www.stopfake.org/en/search/"},
    {"name": "Verificado (Mexico)", "search_url": "https://verificado.com.mx/"},
    {"name": "Chequeado (Argentina)", "search_url": "https://chequeado.com/"},
    {"name": "El Sabueso (Mexico)", "search_url": "https://www.animalpolitico.com/elsabueso/"},
    {"name": "Colombiacheck", "search_url": "https://colombiacheck.com/"},
    {"name": "UOL Confere (Brazil)", "search_url": "https://noticias.uol.com.br/confere/"},
    {"name": "Aos Fatos (Brazil)", "search_url": "https://www.aosfatos.org/"},
    {"name": "FactCheckEU", "search_url": "https://eufactcheck.eu/"},
    {"name": "Observador Fact Check (Portugal)", "search_url": "https://observador.pt/seccao/fact-check/"},
    {"name": "The Quint WebQoof (India)", "search_url": "https://www.thequint.com/news/webqoof"},
    {"name": "Media Bias/Fact Check", "search_url": "https://mediabiasfactcheck.com/"},
    {"name": "Verafiles (Philippines)", "search_url": "https://verafiles.org/"},
    {"name": "Rappler Fact Check (Philippines)", "search_url": "https://www.rappler.com/fact-check/"},
    {"name": "Teyit (Turkey)", "search_url": "https://teyit.org/"},
    {"name": "Channel 4 Fact Check (UK)", "search_url": "https://www.channel4.com/news/factcheck"},
    {"name": "Politifact Australia", "search_url": "https://www.politifact.com/australia/"},
    {"name": "RMIT ABC Fact Check (Australia)", "search_url": "https://www.abc.net.au/news/factcheck/"},
    {"name": "The Conversation Fact Check", "search_url": "https://theconversation.com/au/topics/factcheck-924"},
    {"name": "OpenSecrets.org", "search_url": "https://www.opensecrets.org/search?search="},
    {"name": "Open the Government", "search_url": "https://www.openthegovernment.org/"},
    {"name": "Science Feedback", "search_url": "https://sciencefeedback.co/"},
    {"name": "Health Feedback", "search_url": "https://healthfeedback.org/"},
    {"name": "Climate Feedback", "search_url": "https://climatefeedback.org/"},
    {"name": "FactCheck Ghana", "search_url": "https://fact-checkghana.com/"},
    {"name": "Congo Check", "search_url": "https://congocheck.net/"},
    {"name": "PesaCheck (Africa)", "search_url": "https://pesacheck.org/"},
    {"name": "FactCheckHub (Nigeria)", "search_url": "https://factcheckhub.com/"},
    {"name": "Dubawa (West Africa)", "search_url": "https://dubawa.org/"},
    {"name": "DUBAWA Ghana", "search_url": "https://ghana.dubawa.org/"},
    {"name": "FactCheck Kenya", "search_url": "https://www.mediacouncil.or.ke/en/mck/index.php/fact-check"},
    {"name": "FactCheck Initiative Nigeria", "search_url": "https://factchecki.org/"},
    {"name": "KallxCheck (Kosovo)", "search_url": "https://kallxo.com/fakte/"},
    {"name": "The Healthy Indian Project (India)", "search_url": "https://www.thip.media/"},
    {"name": "NewsMeter (India)", "search_url": "https://newsmeter.in/fact-check"},
    {"name": "FactAlyze (Pakistan)", "search_url": "https://factalyze.org/"},
    {"name": "SMHoaxSlayer (India)", "search_url": "https://www.smhoaxslayer.com/"},
    {"name": "Fake News Debunker (Italy)", "search_url": "https://www.bufale.net/"},
    {"name": "Gulf News Fact Check", "search_url": "https://gulfnews.com/world/fact-check"},
    {"name": "Arab Fact Check", "search_url": "https://fatabyyano.net/"},
    {"name": "MyGoPen (Taiwan)", "search_url": "https://www.mygopen.com/"},
    {"name": "FactCheck Mongolia", "search_url": "https://factcheck.mn/"},
    {"name": "Internews Ukraine", "search_url": "https://internews.ua/"},
    {"name": "Mimikama (Austria)", "search_url": "https://www.mimikama.at/"},
    {"name": "Desinfaux (France)", "search_url": "https://www.desinfaux.com/"},
    {"name": "Rumor Scanner (Bangladesh)", "search_url": "https://rumorscanner.org/"},
    {"name": "Open Data Kosovo", "search_url": "https://opendatakosovo.org/"},
    {"name": "MDF Fact Checking", "search_url": "https://factchecking.mdfgeorgia.ge/"},
    {"name": "Fact-Check Armenia", "search_url": "https://factcheck.ge/en/"},
    {"name": "FactCheck.kz", "search_url": "https://factcheck.kz/en/"},
    {"name": "Kazakh Fact Check", "search_url": "https://kazfactcheck.kz/"},
    {"name": "Lie Detectors", "search_url": "https://lie-detectors.org/"},
    {"name": "Factosfera (Russia)", "search_url": "https://factosfera.ru/"},
    {"name": "VerifySy", "search_url": "https://verify-sy.com/en/"},
    {"name": "FakeHunter (Poland)", "search_url": "https://fakehunter.pap.pl/"},
    {"name": "Faktabaari (Finland)", "search_url": "https://faktabaari.fi/"},
    {"name": "ThinkCheckSubmit (scholarly sources)", "search_url": "https://thinkchecksubmit.org/"},
    {"name": "HoaxEye (Visual Claims)", "search_url": "https://hoaxeye.com/"},
    {"name": "Hoax Slayer (Archive)", "search_url": "http://www.hoax-slayer.net/"},
    {"name": "Hoaxbuster (France)", "search_url": "https://www.hoaxbuster.com/"},
    {"name": "Infotagion", "search_url": "https://infotagion.com/"},
    {"name": "MediaWise", "search_url": "https://www.poynter.org/mediawise/"},
    {"name": "VerifyThis", "search_url": "https://verifythis.com/"},
    {"name": "International Fact-Checking Network", "search_url": "https://ifcncodeofprinciples.poynter.org/signatories"}
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
