import requests

def test_api():
    # API endpoint
    api_url = "http://localhost:8000/analyze"
    
    # Test cases
    test_cases = [
        {
            "type": "Obvious Fake News",
            "data": {
                "headline": "SHOCKING: Government hiding ALIENS in secret underground base - You won't believe the evidence!",
                "content": "Top secret documents reveal that the government is hiding alien technology and extraterrestrial beings in a classified underground facility. Whistleblowers claim that this conspiracy has been ongoing for decades. The mainstream media won't report on this because they are controlled by the same people who are hiding the truth about aliens. This shocking revelation will change everything you know about our world."
            }
        },
        {
            "type": "Fake News",
            "data": {
                "headline": "Chocolate Cures COVID-19, Claims New Study",
                "content": "A recent  by an unknown health organization claims that eating large amounts of chocolate can completely cure COVID-19. The report, which went viral on social media, suggests that compounds in chocolate kill the virus instantly and recommends people consume at least 5 bars daily. Medical professionals, however, have not verified these claims, and no official sources back the study."
            }
        },
        {
            "type": "Suspicious News",
            "data": {
                "headline": "Breaking: New study reveals potential health benefits",
                "content": "A recent study suggests that a common food might have unexpected health benefits. Researchers are still investigating the findings, which have not yet been peer-reviewed. Some claim this discovery could be revolutionary, but experts remain cautious."
            }
        },
        {
            "type": "Real News",
            "data": {
                "headline": "Local community center opens new facility",
                "content": "The community center has completed construction of its new recreation facility. The building includes a gymnasium, swimming pool, and meeting rooms for local residents. The project was funded by a combination of municipal grants and private donations."
            }
        }
    ]
    
    # Test each case
    for case in test_cases:
        print(f"\n\n{'='*50}")
        print(f"Testing {case['type']}:")
        print(f"Headline: {case['data']['headline']}")
        print(f"Content: {case['data']['content'][:100]}...")
        
        try:
            # Send request to API
            response = requests.post(api_url, data=case['data'])
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Print results
                print("\nAPI Response:")
                print(f"Prediction: {result.get('prediction', 'N/A')}")
                print(f"Confidence: {result.get('confidence', 'N/A')}")
                print(f"Explanation: {result.get('explanation', 'N/A')}")
            else:
                print(f"Error: API returned status code {response.status_code}")
                print(response.text)
        
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api() 