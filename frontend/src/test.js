// Test script to check API calls
document.addEventListener('DOMContentLoaded', function() {
    console.log('Test script loaded');
    
    // Test fake news
    const testFakeNews = {
        headline: "SHOCKING: Scientists discover miracle cure that doctors don't want you to know about",
        content: "This incredible breakthrough will change everything. The government is hiding this secret treatment that can cure all diseases overnight. You won't believe what happens next!"
    };
    
    // Test real news
    const testRealNews = {
        headline: "Local community center opens new facility",
        content: "The community center has completed construction of its new recreation facility. The building includes a gymnasium, swimming pool, and meeting rooms for local residents."
    };
    
    // Function to test API call
    async function testApiCall(news, type) {
        console.log(`Testing ${type} news API call:`, news);
        
        const formData = new FormData();
        formData.append('headline', news.headline);
        formData.append('content', news.content);
        
        try {
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log(`${type} news API result:`, result);
            
            // Display result on page
            const resultDiv = document.createElement('div');
            resultDiv.innerHTML = `
                <h3>${type} News Test</h3>
                <p><strong>Headline:</strong> ${news.headline}</p>
                <p><strong>Content:</strong> ${news.content}</p>
                <p><strong>Prediction:</strong> ${result.prediction}</p>
                <p><strong>Confidence:</strong> ${result.confidence}</p>
                <p><strong>Explanation:</strong> ${result.explanation}</p>
            `;
            document.body.appendChild(resultDiv);
            
        } catch (error) {
            console.error(`Error testing ${type} news:`, error);
        }
    }
    
    // Run tests
    testApiCall(testFakeNews, 'Fake');
    setTimeout(() => {
        testApiCall(testRealNews, 'Real');
    }, 2000);
}); 