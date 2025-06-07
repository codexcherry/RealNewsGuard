// RealNewsGuard Frontend JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const newsForm = document.getElementById('news-form');
    const headlineInput = document.getElementById('headline');
    const contentInput = document.getElementById('content');
    const imageInput = document.getElementById('image');
    const fileNameDisplay = document.getElementById('file-name');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeImageButton = document.getElementById('remove-image');
    const analyzeButton = document.getElementById('analyze-button');
    const resultsSection = document.getElementById('results-section');
    const predictionElement = document.getElementById('prediction');
    const confidenceMeter = document.getElementById('confidence-meter');
    const confidenceValue = document.getElementById('confidence-value');
    const explanationElement = document.getElementById('explanation');
    const relatedNewsContainer = document.getElementById('related-news-container');
    const factChecksContainer = document.getElementById('fact-checks-container');
    const loadingOverlay = document.getElementById('loading-overlay');
    const newAnalysisButton = document.getElementById('new-analysis');

    // API Endpoint
    const API_URL = 'http://localhost:8000';

    // Event Listeners
    newsForm.addEventListener('submit', handleFormSubmit);
    imageInput.addEventListener('change', handleImageSelection);
    removeImageButton.addEventListener('click', removeSelectedImage);
    newAnalysisButton.addEventListener('click', resetForm);

    /**
     * Handle form submission
     * @param {Event} event - Form submit event
     */
    async function handleFormSubmit(event) {
        event.preventDefault();
        
        // Validate form
        if (!validateForm()) {
            return;
        }
        
        // Show loading overlay
        loadingOverlay.classList.remove('hidden');
        
        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('headline', headlineInput.value.trim());
            formData.append('content', contentInput.value.trim());
            
            if (imageInput.files.length > 0) {
                formData.append('image', imageInput.files[0]);
            }
            
            console.log('Sending request to:', `${API_URL}/analyze`);
            
            // Send request to API
            const response = await fetch(`${API_URL}/analyze`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            // Parse response
            const result = await response.json();
            console.log('API Response:', result);
            
            // Display results
            displayResults(result);
            
        } catch (error) {
            console.error('Error analyzing news:', error);
            alert('An error occurred while analyzing the news. Please try again.');
        } finally {
            // Hide loading overlay
            loadingOverlay.classList.add('hidden');
        }
    }

    /**
     * Validate form inputs
     * @returns {boolean} - Whether the form is valid
     */
    function validateForm() {
        if (!headlineInput.value.trim()) {
            alert('Please enter a headline');
            headlineInput.focus();
            return false;
        }
        
        if (!contentInput.value.trim()) {
            alert('Please enter content');
            contentInput.focus();
            return false;
        }
        
        return true;
    }

    /**
     * Handle image selection
     * @param {Event} event - Change event
     */
    function handleImageSelection(event) {
        const file = event.target.files[0];
        
        if (file) {
            // Update file name display
            fileNameDisplay.textContent = file.name;
            
            // Show image preview
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreviewContainer.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        } else {
            removeSelectedImage();
        }
    }

    /**
     * Remove the selected image
     */
    function removeSelectedImage() {
        imageInput.value = '';
        fileNameDisplay.textContent = 'Choose an image';
        imagePreviewContainer.classList.add('hidden');
    }

    /**
     * Display analysis results
     * @param {Object} result - API response data
     */
    function displayResults(result) {
        // Hide input section and show results section
        newsForm.parentElement.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
        // Set prediction
        predictionElement.textContent = result.prediction;
        predictionElement.className = 'prediction ' + result.prediction.toLowerCase();
        
        // Log the raw result for debugging
        console.log('Raw prediction result:', result);
        
        // Set confidence meter
        const confidencePercent = (result.confidence * 100).toFixed(0);
        confidenceMeter.className = 'confidence-meter ' + result.prediction.toLowerCase();
        
        // Update the width of the meter
        confidenceMeter.style.setProperty('--width', `${confidencePercent}%`);
        confidenceValue.textContent = `${confidencePercent}%`;
        
        // Set explanation
        explanationElement.textContent = result.explanation || 'No specific explanation available.';
        
        // Display related news
        displayRelatedNews(result.related_news);
        
        // Display fact checks
        displayFactChecks(result.fact_checks);
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * Display related news articles
     * @param {Array} news - Related news articles
     */
    function displayRelatedNews(news) {
        relatedNewsContainer.innerHTML = '';
        
        if (!news || news.length === 0) {
            relatedNewsContainer.innerHTML = '<p class="no-results">No related news found.</p>';
            return;
        }
        
        news.forEach(article => {
            const newsItem = document.createElement('div');
            newsItem.className = 'news-item';
            
            const title = document.createElement('div');
            title.className = 'news-item-title';
            title.textContent = article.title;
            
            const source = document.createElement('div');
            source.className = 'news-item-source';
            source.textContent = `Source: ${article.source}`;
            
            const link = document.createElement('a');
            link.className = 'news-item-link';
            link.href = article.url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = 'Read article';
            
            newsItem.appendChild(title);
            newsItem.appendChild(source);
            
            if (article.published_at) {
                const date = document.createElement('div');
                date.className = 'news-item-date';
                date.textContent = formatDate(article.published_at);
                newsItem.appendChild(date);
            }
            
            newsItem.appendChild(link);
            
            relatedNewsContainer.appendChild(newsItem);
        });
    }

    /**
     * Display fact check results
     * @param {Object} factChecks - Fact check results
     */
    function displayFactChecks(factChecks) {
        factChecksContainer.innerHTML = '';
        
        if (!factChecks) {
            factChecksContainer.innerHTML = '<p class="no-results">No fact-check information available.</p>';
            return;
        }
        
        // Process Snopes results
        if (factChecks.snopes && factChecks.snopes.found) {
            const snopesSection = createFactCheckSection('Snopes', factChecks.snopes.results);
            factChecksContainer.appendChild(snopesSection);
        }
        
        // Process PolitiFact results
        if (factChecks.politifact && factChecks.politifact.found) {
            const politifactSection = createFactCheckSection('PolitiFact', factChecks.politifact.results);
            factChecksContainer.appendChild(politifactSection);
        }
        
        // Process FactCheck.org results
        if (factChecks.factcheck_org && factChecks.factcheck_org.found) {
            const factcheckOrgSection = createFactCheckSection('FactCheck.org', factChecks.factcheck_org.results);
            factChecksContainer.appendChild(factcheckOrgSection);
        }
        
        // If no fact checks found
        if (factChecksContainer.children.length === 0) {
            factChecksContainer.innerHTML = '<p class="no-results">No matches found on fact-checking sites.</p>';
        }
    }

    /**
     * Create a fact check section
     * @param {string} source - Fact check source name
     * @param {Array} results - Fact check results
     * @returns {HTMLElement} - Fact check section element
     */
    function createFactCheckSection(source, results) {
        const section = document.createElement('div');
        section.className = 'fact-check-section';
        
        const sourceHeading = document.createElement('h4');
        sourceHeading.textContent = source;
        section.appendChild(sourceHeading);
        
        results.forEach(result => {
            const item = document.createElement('div');
            item.className = 'fact-check-item';
            
            // Add rating if available
            if (result.rating) {
                const ratingClass = getRatingClass(result.rating);
                const rating = document.createElement('span');
                rating.className = `fact-check-rating ${ratingClass}`;
                rating.textContent = result.rating;
                item.appendChild(rating);
            }
            
            const title = document.createElement('div');
            title.className = 'fact-check-item-title';
            title.textContent = result.title;
            
            const link = document.createElement('a');
            link.className = 'fact-check-item-link';
            link.href = result.url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = 'View fact check';
            
            item.appendChild(title);
            item.appendChild(link);
            
            section.appendChild(item);
        });
        
        return section;
    }

    /**
     * Get CSS class for a rating
     * @param {string} rating - Rating text
     * @returns {string} - CSS class
     */
    function getRatingClass(rating) {
        const lowerRating = rating.toLowerCase();
        
        if (lowerRating.includes('true') || lowerRating.includes('fact') || lowerRating.includes('correct')) {
            return 'rating-true';
        } else if (lowerRating.includes('false') || lowerRating.includes('pants') || lowerRating.includes('fiction')) {
            return 'rating-false';
        } else {
            return 'rating-mixed';
        }
    }

    /**
     * Format date string
     * @param {string} dateString - Date string
     * @returns {string} - Formatted date
     */
    function formatDate(dateString) {
        if (!dateString) return '';
        
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        } catch (error) {
            return dateString;
        }
    }

    /**
     * Reset form and show input section
     */
    function resetForm() {
        // Clear form inputs
        newsForm.reset();
        removeSelectedImage();
        
        // Hide results section and show input section
        resultsSection.classList.add('hidden');
        newsForm.parentElement.classList.remove('hidden');
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}); 