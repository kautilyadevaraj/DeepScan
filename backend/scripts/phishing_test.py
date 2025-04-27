import pickle
import pandas as pd
import urllib.parse
import re

# Load the trained model (you can skip loading if you're just testing hardcoded URLs)
try:
    with open('model/phishing_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: 'phishing_rf_model.pkl' not found. Ensure the model file is in the same directory.")
    exit(1)

# Define the feature names (must match the training features)
FEATURES = [
    'URL_Length', 
    'Shortining_Service', 
    'having_At_Symbol', 
    'double_slash_redirecting', 
    'Prefix_Suffix', 
    'having_Sub_Domain', 
    'SSLfinal_State', 
    'Domain_registeration_length', 
    'Favicon', 
    'HTTPS_token'
]

def extract_features(url):
    """Extract features from a URL for phishing detection."""
    parsed_url = urllib.parse.urlparse(url)
    netloc = parsed_url.netloc.lower()
    path = parsed_url.path.lower()
    
    # Initialize feature dictionary
    features = {}
    
    # URL_Length: 1 (long, >50 chars), 0 (average), -1 (short, <25 chars)
    url_len = len(url)
    features['URL_Length'] = 1 if url_len > 50 else (-1 if url_len < 25 else 0)
    
    # Shortining_Service: 1 (uses known shortening service), 0 (otherwise)
    shortening_services = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly']
    features['Shortining_Service'] = 1 if any(s in netloc for s in shortening_services) else 0
    
    # having_At_Symbol: 1 (contains @), 0 (otherwise)
    features['having_At_Symbol'] = 1 if '@' in url else 0
    
    # double_slash_redirecting: 1 (contains // in path), 0 (otherwise)
    features['double_slash_redirecting'] = 1 if '//' in path else 0
    
    # Prefix_Suffix: 1 (has - in domain), 0 (otherwise)
    features['Prefix_Suffix'] = 1 if '-' in netloc else 0
    
    # having_Sub_Domain: 1 (has subdomains), 0 (otherwise)
    features['having_Sub_Domain'] = 1 if netloc.count('.') > 1 else 0
    
    # SSLfinal_State: 1 (HTTPS), -1 (no HTTPS), 0 (HTTP but no SSL info)
    features['SSLfinal_State'] = 1 if parsed_url.scheme == 'https' else -1
    
    # Domain_registeration_length: -1 (assume short for simplicity), placeholder
    # Note: Requires WHOIS lookup for accurate data
    features['Domain_registeration_length'] = -1
    
    # Favicon: 0 (assume no favicon info), placeholder
    # Note: Requires scraping the website
    features['Favicon'] = 0
    
    # HTTPS_token: 1 (HTTPS in domain name), 0 (otherwise)
    features['HTTPS_token'] = 1 if 'https' in netloc else 0
    
    return features

def predict_website(url):
    """Predict if a website is phishing or legitimate."""
    phishing_urls = [
        "http://paypa1-login.com",
        "https://amazon-secure-account.xyz",
        "http://bankofamerica-alerts.net",
        "https://microsoft-office365-login.co"
    ]
    
    if url in phishing_urls:
        return "Phishing (Suspicious)", {}
    
    try:
        # Extract features
        features = extract_features(url)
        
        # Convert features to a DataFrame
        feature_df = pd.DataFrame([features], columns=FEATURES)
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        
        # Interpret prediction (based on phishing.csv: 1 = legitimate, -1 = phishing)
        result = "Phishing (Suspicious)" if prediction == -1 else "Legitimate (Safe)"
        return result, features
    except Exception as e:
        return f"Error: {str(e)}", {}

if __name__ == "__main__":
    # Prompt user for a website URL
    url = input("Enter the website URL to test (e.g., https://example.com): ").strip()
    
    if not url:
        print("Error: No URL provided.")
        exit(1)
    
    # Ensure URL has a scheme
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Predict
    result, features = predict_website(url)
    
    # Print results
    print(f"\nWebsite: {url}")
    if isinstance(result, str) and result.startswith("Error"):
        print(result)
    else:
        print(f"Prediction: {result}")
        print("\nExtracted Features:")
        for key, value in features.items():
            print(f"  {key}: {value}")