#!/usr/bin/env python
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def search_kayako_articles(query, locale=None, brand_id=None):
    """
    Search for articles in Kayako Help Center
    
    Args:
        query (str): Search query (must be more than 3 characters)
        locale (str, optional): Locale for the search
        brand_id (int, optional): Brand ID to filter by
    
    Returns:
        dict: JSON response from the API
    """
    # Get Kayako API credentials from environment variables
    api_url = os.getenv('KAYAKO_API_URL', 'https://your-kayako-instance.kayako.com/api/v1')
    api_username = os.getenv('KAYAKO_API_USERNAME')
    api_password = os.getenv('KAYAKO_API_PASSWORD')
    
    # Ensure URL ends with the correct endpoint
    search_url = f"{api_url}/helpcenter/search/articles.json"
    
    # Prepare the request data
    data = {
        "query": query
    }
    
    # Add optional parameters if provided
    if locale:
        data["locale"] = locale
    if brand_id:
        data["brand_id"] = brand_id
    
    # Set up authentication
    auth = (api_username, api_password)
    
    # Make the API request
    print(f"Searching for articles with query: '{query}'")
    try:
        response = requests.post(search_url, auth=auth, json=data)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Return the JSON response
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None

def main():
    """Main function to run the script"""
    # Example search query
    query = "login"
    
    # Search for articles
    response = search_kayako_articles(query)
    
    # Print the response
    if response:
        print("\nAPI Response:")
        print(json.dumps(response, indent=4))
        
        # Print summary of results
        if response.get('status') == 200:
            articles = response.get('data', [])
            print(f"\nFound {len(articles)} articles:")
            for i, article in enumerate(articles, 1):
                title = article.get('titles', [{}])[0].get('id', 'Untitled')
                print(f"{i}. Article ID: {article.get('id')} - Title ID: {title}")
        else:
            print(f"\nAPI returned status: {response.get('status')}")
    else:
        print("No response received from the API")

if __name__ == "__main__":
    main()
