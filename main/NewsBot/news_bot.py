# here there will be a news bot for free news.
import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#the function of this script is to tell which stocks to trade in

"""stocks = [
    'Apple Inc.',
    'Microsoft Corporation',
    'Alphabet Inc.',
    'Amazon.com, Inc.',
    'Facebook, Inc. (Meta Platforms, Inc.)',
    'Tesla, Inc.',
    'Berkshire Hathaway Inc.',
    'Johnson & Johnson',
    'JPMorgan Chase & Co.',
    'Visa Inc.',
    'Procter & Gamble Co.',
    'UnitedHealth Group Incorporated',
    'Mastercard Incorporated',
    'The Home Depot, Inc.',
    'NVIDIA Corporation',
    'PayPal Holdings, Inc.',
    'Intel Corporation',
    'Adobe Inc.',
    'Netflix, Inc.',
    'The Walt Disney Company',
    'Cisco Systems, Inc.',
    'PepsiCo, Inc.',
    'Pfizer Inc.',
    'Nike, Inc.',
    'Coca-Cola Company',
    'Exxon Mobil Corporation'
]"""
stocks = [
    'Bitcoin',
    'Ethereum',
    'Binance Coin',
    'Cardano',
    'Solana',
    'Ripple (XRP)',
    'Polkadot',
    'Dogecoin',
    'Shiba Inu',
    'Litecoin',
    'Chainlink',
    'Uniswap',
    'Stellar',
    'Polygon (MATIC)',
    'Avalanche',
    'Terra (LUNA)',
    'VeChain',
    'Tron',
    'Cosmos (ATOM)',
    'Tezos',
    'Monero',
    'Algorand',
    'Aave',
    'Elrond',
    'IOTA',
    'Filecoin'
]

def get_news(api_key, company_name, num_articles=100):
    # Get the date for one day ago
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # News API endpoint
    endpoint = "https://newsapi.org/v2/everything"

    # Set the parameters
    params = {
        'apiKey': api_key,
        'q': company_name,
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': num_articles,
        'from': start_date,
        'to': datetime.now().strftime('%Y-%m-%d')
    }

    try:
        # Make the request to the API
        response = requests.get(endpoint, params=params)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Extract and print the news articles
            articles = data['articles']
            all_texts = ""
            for article in articles:
                if company_name.lower() in article['title'].lower():
                    title = article['title']
                    description = article['description']
                    url = article['url']
                    
                    print(f"Title: {title}")
                    print(f"Description: {description}")
                    print(f"URL: {url}")
                    print("\n" + "="*50 + "\n")

                    # Concatenate title and description to form the text for sentiment analysis
                    all_texts += f"{title} {description} "

            return all_texts.strip()
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Replace 'YOUR_API_KEY' with your actual News API key
api_key = '1f30615161cf4037b369adb5e4e0efb6'

# Call the function to get news about the specified company
def main():
    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()
    
    for company_name in stocks:
        text = get_news(api_key, company_name, num_articles=100)
        if text:
            sentiment = analyzer.polarity_scores(text)
            print(f"Text: {text}")
            print(f"The compound Sentiment for {company_name} is: {sentiment['compound']} the positive Sentiment is: {sentiment['pos']} the negative sentiment is {sentiment['neg']} and neutral sentiment is {sentiment['neu']}")
            if sentiment['pos'] > sentiment['neg']*1.5:
                print(f'Buy {company_name}')
            elif sentiment['pos'] < sentiment['neg']*1.5:
                print(f'Sell {company_name}')
            else:
                print(f'Hold {company_name}')
        # else:
        #     print(f"No relevant articles found for {company_name}")
main()


# so now only trade buy for the stocks which are here