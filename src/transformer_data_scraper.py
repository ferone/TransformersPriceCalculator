import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import re
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransformerDataScraper:
    """
    Scraper for transformer price data from Volza.com
    
    This scraper collects real market data for electrical transformers
    including country of origin, destination, date, HSN code, quantity,
    description, and price.
    """
    
    def __init__(self, output_dir='data'):
        """
        Initialize the scraper
        
        Parameters:
        -----------
        output_dir : str
            Directory to save scraped data
        """
        self.output_dir = output_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://www.google.com/'
        }
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _random_delay(self, min_seconds=1, max_seconds=3):
        """Add random delay between requests to be respectful to the server"""
        time.sleep(random.uniform(min_seconds, max_seconds))
    
    def _extract_transformer_data(self, row):
        """
        Extract transformer data from a table row
        
        Parameters:
        -----------
        row : BeautifulSoup object
            Table row containing transformer data
        
        Returns:
        --------
        dict
            Dictionary containing extracted data or None if not relevant
        """
        try:
            cells = row.find_all('td')
            if len(cells) < 5:  # Skip rows with insufficient data
                return None
                
            # Try to extract the data (table format may vary across pages)
            date_text = cells[0].get_text(strip=True)
            hsn_code = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            description = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            origin_country = cells[3].get_text(strip=True) if len(cells) > 3 else ""
            quantity_text = cells[4].get_text(strip=True) if len(cells) > 4 else ""
            price_text = cells[5].get_text(strip=True) if len(cells) > 5 else ""
            
            # Skip if not transformer related
            if not self._is_transformer_description(description):
                return None
                
            # Clean and extract data
            quantity = self._extract_number(quantity_text)
            unit_price = self._extract_price(price_text)
            
            # Try to extract power rating from description
            power_rating = self._extract_power_rating(description)
            
            return {
                'Date': date_text,
                'HSN Code': hsn_code,
                'Description': description,
                'Origin': origin_country,
                'Destination': self._extract_destination_from_url(self.current_url),
                'Quantity': quantity,
                'Unit Price (USD)': unit_price,
                'Power Rating (KVA)': power_rating,
                'Source URL': self.current_url
            }
            
        except Exception as e:
            logger.warning(f"Error extracting data from row: {e}")
            return None
    
    def _is_transformer_description(self, description):
        """Check if the description is related to electrical transformers"""
        description = description.lower()
        transformer_keywords = [
            'transformer', 'transformers', 'electrical transformer', 
            'power transformer', 'distribution transformer'
        ]
        
        return any(keyword in description for keyword in transformer_keywords)
    
    def _extract_number(self, text):
        """Extract numeric values from text"""
        if not text:
            return None
        
        numbers = re.findall(r'[\d,]+\.?\d*', text)
        if numbers:
            return float(numbers[0].replace(',', ''))
        return None
    
    def _extract_price(self, text):
        """Extract price from text"""
        if not text:
            return None
            
        # Remove currency symbols and commas
        cleaned_text = re.sub(r'[^\d.,]', '', text)
        numbers = re.findall(r'[\d,]+\.?\d*', cleaned_text)
        if numbers:
            return float(numbers[0].replace(',', ''))
        return None
    
    def _extract_power_rating(self, description):
        """Extract power rating in KVA from transformer description"""
        description = description.lower()
        
        # Look for KVA or kVA or kva followed by numbers
        kva_patterns = [
            r'(\d+[\d\.,]*)[\s-]*kva',
            r'(\d+[\d\.,]*)[\s-]*k\.?v\.?a',
            r'(\d+[\d\.,]*)[\s-]*kilo\s*volt\s*ampere',
            r'(\d+[\d\.,]*)[\s-]*kilo\s*va'
        ]
        
        for pattern in kva_patterns:
            match = re.search(pattern, description)
            if match:
                return float(match.group(1).replace(',', ''))
        
        # Look for MVA and convert to KVA
        mva_patterns = [
            r'(\d+[\d\.,]*)[\s-]*mva',
            r'(\d+[\d\.,]*)[\s-]*m\.?v\.?a',
            r'(\d+[\d\.,]*)[\s-]*mega\s*volt\s*ampere',
            r'(\d+[\d\.,]*)[\s-]*mega\s*va'
        ]
        
        for pattern in mva_patterns:
            match = re.search(pattern, description)
            if match:
                # Convert MVA to KVA (1 MVA = 1000 KVA)
                return float(match.group(1).replace(',', '')) * 1000
                
        return None
    
    def _extract_destination_from_url(self, url):
        """Extract destination country from URL"""
        pattern = r'import-in-([a-z-]+)'
        match = re.search(pattern, url)
        if match:
            country = match.group(1).replace('-', ' ').title()
            return country
        return None
    
    def scrape_url(self, url):
        """
        Scrape transformer data from the specified URL
        
        Parameters:
        -----------
        url : str
            URL to scrape
        
        Returns:
        --------
        list
            List of dictionaries containing transformer data
        """
        self.current_url = url
        logger.info(f"Scraping URL: {url}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            data_table = soup.find('table', {'class': 'table'})
            
            if not data_table:
                logger.warning(f"No data table found at {url}")
                return []
            
            all_rows = data_table.find_all('tr')
            transformers_data = []
            
            # Skip the header row
            for row in all_rows[1:]:
                data = self._extract_transformer_data(row)
                if data:
                    transformers_data.append(data)
                    
            logger.info(f"Scraped {len(transformers_data)} transformer entries from {url}")
            return transformers_data
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []
    
    def scrape_multiple_urls(self, urls):
        """
        Scrape transformer data from multiple URLs
        
        Parameters:
        -----------
        urls : list
            List of URLs to scrape
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing all scraped transformer data
        """
        all_data = []
        
        for url in urls:
            data = self.scrape_url(url)
            all_data.extend(data)
            self._random_delay(2, 5)  # Be respectful to the server
        
        if not all_data:
            logger.warning("No transformer data was scraped from any URL")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Add timestamp
        df['Scrape Date'] = datetime.now().strftime('%Y-%m-%d')
        
        return df
    
    def save_data(self, df, filename=None):
        """
        Save scraped data to CSV
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing scraped data
        filename : str, optional
            Filename to save data to. If None, a default name will be used.
            
        Returns:
        --------
        str
            Path to saved file
        """
        if df.empty:
            logger.warning("No data to save")
            return None
            
        if filename is None:
            filename = f"transformer_market_data_{datetime.now().strftime('%Y%m%d')}.csv"
            
        output_path = os.path.join(self.output_dir, filename)
        
        # If file exists, append without duplicates
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            # Concatenate and drop duplicates based on all columns
            combined_df = pd.concat([existing_df, df]).drop_duplicates()
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Appended data to existing file. Total rows: {len(combined_df)}")
        else:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} rows to {output_path}")
            
        return output_path


def scrape_transformer_data(urls=None):
    """
    Main function to scrape transformer data from Volza
    
    Parameters:
    -----------
    urls : list, optional
        List of URLs to scrape. If None, default URLs will be used.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all scraped transformer data
    """
    if urls is None:
        urls = [
            "https://www.volza.com/p/electrical-transformer/import/import-in-united-kingdom/",
            "https://www.volza.com/p/electrical-transformer/import/import-in-united-states/",
            "https://www.volza.com/p/hsn-code-850423/import/import-in-spain/",
            "https://www.volza.com/p/hsn-code-850423/import/import-in-united-kingdom/",
            "https://www.volza.com/p/hsn-code-850423/import/import-in-india/",
            "https://www.volza.com/p/electrical-transformer/import/import-in-germany/",
            "https://www.volza.com/p/electrical-transformer/import/import-in-france/",
            "https://www.volza.com/p/electrical-transformer/import/import-in-japan/",
            "https://www.volza.com/p/electrical-transformer/import/import-in-canada/"
        ]
    
    scraper = TransformerDataScraper()
    df = scraper.scrape_multiple_urls(urls)
    
    if not df.empty:
        scraper.save_data(df)
        
    return df


def load_scraped_data():
    """
    Load the most recent scraped transformer data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing scraped transformer data
    """
    data_dir = 'data'
    
    # Find the most recent transformer market data file
    files = [f for f in os.listdir(data_dir) if f.startswith('transformer_market_data_') and f.endswith('.csv')]
    
    if not files:
        logger.warning("No scraped transformer data found")
        return pd.DataFrame()
    
    # Sort by modification time (most recent first)
    latest_file = sorted(files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)[0]
    file_path = os.path.join(data_dir, latest_file)
    
    logger.info(f"Loading scraped data from {file_path}")
    return pd.read_csv(file_path)


if __name__ == "__main__":
    print("Scraping transformer market data from Volza...")
    df = scrape_transformer_data()
    
    if not df.empty:
        print(f"Successfully scraped {len(df)} transformer entries")
        print("\nData sample:")
        print(df.head())
    else:
        print("No transformer data was scraped") 