import requests
from bs4 import BeautifulSoup

#Fetches and displays all holiday dates and details from the specified URL.
def extract_holidays(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Locate the holiday table by inspecting the page's HTML structure
        holiday_table = soup.find('table', class_='country-table')
        
        if not holiday_table:
            print("No holiday table found on the page.")
            return
        
        # Extract table rows
        rows = holiday_table.find_all('tr')
        
        print("Holidays in Sri Lanka:")
        print("=" * 40)
        
        # Iterate through rows and extract data
        for row in rows[1:]:  # Skip the header row
            columns = row.find_all('td')
            if len(columns) >= 2:  # Ensure there are enough columns
                day = columns[0].text.strip()  # Day of the event
                date = columns[1].text.strip()  # Date of the event
                holiday_name = columns[2].text.strip()  # Name of the event
                
                print(f"Day: {day}")
                print(f"Date: {date}")
                print(f"Event: {holiday_name}")
                print('-' * 40)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the URL: {e}")

# URL related to holidays
url_to_crawl = 'https://www.officeholidays.com/countries/sri-lanka/2025'
extract_holidays(url_to_crawl)
