import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

#Fetches and displays all holiday dates and details from the specified URL.
def fetch_latest_data():
    # URL related to holidays
    current_year = datetime.now().year
    years = [
        current_year,
        current_year - 1,  # Previous year
        current_year - 2,  # Two years ago
        current_year - 3,  # Three years ago
        current_year - 4,  # Four years ago
        current_year - 5,  # Five years ago
    ]
    data = []

    for year in years:

        url = f'https://www.officeholidays.com/countries/sri-lanka/{year}'

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
                    # Convert to YYYY-MM-DD format
                    full_date = datetime.strptime(f"{date} {year}", "%b %d %Y").strftime("%Y-%m-%d")
                    holiday_name = columns[2].text.strip()  # Name of the event
                    data.append([full_date, holiday_name])

                    print(f"date: {full_date}")
                    print(f"holiday_name: {holiday_name}")
                    print('-' * 40)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching the URL: {e}")

    df = pd.DataFrame(data, columns=["date", "holiday_name"])
    """
    date: string yyyy-mm-dd
    holiday_name: string
    """

    return df

"""
fetch_latest_data()
"""
