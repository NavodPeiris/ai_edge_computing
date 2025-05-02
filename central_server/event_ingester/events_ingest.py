import asyncio
import json
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import re
import mysql.connector
import pandas as pd

# Database connection
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="rootpassword",
    database="mydb"
)

cursor = conn.cursor()

query = """
CREATE TABLE IF NOT EXISTS events (
    event_name VARCHAR(255),
    date DATETIME,
    location VARCHAR(500),
    estimated_visitors INT,
    PRIMARY KEY (event_name, date)
)
"""

cursor.execute(query)
conn.commit()

def convert_turnout_to_integer(turnout: str) -> int:
    """
    Converts a turnout string like '10.0k' or '100-500' to an integer.
    - For '10.0k', converts to 10000.
    - For '100-500', calculates and returns the median (e.g., 300).
    """
    if turnout.lower().endswith('k'):
        # Remove the 'k' and convert to float, then multiply by 1000
        return int(float(turnout[:-1]) * 1000)
    elif '-' in turnout:
        # Handle ranges like '100-500'
        try:
            low, high = map(int, turnout.split('-'))
            return (low + high) // 2  # Return the median as an integer
        except ValueError:
            return None  # Return None for invalid range formats
    elif turnout.isdigit():
        # If the value is already a digit, return it as an integer
        return int(turnout)
    return None  # Return None for invalid or unhandled cases

async def main():
    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler(verbose=True) as crawler:

        while True:
            # Run the crawler on the URL
            result = await crawler.arun(url="https://10times.com/sri-lanka")
            
            # Extract the raw HTML content
            html_content = result.html

            # Parse the JSON content
            try:
                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")

                # Find the script tag containing JSON data
                script_tag = soup.find("script", text=lambda t: t and "cache_eventData" in t)

                if script_tag:
                    # Extract JSON-like text from script
                    json_text = script_tag.string.split("var cache_eventData =")[-1].strip().rstrip(";")
                    
                    # Convert it into a Python dictionary
                    data = json.loads(json_text)
                    
                    # Extract event details
                    events = data.get("data", {}).get("events", [])

                    # Insert data
                    insert_query = """
                    INSERT INTO events (
                        `event_name`, `date`, `location`, `estimated_visitors`
                    ) VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                        `date` = VALUES(`date`),
                        `location` = VALUES(`location`),
                        `estimated_visitors` = VALUES(`estimated_visitors`);
                    """

                    unique_events = []
                    seen_event_names = set()
                    values = []

                    for event in events:
                        event_details = {
                            "Event Name": event.get("name", "Unknown Event") or "Not Found",
                            "Start Date": event.get("startDate", "N/A") or "Not Found",
                            "End Date": event.get("endDate", "N/A") or "Not Found",
                            "Place Name": event["location"].get("venueName", "N/A") or "Not Found",
                            "Street Address": event["location"].get("venueAddress", "N/A") or "Not Found",
                            "Address Locality": event["location"].get("cityName", "N/A") or "Not Found",
                            "Address Region": event["location"].get("state", "N/A") or "Not Found",
                            "Estimated Visitors": event["stats"].get("estimatedVisitors", 0) or "Not Found",
                        }

                        if event_details["Event Name"] not in seen_event_names:
                            unique_events.append(event_details)
                            seen_event_names.add(event_details["Event Name"])

                else:
                    print("No event data found in the HTML.")

                # Print the extracted event details
                print("Extracted New Events!")
                for event_details in unique_events:
                    if event_details["Event Name"] != "Not Found" \
                    and event_details["Start Date"] != "Not Found" \
                    and event_details["End Date"] != "Not Found" \
                    and event_details["Place Name"] != "Not Found" \
                    and event_details["Street Address"] != "Not Found" \
                    and event_details["Address Locality"] != "Not Found" \
                    and event_details["Address Region"] != "Not Found" \
                    and event_details["Estimated Visitors"] != "Not Found":
                        # Convert start and end date to datetime objects
                        start_date = pd.to_datetime(event_details["Start Date"])
                        end_date = pd.to_datetime(event_details["End Date"])

                        # Generate date range
                        date_range = pd.date_range(start=start_date, end=end_date)

                        # Append one record per date
                        for single_date in date_range:
                            values.append((
                                event_details["Event Name"],
                                single_date.strftime("%Y-%m-%d"),
                                event_details["Place Name"] + ", " + event_details["Address Locality"],
                                event_details["Estimated Visitors"],
                            ))

                cursor.executemany(insert_query, values)

                # Commit and close
                conn.commit()

                print("Data inserted successfully!")

            except Exception as e:
                print("Error:", e)


# Run the async main function
asyncio.run(main())
