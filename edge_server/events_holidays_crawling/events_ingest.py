import asyncio
import json
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import re
import mysql.connector

# Database connection
conn = mysql.connector.connect(
    host="localhost",  # or use "mysql-fyp" if running inside a Docker container
    user="root",
    password="rootpassword",
    database="mydb"
)

cursor = conn.cursor()

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
                    `event_name`, `start_date`, `end_date`, `place_name`, 
                    `street_address`, `address_locality`, `address_region`, `estimated_visitors`
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    `start_date` = VALUES(`start_date`),
                    `end_date` = VALUES(`end_date`),
                    `place_name` = VALUES(`place_name`),
                    `street_address` = VALUES(`street_address`),
                    `address_locality` = VALUES(`address_locality`),
                    `address_region` = VALUES(`address_region`),
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
                    values.append((
                        event_details["Event Name"],
                        event_details["Start Date"],
                        event_details["End Date"],
                        event_details["Place Name"],
                        event_details["Street Address"],
                        event_details["Address Locality"],
                        event_details["Address Region"],
                        event_details["Estimated Visitors"]
                    ))

            cursor.executemany(insert_query, values)

            # Commit and close
            conn.commit()
            cursor.close()
            conn.close()

            print("Data inserted successfully!")

        except Exception as e:
            print("Error:", e)


# Run the async main function
asyncio.run(main())
