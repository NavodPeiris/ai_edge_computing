import asyncio
import json
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import re

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

        # Locate and extract the JSON content from <script type="application/ld+json">
        start_idx = html_content.find('<script type="application/ld+json">')
        end_idx = html_content.find('</script>', start_idx)
        if start_idx == -1 or end_idx == -1:
            print("No JSON-LD content found.")
            return
        
        json_content = html_content[start_idx + len('<script type="application/ld+json">'):end_idx].strip()

        # Parse the JSON content
        try:
            data = json.loads(json_content)
            events = data.get("itemListElement", [])
            
            # Extract relevant details from each event
            extracted_events = []
            for event_entry in events:
                item = event_entry.get("item", {})
                location = item.get("location", [])
                
                # Extract the first physical location details (if available)
                physical_location = next(
                    (loc for loc in location if loc["@type"] == "Place"), {}
                )
                address = physical_location.get("address", {})

                event_details = {
                    "Event Name": item.get("name"),
                    "Start Date": item.get("startDate"),
                    "End Date": item.get("endDate"),
                    "Event URL": item.get("url"),
                    "Place Name": physical_location.get("name"),
                    "Street Address": address.get("streetAddress"),
                    "Address Locality": address.get("addressLocality"),
                    "Address Region": address.get("addressRegion"),
                }

                # Fetch the event page to find "Estimated Turnout"
                if event_details["Event URL"]:
                    turnout_result = await crawler.arun(url=event_details["Event URL"])
                    turnout_html_content = turnout_result.html

                    # Parse the HTML snippet with BeautifulSoup
                    soup = BeautifulSoup(turnout_html_content, 'html.parser')

                    # Attempt to locate the "Estimated Turnout" section
                    turnout_element_h2 = soup.find('h2', string=lambda text: text and "Estimated Turnout" in text)

                    # Extract the turnout value
                    estimated_turnout = None
                    if turnout_element_h2:
                        # Check if the text is directly available after the <h2> tag
                        direct_text = turnout_element_h2.find_next_sibling(text=True)
                        if direct_text:
                            direct_text = direct_text.strip()
                            # Check if the text is a valid number or a range (e.g., "100 - 500")
                            if direct_text.isdigit():
                                estimated_turnout = convert_turnout_to_integer(direct_text)
                            elif re.match(r"^\d+\s*-\s*\d+$", direct_text):  # Matches a range like "100 - 500"
                                estimated_turnout = convert_turnout_to_integer(direct_text)
                            elif direct_text.lower().endswith('k'):
                                estimated_turnout = convert_turnout_to_integer(direct_text)

                        # If no valid direct text, look for the value in the next <a> tag
                        if not estimated_turnout:
                            turnout_element_value = turnout_element_h2.find_next('a')
                            if turnout_element_value:
                                turnout_text = turnout_element_value.get_text(strip=True)
                                estimated_turnout = convert_turnout_to_integer(turnout_text)

                    event_details["Estimated Turnout"] = estimated_turnout or "Not Found"
                
                extracted_events.append(event_details)

            # Print the extracted event details with "Estimated Turnout"
            for event in extracted_events:
                print(event)

        except json.JSONDecodeError:
            print("Failed to parse JSON content.")

# Run the async main function
asyncio.run(main())
