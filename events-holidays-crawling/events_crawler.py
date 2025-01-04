import asyncio
import json
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup

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

                extracted_events.append({
                    "Event Name": item.get("name"),
                    "Start Date": item.get("startDate"),
                    "End Date": item.get("endDate"),
                    "Event URL": item.get("url"),
                    "Place Name": physical_location.get("name"),
                    "Street Address": address.get("streetAddress"),
                    "Address Locality": address.get("addressLocality"),
                    "Address Region": address.get("addressRegion"),
                })
            
            import re

            # Print the extracted event details
            for event in extracted_events:
                # Run the crawler on the URL
                result = await crawler.arun(url=event["Event URL"])
                # Extract the raw HTML content
                html_content = result.html

                # Parse the HTML snippet with BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')

                # Attempt to locate the "Estimated Turnout" section
                turnout_element_h2 = soup.find('h2', string=lambda text: text and "Estimated Turnout" in text)

                # Extract the turnout value
                if turnout_element_h2:
                    # Check if the text is directly available after the <h2> tag
                    direct_text = turnout_element_h2.find_next_sibling(text=True)
                    if direct_text:
                        direct_text = direct_text.strip()
                        # Check if the text is a valid number or a range (e.g., "100 - 500")
                        if direct_text.isdigit():
                            estimated_turnout = direct_text
                        elif re.match(r"^\d+\s*-\s*\d+$", direct_text):  # Matches a range like "100 - 500"
                            estimated_turnout = direct_text
                        else:
                            estimated_turnout = None
                    else:
                        estimated_turnout = None

                    # If no valid direct text, look for the value in the next <a> tag
                    if not estimated_turnout:
                        turnout_element_value = turnout_element_h2.find_next('a')
                        estimated_turnout = turnout_element_value.get_text(strip=True) if turnout_element_value else "Not Found"

                    print("Estimated Turnout:", estimated_turnout)
                else:
                    print("Estimated Turnout not found.")

        except json.JSONDecodeError:
            print("Failed to parse JSON content.")

# Run the async main function
asyncio.run(main())
