import requests
from bs4 import BeautifulSoup
import csv

# URL of the main event page
main_page_url = "https://www.eventbrite.com/d/sri-lanka--colombo/all-events/?page=1"

# Fetch the main page content
response = requests.get(main_page_url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all event links on the main page
    event_links = soup.find_all('a', class_='event-card-link')

    # a list to hold all event details
    event_data = []

    # a set to track already processed URLs
    processed_urls = set()

    # Extract details for each event
    for link in event_links:
        event_url = link.get('href', '')

        # Skip this event if it's already been processed
        if event_url in processed_urls:
            continue

        # Mark the event URL as processed
        processed_urls.add(event_url)

        # Extract event title and clean it 
        event_title = link.get('aria-label', 'No title')
        if event_title.lower().startswith("view "):
            event_title = event_title[5:].strip()  # Remove "View " 

        location = link.get('data-event-location', 'No location')
        paid_status = link.get('data-event-paid-status', 'No status')
        category = link.get('data-event-category', 'No category')

        # Fetch the detailed event page
        event_response = requests.get(event_url)
        if event_response.status_code == 200:
            event_soup = BeautifulSoup(event_response.text, 'html.parser')

            # Extract date and time
            date_time = event_soup.find('span', class_='date-info__full-datetime')
            date_time = date_time.get_text(strip=True) if date_time else "Date and time not found"

            # Extract location details
            location_address = event_soup.find('p', class_='location-info__address-text')
            location_address = location_address.get_text(strip=True) if location_address else "Location not found"

            # Append event details to the event_data list
            event_data.append([event_title, event_url, date_time, location_address, location, paid_status, category])

        else:
            print(f"Failed to fetch details for event: {event_url}")

    # Save event details into a CSV file
    with open('events.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Event Title", "Event URL", "Date and Time", "Location Address", "Location", "Paid Status", "Category"])
        # Write event data
        writer.writerows(event_data)

    print("Event details have been saved to 'events.csv'.")

else:
    print(f"Failed to fetch the main page. Status code: {response.status_code}")
