import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

books = []

print("Starting web scraping...")

# Scrape all 50 pages
for page in range(1, 51):

    url = BASE_URL.format(page)

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Could not access page {page}")
        continue

    soup = BeautifulSoup(response.text, "html.parser")

    book_items = soup.select("article.product_pod")

    for book in book_items:

        # Book title
        title = book.h3.a["title"]

        # Price
        price = book.select_one(".price_color").text.strip()

        # Availability
        availability = book.select_one(".availability").text.strip()

        # Rating
        rating_element = book.select_one("p.star-rating")
        rating = rating_element.get("class")[1]

        rating_map = {
            "One": 1,
            "Two": 2,
            "Three": 3,
            "Four": 4,
            "Five": 5
        }

        rating = rating_map.get(rating, 0)

        # Book link
        relative_link = book.h3.a["href"]
        book_link = "https://books.toscrape.com/catalogue/" + relative_link.replace("../", "")

        books.append({
            "Title": title,
            "Price": price,
            "Rating": rating,
            "Availability": availability,
            "Book_Link": book_link
        })

    print(f"Page {page}/50 scraped successfully")

    # Small delay between requests
    time.sleep(0.2)


# Create DataFrame
df = pd.DataFrame(books)

# Save raw data
df.to_csv("raw_books.csv", index=False)

print("\nScraping completed!")
print(f"Total books collected: {len(df)}")
print("File saved as: raw_books.csv")

# Display first 10 records
print("\nFirst 10 records:")
print(df.head(10))