import asyncio
import csv
from playwright.async_api import async_playwright

async def scrape_enam():
    url = "https://enam.gov.in/web/dashboard/trade-data"
    output_file = "enam_trade_data.csv"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # set False to watch
        page = await browser.new_page()
        await page.goto(url, timeout=60000)

        # Wait for table
        await page.wait_for_selector("table.table.table-striped.table-bordered")

        # Find total number of pages (dropdown options)
        page_options = await page.query_selector_all("#min_max_no_of_list option")
        total_pages = len(page_options)
        print(f"Found {total_pages} pages")

        data = []
        for i in range(total_pages):
            # Select page i
            await page.select_option("#min_max_no_of_list", str(i))
            await page.wait_for_selector("table.table.table-striped.table-bordered tbody tr")

            # Extract rows
            rows = await page.query_selector_all("table.table.table-striped.table-bordered tbody tr")

            for row in rows:
                cols = await row.query_selector_all("td")
                values = [await col.inner_text() for col in cols]

                if len(values) >= 10:  # ensure enough columns
                    state = values[0].strip()
                    apmc = values[1].strip()
                    commodity = values[2].strip()
                    min_price = values[3].strip()
                    modal_price = values[4].strip()
                    max_price = values[5].strip()
                    date = values[9].strip()

                    data.append([
                        state, apmc, commodity, min_price, modal_price, max_price, date
                    ])

            print(f"✅ Scraped page {i+1}/{total_pages}")

        # Save CSV
        headers = ["State", "APMC's", "Commodity", "Min Price", "Modal Price", "Max Price", "Date"]
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)

        print(f"🎉 Scraped {len(data)} rows across {total_pages} pages into {output_file}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_enam())
