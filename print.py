from playwright.sync_api import sync_playwright

url = "https://gs-demo.streamlit.app/"
output_pdf = "website.pdf"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()

    page.goto(url, wait_until="networkidle")

    page.pdf(
        path=output_pdf,
        format="A4",
        print_background=True
    )

    browser.close()

print(f"Saved to {output_pdf}")