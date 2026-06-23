import subprocess

url = "https://gs-demo.streamlit.app/"

subprocess.run([
    "chrome",
    "--headless",
    "--disable-gpu",
    "--print-to-pdf=website.pdf",
    url
])