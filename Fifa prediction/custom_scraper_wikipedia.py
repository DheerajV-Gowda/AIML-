"""
custom_scraper_wikipedia.py
---------------------------
Educational web scraper for Task 1 of the FIFA World Cup Prediction Project.
Target: Wikipedia (https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_squads)
Purpose: Extract each team and count the number of players listed.
Output: data/raw/wiki_worldcup_squads_2022.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

def scrape_wikipedia_squads(year=2022):
    url = f"https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup_squads"
    print(f"Fetching squad data from {url} ...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    teams_data = []
    tables = soup.find_all("table", {"class": "wikitable"})

    for table in tables:
        header = table.find_previous(["h3", "h2"])
        if header:
            team_name = header.get_text(strip=True).replace("[edit]", "")
            players = len(table.find_all("tr")) - 1
            teams_data.append({"team": team_name, "players_listed": players, "year": year})

    df = pd.DataFrame(teams_data)
    os.makedirs("data/raw", exist_ok=True)
    output_path = "D:/aiml/assignment2/v3/data/raw/wiki_worldcup_squads_2022.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")
    print(df.head())
    return df


if __name__ == "__main__":
    df = scrape_wikipedia_squads()
    time.sleep(1)
