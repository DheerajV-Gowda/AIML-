from browser_use_sdk import BrowserUseSdk

TODAY = "2025-08-30"

PEOPLE = [
    # ...
    {
        "name": "Dheeraj V Gowda",
        "location": "SF, CA",
        "loves": "Exploring new places like Slovenia.",
    },
]

sdk = BrowserUseSdk(api_key="bu_n6iVqfjPp8fdjrDTeC5aG1rmxkJ0-RiUliX5hlxRoN0")

for person in PEOPLE:
    result = sdk.run(
        llm_model="o3",
        task="""
Go to wikipedia.com and search for bread then go to agriculture from there and laterally navigate to the article on wheat,
later navigate to the article on gluten and then return to the agriculture article.
Finally, return to the wikipedia homepage and search for Slovenia. additionally, navigate to the article on Ljubljana.
""",
    )