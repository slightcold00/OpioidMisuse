
TRACK_TERMS = ["codeine", "fentanyl", "hydrocodone", "oxycodone", "Oxycontin", "Percocet","norco","Vicodin"]
CONNECTION_STRING = ""
CSV_NAME = "tweets.csv"
TABLE_NAME = "opioid"

try:
    from private import *
except Exception:
    pass