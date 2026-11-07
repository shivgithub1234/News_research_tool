# Streamlit deployment

The existing Community Cloud app tracks main and runs main.py. Install the root requirements.txt on Python 3.12 or later. The default mode is hosted. If overriding the mode in hosting secrets, set NEWS_RESEARCH_MODE="hosted".

Visitors supply their own free-tier Groq key in the password field. The server checks model access and opens the collection directory associated with the key. Disconnect clears the session. The host needs no shared Groq key or workspace password in hosted mode. Community Cloud local disk can reset. Back up important research separately.

To verify a release, open https://mynewsresearchtool.streamlit.app/ and look for **Open your research workspace**. Connect, process two public HTML articles, ask a question and inspect its citations. Select both articles and run a coverage comparison. Explicit demo mode remains available for the offline walkthrough.

## Private operation

Run `pip install -r requirements-private.txt` in a separate private environment. Configure `NEWS_RESEARCH_MODE=private`, a strong `APP_PASSWORD`, and a durable `NEWS_RESEARCH_DATA_DIR`. Use provider-level access restrictions. A public Community Cloud demo should not be treated as a durable private research database.

Groq generation additionally needs `GROQ_API_KEY` and the owner's `GROQ_FREE_TIER_CONFIRMED=true` assertion after checking the provider account. This does not change the billing plan or enforce a provider-side cap. Leave generation disabled if the plan cannot be verified. There is no paid fallback. No secrets are included in CI, samples, logs, or the repository.
