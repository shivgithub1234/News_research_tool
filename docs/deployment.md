# Release and deployment

The target is the existing Streamlit Community Cloud app. The new default is a public read-only demo: `main.py`, Python 3.12, minimal root `requirements.txt`, no Groq secret, no model download, and no private data access. `NEWS_RESEARCH_MODE=demo` can be set explicitly. Keep private credentials out of a public demo.

## Owner release checkpoint

1. Review the development branch and its GitHub Actions run. The branch is `codex/organize-stage-one` in `shivgithub1234/News_research_tool`.
2. Sign in to the existing Streamlit Community Cloud workspace. Confirm the app's configured repository, branch and entry point. Use `main.py` and Python 3.12. If an existing app tracks `main` and its branch cannot be changed in place, merge the reviewed branch through a pull request, or deploy a separate preview before switching production. Do not delete the existing app just to change branches.
3. Keep demo mode enabled. No Groq or Hugging Face credentials are necessary. Redeploy or reboot as appropriate and inspect build logs.
4. Verify the title is **News Research Workspace**, the banner says **Free offline demo**, and **Find evidence** returns dated Harbor Bank passages. The old MVP title is not a successful release of this version.
5. For rollback, restore the previously deployed commit/branch through the same release path. Do not delete private data directories during code rollback.

The existing public URL was observed serving the original app during development. The dashboard required sign-in; a repository push alone does not verify a production deployment.

## Private operation

Run `pip install -r requirements-private.txt` in a separate private environment. Configure `NEWS_RESEARCH_MODE=private`, a strong `APP_PASSWORD`, and a durable `NEWS_RESEARCH_DATA_DIR`. Use provider-level access restrictions. A public Community Cloud demo should not be treated as a durable private research database.

Groq generation additionally needs `GROQ_API_KEY` and the owner's `GROQ_FREE_TIER_CONFIRMED=true` assertion after checking the provider account. This does not change the billing plan or enforce a provider-side cap. Leave generation disabled if the plan cannot be verified. There is no paid fallback. No secrets are included in CI, samples, logs, or the repository.
