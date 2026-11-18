import os

from config import load_dotenv_file

# Local development convenience: read ./.env if it exists (real environment variables win).
load_dotenv_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from backend import create_app  # noqa: E402  (must come after the .env is loaded)

app = create_app()

if __name__ == "__main__":
    # Local development only. In production Gunicorn serves `app:app`.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
