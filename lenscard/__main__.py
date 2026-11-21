"""Run the local workbench."""
import os

from waitress import serve

from .http import create_app


def main():
    serve(
        create_app(),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "19957")),
        threads=3,
        max_request_body_size=144 * 1024**2,
    )


if __name__ == "__main__":
    main()
