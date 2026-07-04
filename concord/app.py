"""
Concord desktop app entry point.

    python -m concord.app        # or: python run.py
"""
import os
import webview
from concord.api import ConcordAPI


def main():
    api = ConcordAPI()
    here = os.path.dirname(os.path.abspath(__file__))
    index = os.path.join(here, "ui", "index.html")

    window = webview.create_window(
        "Concord — Aligned XLIFF Consistency",
        index,
        js_api=api,
        width=1320,
        height=900,
        min_size=(1024, 680),
    )
    api.set_window(window)
    webview.start(debug=False)


if __name__ == "__main__":
    main()
