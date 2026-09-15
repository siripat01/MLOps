from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request


def main() -> None:
    url = "http://127.0.0.1:8080/api/v1/info"
    deadline = time.time() + 120

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(2)

    print(f"ZenML server did not become ready at {url}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
