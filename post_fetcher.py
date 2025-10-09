import os, sys

import requests

def check_url_exists(url):
    """
    Checks if a given URL exists and is accessible.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL exists and returns a 2xx status code, False otherwise.
    """
    try:
        response = requests.head(url, timeout=5)  # Use HEAD request for efficiency
        # A 2xx status code indicates success
        return 200 <= response.status_code < 300
    except requests.exceptions.RequestException as e:
        # Handle various request-related errors (connection, timeout, etc.)
        print(f"Error checking URL {url}: {e}")
        return False

template = """<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="refresh" content="0; url={url}">
  <link rel="canonical" href="{url}" />
</head>
<body>
  <p>Redirecting to <a href="{url}">{url}</a>...</p>
</body>
</html>
"""

if __name__ == "__main__":
    post_name = sys.argv[1]
    url = f"https://guydar.substack.com/p/{post_name}"
    # if check_url_exists(url):
    #   print("URL exists :)")
    # else:
    #   print("URL is wrong :(")
    #   exit()
    post_dir = f"posts/{post_name}/"
    os.makedirs(post_dir, exist_ok=True)
    with open(os.path.join(post_dir, "index.html"), 'w') as f:
      f.write(template.format(url=url))