import os, sys

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
    url = f"https://guydar.substack.com/posts/{post_name}"
    post_dir = f"posts/{post_name}/"
    os.makedirs(post_dir, exist_ok=True)
    with open(os.path.join(post_dir, "index.html"), 'w') as f:
      f.write(template.format(url=url))