# Performance Audit Studio (Streamlit)

A colorful Streamlit app that:

1. Accepts a client URL.
2. Discovers sitemaps via `robots.txt` and common sitemap endpoints.
3. Extracts all URLs from sitemap index/urlset files.
4. Runs Google PageSpeed Insights API audits in parallel.
5. Lets users customize report scope and output using UI toggles, switches, and checkboxes.

## Features

- **Device strategy controls**: mobile and/or desktop.
- **Category controls**: performance, accessibility, best-practices, SEO.
- **Field data toggle**: include/exclude CrUX field metrics.
- **Sitemap-only toggle**: optionally allow fallback same-site crawl if sitemaps are unavailable.
- **Fine-tuning filters**:
  - Include URL regex
  - Exclude URL regex
  - Minimum performance score
- **Parallelism and reliability controls**:
  - max URLs
  - worker count
  - retries
  - timeout
- **Downloads**: CSV and JSON reports.

## API Research Notes (implemented as UI controls)

This app is built around the PageSpeed Insights v5 endpoint:

- `https://pagespeedonline.googleapis.com/pagespeedonline/v5/runPagespeed`

Exposed user controls map to supported API request parameters:

- `url`
- `strategy` (`mobile` / `desktop`)
- `category` (`performance`, `accessibility`, `best-practices`, `seo`)
- `locale`
- `key`

The app also surfaces response-derived values such as Lighthouse scores and lab metrics, with optional field data columns.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional API key:

```bash
export PSI_API_KEY="your_google_pagespeed_api_key"
```

## Run

```bash
streamlit run app.py
```

## Security

Do **not** hard-code API keys in source files. Use environment variables or Streamlit secrets.
