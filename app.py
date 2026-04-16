import concurrent.futures
import gzip
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
import streamlit as st

PSI_ENDPOINT = "https://pagespeedonline.googleapis.com/pagespeedonline/v5/runPagespeed"
DEFAULT_HEADERS = {"User-Agent": "performance-audit-streamlit/1.0", "Accept": "*/*"}


@dataclass
class AuditConfig:
    api_key: str
    strategies: List[str]
    categories: List[str]
    locale: str
    max_urls: int
    timeout_seconds: int
    max_workers: int
    retries: int
    include_field_data: bool


def normalize_base_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    if not raw_url:
        raise ValueError("Please enter a URL.")
    if not re.match(r"^https?://", raw_url, flags=re.IGNORECASE):
        raw_url = f"https://{raw_url}"
    parsed = urllib.parse.urlparse(raw_url)
    if not parsed.netloc:
        raise ValueError("Invalid URL. Please provide a full domain like https://example.com")
    return f"{parsed.scheme}://{parsed.netloc}"


def _request(url: str, timeout_seconds: int) -> requests.Response:
    return requests.get(url, timeout=timeout_seconds, headers=DEFAULT_HEADERS)


def discover_sitemaps(base_url: str, timeout_seconds: int = 20) -> List[str]:
    found: Set[str] = set()

    robots_url = f"{base_url.rstrip('/')}/robots.txt"
    try:
        r = _request(robots_url, timeout_seconds)
        if r.ok:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    if sitemap_url:
                        found.add(sitemap_url)
    except requests.RequestException:
        pass

    common_paths = ["/sitemap.xml", "/sitemap_index.xml", "/wp-sitemap.xml"]
    for path in common_paths:
        candidate = f"{base_url.rstrip('/')}{path}"
        try:
            r = _request(candidate, timeout_seconds)
            if r.ok and "xml" in r.headers.get("Content-Type", "").lower():
                found.add(candidate)
        except requests.RequestException:
            continue

    return sorted(found)


def _load_xml_bytes(url: str, timeout_seconds: int) -> bytes:
    r = _request(url, timeout_seconds)
    r.raise_for_status()
    content = r.content
    if url.endswith(".gz") or r.headers.get("Content-Encoding", "").lower() == "gzip":
        try:
            return gzip.decompress(content)
        except OSError:
            return content
    return content


def parse_sitemap_urls(sitemap_url: str, timeout_seconds: int = 20, visited: Optional[Set[str]] = None) -> List[str]:
    if visited is None:
        visited = set()
    if sitemap_url in visited:
        return []
    visited.add(sitemap_url)

    try:
        data = _load_xml_bytes(sitemap_url, timeout_seconds)
        root = ET.fromstring(data)
    except Exception:
        return []

    ns_match = re.match(r"\{(.*)\}", root.tag)
    ns = {"sm": ns_match.group(1)} if ns_match else {}

    urls: List[str] = []

    if root.tag.endswith("sitemapindex"):
        path = ".//sm:sitemap/sm:loc" if ns else ".//sitemap/loc"
        for loc_node in root.findall(path, ns):
            child_url = (loc_node.text or "").strip()
            if child_url:
                urls.extend(parse_sitemap_urls(child_url, timeout_seconds, visited))
    elif root.tag.endswith("urlset"):
        path = ".//sm:url/sm:loc" if ns else ".//url/loc"
        for loc_node in root.findall(path, ns):
            u = (loc_node.text or "").strip()
            if u:
                urls.append(u)

    return urls


def crawl_fallback_urls(base_url: str, timeout_seconds: int, max_urls: int) -> List[str]:
    """Very light same-origin fallback crawl from homepage links if sitemap is unavailable."""
    visited: Set[str] = set()
    queue: List[str] = [f"{base_url.rstrip('/')}/"]

    while queue and len(visited) < max_urls:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        try:
            r = _request(current, timeout_seconds)
            if not r.ok or "html" not in r.headers.get("Content-Type", "").lower():
                continue
            html = r.text
        except requests.RequestException:
            continue

        for match in re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE):
            normalized = urllib.parse.urljoin(current, match)
            parsed = urllib.parse.urlparse(normalized)
            if parsed.scheme not in {"http", "https"}:
                continue
            if f"{parsed.scheme}://{parsed.netloc}" != base_url:
                continue
            clean = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", "", "", ""))
            if clean not in visited and clean not in queue:
                queue.append(clean)

            if len(queue) + len(visited) >= max_urls:
                break

    return sorted(visited)


def score_100(categories: Dict, key: str) -> Optional[float]:
    score = categories.get(key, {}).get("score")
    return round(score * 100, 1) if isinstance(score, (int, float)) else None


def metric(audits: Dict, key: str, divisor: int = 1) -> Optional[float]:
    value = audits.get(key, {}).get("numericValue")
    return round(value / divisor, 2) if isinstance(value, (int, float)) else None


def get_nested(obj: Dict, path: Iterable[str], default=None):
    cur = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def fetch_one(url: str, strategy: str, config: AuditConfig) -> Dict:
    params: List[Tuple[str, str]] = [("url", url), ("strategy", strategy), ("locale", config.locale)]
    for cat in config.categories:
        params.append(("category", cat))
    if config.api_key:
        params.append(("key", config.api_key))

    last_error = None
    for attempt in range(1, config.retries + 1):
        try:
            resp = requests.get(PSI_ENDPOINT, params=params, timeout=config.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()

            lighthouse = data.get("lighthouseResult", {})
            categories = lighthouse.get("categories", {})
            audits = lighthouse.get("audits", {})
            loading_exp = data.get("loadingExperience", {})

            row = {
                "url_requested": url,
                "url_final": lighthouse.get("finalDisplayedUrl") or data.get("id") or url,
                "strategy": strategy,
                "fetch_time_utc": data.get("analysisUTCTimestamp"),
                "performance_score": score_100(categories, "performance"),
                "accessibility_score": score_100(categories, "accessibility"),
                "best_practices_score": score_100(categories, "best-practices"),
                "seo_score": score_100(categories, "seo"),
                "fcp_seconds": metric(audits, "first-contentful-paint", 1000),
                "lcp_seconds": metric(audits, "largest-contentful-paint", 1000),
                "speed_index_seconds": metric(audits, "speed-index", 1000),
                "tbt_ms": metric(audits, "total-blocking-time", 1),
                "cls": metric(audits, "cumulative-layout-shift", 1),
                "tti_seconds": metric(audits, "interactive", 1000),
                "inp_ms_lab": metric(audits, "interaction-to-next-paint", 1),
                "error": None,
            }

            if config.include_field_data:
                row.update(
                    {
                        "field_overall_category": loading_exp.get("overall_category"),
                        "field_lcp_category": get_nested(
                            loading_exp, ["metrics", "LARGEST_CONTENTFUL_PAINT_MS", "category"]
                        ),
                        "field_cls_category": get_nested(
                            loading_exp, ["metrics", "CUMULATIVE_LAYOUT_SHIFT_SCORE", "category"]
                        ),
                        "field_inp_category": get_nested(
                            loading_exp, ["metrics", "INTERACTION_TO_NEXT_PAINT", "category"]
                        ),
                    }
                )
            return row
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            time.sleep(min(attempt * 1.5, 4))

    return {
        "url_requested": url,
        "url_final": None,
        "strategy": strategy,
        "fetch_time_utc": None,
        "error": last_error,
    }


def run_audit(urls: List[str], config: AuditConfig, progress_placeholder) -> pd.DataFrame:
    tasks = [(u, s) for u in urls for s in config.strategies]
    rows: List[Dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = [pool.submit(fetch_one, url, strategy, config) for url, strategy in tasks]
        total = len(futures)
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rows.append(future.result())
            progress_placeholder.progress(i / total, text=f"Auditing {i}/{total} URL-strategy combinations")

    df = pd.DataFrame(rows)
    if not df.empty and "performance_score" in df.columns:
        df = df.sort_values(["performance_score", "lcp_seconds", "url_requested"], ascending=[True, False, True])
    return df


def apply_custom_filters(df: pd.DataFrame, include_pattern: str, exclude_pattern: str, min_perf: int) -> pd.DataFrame:
    out = df.copy()
    if include_pattern:
        out = out[out["url_requested"].str.contains(include_pattern, case=False, na=False, regex=True)]
    if exclude_pattern:
        out = out[~out["url_requested"].str.contains(exclude_pattern, case=False, na=False, regex=True)]
    if "performance_score" in out.columns:
        out = out[out["performance_score"].fillna(-1) >= min_perf]
    return out.reset_index(drop=True)


def render():
    st.set_page_config(page_title="Performance Audit Studio", page_icon="🚀", layout="wide")

    st.markdown(
        """
        <style>
            .stApp {background: linear-gradient(135deg, #f8fbff 0%, #f3efff 40%, #e9fff7 100%);} 
            .block-container {padding-top: 1.2rem;}
            h1, h2, h3 {color: #2D1B69;}
            [data-testid="stSidebar"] {background: linear-gradient(180deg, #2D1B69 0%, #4D2BA7 100%);} 
            [data-testid="stSidebar"] * {color: #ffffff !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🚀 Performance Audit Studio")
    st.caption("Sitemap-driven PageSpeed Insights auditing with customizable report controls.")

    with st.expander("API capabilities used in this app", expanded=False):
        st.markdown(
            """
            This app uses **PageSpeed Insights API v5** (`runPagespeed`) and exposes controls for the supported query parameters:
            - `strategy`: `mobile` and/or `desktop`
            - `category`: `performance`, `accessibility`, `best-practices`, `seo`
            - `locale`: localization code for formatted output
            - `url`: page URL being audited
            - `key`: your API key (optional in UI, recommended for quota)

            The report table is additionally customizable client-side with URL include/exclude patterns and performance score thresholds.
            """
        )

    with st.sidebar:
        st.header("⚙️ Audit Controls")
        api_key = st.text_input("PageSpeed API Key", value=os.getenv("PSI_API_KEY", ""), type="password")
        client_url = st.text_input("Client Website URL", placeholder="https://example.com")

        strategies = st.multiselect("Device Strategy", ["mobile", "desktop"], default=["mobile"])
        categories = st.multiselect(
            "Lighthouse Categories",
            ["performance", "accessibility", "best-practices", "seo"],
            default=["performance", "accessibility", "best-practices", "seo"],
        )
        locale = st.text_input("Locale", value="en-US")

        include_field_data = st.toggle("Include CrUX field-data columns", value=True)
        sitemap_only = st.toggle("Use sitemap URLs only (no fallback crawl)", value=True)

        max_urls = st.slider("Max URLs to audit", min_value=5, max_value=500, value=50, step=5)
        timeout_seconds = st.slider("Request timeout (seconds)", min_value=10, max_value=180, value=60, step=5)
        max_workers = st.slider("Parallel workers", min_value=1, max_value=10, value=4)
        retries = st.slider("Retries per request", min_value=1, max_value=5, value=2)

        run_clicked = st.button("▶️ Discover URLs & Run Audit", type="primary", use_container_width=True)

    st.subheader("Report Filters")
    c1, c2, c3 = st.columns(3)
    include_pattern = c1.text_input("Include URL regex")
    exclude_pattern = c2.text_input("Exclude URL regex")
    min_perf = c3.slider("Minimum performance score", 0, 100, 0)

    if run_clicked:
        if not strategies:
            st.error("Select at least one strategy.")
            return
        if not categories:
            st.error("Select at least one category.")
            return

        try:
            base_url = normalize_base_url(client_url)
        except ValueError as err:
            st.error(str(err))
            return

        st.info(f"Starting discovery for **{base_url}**")

        sitemaps = discover_sitemaps(base_url, timeout_seconds=timeout_seconds)
        all_urls: List[str] = []

        if sitemaps:
            st.success(f"Discovered {len(sitemaps)} sitemap(s)")
            with st.expander("View discovered sitemap URLs"):
                st.write(sitemaps)
            for sm in sitemaps:
                all_urls.extend(parse_sitemap_urls(sm, timeout_seconds=timeout_seconds))

        unique_urls = sorted({u for u in all_urls if u.startswith(("http://", "https://"))})

        if not unique_urls and not sitemap_only:
            st.warning("No sitemap URLs found. Running fallback same-site crawl from homepage.")
            unique_urls = crawl_fallback_urls(base_url, timeout_seconds=timeout_seconds, max_urls=max_urls)

        if not unique_urls:
            st.error("No URLs discovered. Try disabling sitemap-only mode or verify robots/sitemap accessibility.")
            return

        unique_urls = unique_urls[:max_urls]
        st.write(f"URLs queued for audit: **{len(unique_urls)}**")

        config = AuditConfig(
            api_key=api_key.strip(),
            strategies=strategies,
            categories=categories,
            locale=locale.strip() or "en-US",
            max_urls=max_urls,
            timeout_seconds=timeout_seconds,
            max_workers=max_workers,
            retries=retries,
            include_field_data=include_field_data,
        )

        progress = st.progress(0.0, text="Preparing...")
        df = run_audit(unique_urls, config, progress)

        if df.empty:
            st.error("Audit returned no rows.")
            return

        filtered = apply_custom_filters(df, include_pattern, exclude_pattern, min_perf)

        k1, k2, k3 = st.columns(3)
        k1.metric("Rows", len(filtered))
        k2.metric("Avg Performance", round(filtered["performance_score"].dropna().mean(), 1) if "performance_score" in filtered else "n/a")
        k3.metric("Errors", int(filtered["error"].notna().sum() if "error" in filtered.columns else 0))

        st.dataframe(filtered, use_container_width=True, height=500)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        json_bytes = filtered.to_json(orient="records", indent=2).encode("utf-8")

        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Download CSV report",
            data=csv_bytes,
            file_name=f"psi_audit_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        d2.download_button(
            "⬇️ Download JSON report",
            data=json_bytes,
            file_name=f"psi_audit_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    render()
