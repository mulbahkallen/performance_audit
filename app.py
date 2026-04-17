import concurrent.futures
import gzip
import hashlib
import io
import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI

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


SCHEMA_SKIP_PATTERNS = ["/wp-json/", "/feed/", "/tag/", "/author/", "/search"]
SCHEMA_CACHE_VERSION = "v3_remove_jobtitle_and_fix_provider_types"
PHYSICIAN_CREDENTIAL_RE = re.compile(r"\b(md|m\.d\.|do|d\.o\.|dpm|mbbs|mbchb)\b", re.I)
PHYSICIAN_ROLE_RE = re.compile(r"\b(physician|surgeon|doctor|dr\.)\b", re.I)
NON_PHYSICIAN_RE = re.compile(
    r"\b(rn|r\.n\.|np|n\.p\.|fnp|dnp|pa|pa-c|aprn|msn|bsn|cac|esthetician|aesthetician|nurse injector|registered nurse|nurse practitioner|physician assistant|injector)\b",
    re.I,
)


def stable_slug(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip("/") or "home"
    path = re.sub(r"[^a-zA-Z0-9_-]+", "_", path)[:100]
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    return f"{path}_{digest}"


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def should_skip_schema_url(url: str) -> bool:
    low = url.lower()
    return any(pattern in low for pattern in SCHEMA_SKIP_PATTERNS)


def extract_text_list(nodes, limit=20, max_len=300) -> List[str]:
    out: List[str] = []
    for node in nodes[:limit]:
        text = " ".join(node.get_text(" ", strip=True).split())
        if text:
            out.append(text[:max_len])
    return out


def classify_page(url: str, title: str, h1: str) -> str:
    u = url.lower()
    t = (title or "").lower()
    h = (h1 or "").lower()
    site_root = normalize_base_url(url)

    if url.rstrip("/") == site_root:
        return "homepage"
    if "/conditions/" in u or "condition" in t or "condition" in h:
        return "condition_page"
    if "/services/" in u or "treatment" in t or "service" in h:
        return "service_page"
    if "/faq" in u or "faq" in t or "faq" in h:
        return "faq_page"
    if "/blog/" in u or "/articles/" in u or "article" in t:
        return "article_page"
    if "doctor" in u or "provider" in u or "meet-" in u or "bio" in u:
        return "provider_page"
    return "general_page"


def extract_provider_mentions(full_text: str) -> List[str]:
    matches: List[str] = []
    patterns = [
        r"\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3},\s*(?:MD|M\.D\.|DO|D\.O\.|DPM|MBBS|MBChB|RN|R\.N\.|NP|N\.P\.|PA|PA-C|FNP|DNP|MSN|BSN|CAC)\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, full_text):
            matches.append(match.strip())
    return unique_keep_order(matches)[:20]


def extract_page_summary(url: str, html: str) -> Dict:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    meta_description = ""
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        meta_description = md["content"].strip()

    canonical = ""
    canon = soup.find("link", attrs={"rel": "canonical"})
    if canon and canon.get("href"):
        canonical = canon["href"].strip()

    h1 = extract_text_list(soup.find_all("h1"), limit=3)
    h2 = extract_text_list(soup.find_all("h2"), limit=15)
    body = soup.find("main") or soup.find("article") or soup.body
    content_snippets = extract_text_list(body.find_all(["p", "li"]) if body else [], limit=40)

    phones = [a["href"].replace("tel:", "").strip() for a in soup.find_all("a", href=True) if a["href"].strip().startswith("tel:")]

    social_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if any(x in href.lower() for x in ["instagram.com", "facebook.com", "youtube.com", "tiktok.com", "linkedin.com", "twitter.com", "x.com"]):
            social_links.append(href)

    breadcrumb_text: List[str] = []
    for node in soup.find_all(attrs={"class": re.compile("breadcrumb", re.I)}):
        breadcrumb_text.extend(extract_text_list(node.find_all(["a", "li", "span"]), limit=20))

    faq_questions = []
    for node in soup.find_all(["h2", "h3", "summary", "button"]):
        txt = " ".join(node.get_text(" ", strip=True).split())
        if "?" in txt and 2 < len(txt) < 220:
            faq_questions.append(txt)

    all_text = " ".join(soup.get_text(" ", strip=True).split())
    provider_mentions = extract_provider_mentions(all_text)

    internal_links = []
    site_root = normalize_base_url(url)
    for a in soup.find_all("a", href=True):
        full = urllib.parse.urljoin(site_root, a["href"].strip())
        if full.startswith(site_root):
            internal_links.append(full)

    return {
        "page_url": url,
        "site_root": site_root,
        "canonical": canonical,
        "page_type_guess": classify_page(url, title, h1[0] if h1 else ""),
        "title": title,
        "meta_description": meta_description,
        "h1": h1,
        "h2": h2,
        "content_snippets": content_snippets[:25],
        "phone_numbers": unique_keep_order(phones),
        "social_links": unique_keep_order(social_links)[:20],
        "breadcrumb_text": unique_keep_order(breadcrumb_text)[:20],
        "faq_questions": unique_keep_order(faq_questions)[:20],
        "provider_mentions": provider_mentions,
        "internal_links_sample": unique_keep_order(internal_links)[:40],
    }


def build_schema_prompt(summary: Dict) -> str:
    return f"""
You are an expert in Schema.org structured data and technical SEO.

Return exactly ONE valid JSON object.
Do not use markdown.
Do not use code fences.
Do not return explanations.

The top-level JSON object must have exactly these keys:
- "page_url"
- "schema_jsonld"

Rules for "schema_jsonld":
- It must be a valid JSON object, not a string
- Use "@context": "https://schema.org"
- Prefer "@graph"
- Use absolute URLs and stable "@id" values
- Keep output minimal and valid
- Use the most specific valid page/content types supported by the page
- Connect entities with:
  - WebPage -> isPartOf -> WebSite
  - WebPage -> about -> clinic/business
  - WebPage -> mainEntity -> primary topics or provider
- Do NOT mix business and personal social profiles

Provider rules:
- Never use "Physician" for a human provider node
- Use "IndividualPhysician" only when clearly supported by credentials/role like MD, DO, DPM, MBBS, MBChB, physician, surgeon, or doctor
- Use "Person" for RN, NP, PA, PA-C, FNP, DNP, injector, aesthetician, esthetician, CAC, or any non-physician role
- If uncertain, use "Person"
- For "IndividualPhysician", use "practicesAt"
- For "Person", use "worksFor"
- Never use "jobTitle" on "IndividualPhysician"
- "jobTitle" may be used on "Person" only
- Do not use "MD" as "jobTitle"

Do not invent unsupported credentials, specialties, or people.

Page summary:
{json.dumps(summary, ensure_ascii=False)}
""".strip()


def ensure_graph(schema_obj: Dict, page_url: str) -> Dict:
    if not isinstance(schema_obj, dict):
        schema_obj = {}
    if "@graph" not in schema_obj:
        if schema_obj.get("@type"):
            schema_obj = {"@context": "https://schema.org", "@graph": [schema_obj]}
        else:
            schema_obj = {"@context": "https://schema.org", "@graph": []}
    if not isinstance(schema_obj["@graph"], list):
        schema_obj["@graph"] = []
    schema_obj["@context"] = "https://schema.org"
    return schema_obj


def find_clinic_id(graph: List[Dict], page_url: str) -> str:
    for node in graph:
        if node.get("@type") in ["MedicalClinic", "MedicalBusiness", "MedicalOrganization"] and node.get("@id"):
            return node["@id"]
    return normalize_base_url(page_url) + "/#clinic"


def make_default_ids(graph: List[Dict], page_url: str):
    site_root = normalize_base_url(page_url)
    for node in graph:
        ntype = node.get("@type")
        if ntype == "WebPage":
            node.setdefault("@id", page_url.rstrip("/") + "#webpage")
            node.setdefault("url", page_url)
        elif ntype == "WebSite":
            node.setdefault("@id", site_root + "/#website")
            node.setdefault("url", site_root + "/")
        elif ntype in ["MedicalClinic", "MedicalBusiness", "MedicalOrganization"]:
            node.setdefault("@id", site_root + "/#clinic")
            node.setdefault("url", site_root + "/")
        elif ntype in ["Person", "IndividualPhysician", "Physician"]:
            name = node.get("name", "provider")
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-") or "provider"
            node.setdefault("@id", site_root + f"/#{slug}")


def infer_provider_kind(node: Dict) -> str:
    fields: List[str] = []
    for key in ["name", "jobTitle", "description", "headline", "medicalSpecialty"]:
        value = node.get(key)
        if isinstance(value, str):
            fields.append(value)
        elif isinstance(value, list):
            fields.extend([str(x) for x in value])
    combined = " ".join(fields).strip()
    if PHYSICIAN_CREDENTIAL_RE.search(combined):
        return "individual_physician"
    if NON_PHYSICIAN_RE.search(combined):
        return "person"
    if PHYSICIAN_ROLE_RE.search(combined):
        return "individual_physician"
    return "person"


def sanitize_provider_nodes(graph: List[Dict], page_url: str):
    clinic_id = find_clinic_id(graph, page_url)
    for node in graph:
        if node.get("@type") in ["Physician", "IndividualPhysician", "Person"]:
            inferred = infer_provider_kind(node)
            if inferred == "individual_physician":
                node["@type"] = "IndividualPhysician"
                if "worksFor" in node and "practicesAt" not in node:
                    node["practicesAt"] = node["worksFor"]
                node.setdefault("practicesAt", {"@id": clinic_id})
                node.pop("worksFor", None)
                node.pop("jobTitle", None)
            else:
                node["@type"] = "Person"
                if "practicesAt" in node and "worksFor" not in node:
                    node["worksFor"] = node["practicesAt"]
                node.setdefault("worksFor", {"@id": clinic_id})
                node.pop("practicesAt", None)
        if node.get("@type") in ["IndividualPhysician", "Physician"]:
            node.pop("jobTitle", None)


def ensure_required_core_nodes(graph: List[Dict], summary: Dict):
    site_root = summary["site_root"]
    page_url = summary["page_url"]
    has_webpage = any(node.get("@type") == "WebPage" for node in graph)
    has_website = any(node.get("@type") == "WebSite" for node in graph)
    has_clinic = any(node.get("@type") in ["MedicalClinic", "MedicalBusiness", "MedicalOrganization"] for node in graph)

    if not has_website:
        graph.append({"@type": "WebSite", "@id": site_root + "/#website", "url": site_root + "/", "name": summary["title"] or urllib.parse.urlparse(site_root).netloc})
    if not has_clinic and (summary["phone_numbers"] or summary["social_links"]):
        graph.append({"@type": "MedicalClinic", "@id": site_root + "/#clinic", "url": site_root + "/", "name": summary["title"] or urllib.parse.urlparse(site_root).netloc})
    if not has_webpage:
        graph.append({"@type": "WebPage", "@id": page_url.rstrip("/") + "#webpage", "url": page_url, "name": summary["title"] or page_url, "inLanguage": "en"})


def connect_core_nodes(graph: List[Dict], page_url: str):
    site_root = normalize_base_url(page_url)
    webpage_id = page_url.rstrip("/") + "#webpage"
    website_id = site_root + "/#website"
    clinic_id = find_clinic_id(graph, page_url)
    for node in graph:
        if node.get("@type") == "WebPage":
            node.setdefault("@id", webpage_id)
            node.setdefault("url", page_url)
            node.setdefault("isPartOf", {"@id": website_id})
            node.setdefault("about", {"@id": clinic_id})


def remove_empty(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, child in value.items():
            c = remove_empty(child)
            if c not in [None, "", [], {}]:
                cleaned[key] = c
        return cleaned
    if isinstance(value, list):
        return [v for v in [remove_empty(child) for child in value] if v not in [None, "", [], {}]]
    return value


def sanitize_schema(schema_obj: Dict, summary: Dict) -> Dict:
    page_url = summary["page_url"]
    schema_obj = ensure_graph(schema_obj, page_url)
    graph = schema_obj["@graph"]
    ensure_required_core_nodes(graph, summary)
    make_default_ids(graph, page_url)
    sanitize_provider_nodes(graph, page_url)
    connect_core_nodes(graph, page_url)
    return remove_empty(schema_obj)


def page_signature(summary: Dict) -> str:
    payload = {"cache_version": SCHEMA_CACHE_VERSION, "summary": summary}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


def generate_schema_json_zip(
    root_site: str,
    openai_api_key: str,
    model: str,
    timeout_seconds: int,
    max_urls: int,
    pause_seconds: float,
    max_retries: int,
    progress_placeholder,
) -> Tuple[bytes, Dict]:
    client = OpenAI(api_key=openai_api_key)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SchemaCrawler/1.0)"})

    sitemaps = discover_sitemaps(root_site, timeout_seconds=timeout_seconds)
    discovered_urls: List[str] = []
    for sitemap_url in sitemaps:
        discovered_urls.extend(parse_sitemap_urls(sitemap_url, timeout_seconds=timeout_seconds))

    urls = [u for u in unique_keep_order(discovered_urls) if u.startswith(("http://", "https://")) and not should_skip_schema_url(u)]
    if max_urls > 0:
        urls = urls[:max_urls]

    cache_store = st.session_state.setdefault("schema_cache_store", {})
    schema_files: Dict[str, bytes] = {}
    error_files: Dict[str, bytes] = {}
    cached_count = 0
    success_count = 0

    total = max(len(urls), 1)
    for i, url in enumerate(urls, start=1):
        progress_placeholder.progress(i / total, text=f"Generating schema JSON {i}/{len(urls)}")
        try:
            html = session.get(url, timeout=timeout_seconds).text
            summary = extract_page_summary(url, html)
            signature = page_signature(summary)

            cached = cache_store.get(url)
            if cached and cached.get("signature") == signature:
                schema_obj = cached["schema"]
                cached_count += 1
            else:
                last_error = None
                schema_obj = None
                for attempt in range(1, max_retries + 1):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "Return only valid JSON."},
                                {"role": "user", "content": build_schema_prompt(summary)},
                            ],
                            response_format={"type": "json_object"},
                        )
                        text = (response.choices[0].message.content or "").strip()
                        parsed = json.loads(text)
                        if "schema_jsonld" not in parsed or not isinstance(parsed["schema_jsonld"], dict):
                            raise ValueError("schema_jsonld missing or invalid.")
                        schema_obj = sanitize_schema(parsed["schema_jsonld"], summary)
                        cache_store[url] = {"signature": signature, "schema": schema_obj}
                        success_count += 1
                        break
                    except Exception as err:
                        last_error = err
                        time.sleep(1.5 * attempt)

                if schema_obj is None:
                    raise RuntimeError(str(last_error) if last_error else "Unknown schema generation error")

            schema_path = f"schema_json/{stable_slug(url)}.json"
            schema_files[schema_path] = json.dumps(schema_obj, indent=2, ensure_ascii=False).encode("utf-8")
            if pause_seconds > 0:
                time.sleep(pause_seconds)
        except Exception as err:
            error_path = f"schema_errors/{stable_slug(url)}.txt"
            error_files[error_path] = f"URL: {url}\n\n{err}".encode("utf-8")

    summary_payload = {
        "root_site": root_site,
        "model": model,
        "generated_at_utc": datetime.utcnow().isoformat(),
        "total_urls": len(urls),
        "succeeded": len(schema_files),
        "cached": cached_count,
        "failed": len(error_files),
    }

    output_buffer = io.BytesIO()
    with zipfile.ZipFile(output_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in schema_files.items():
            archive.writestr(path, content)
        for path, content in error_files.items():
            archive.writestr(path, content)
        archive.writestr("summary.json", json.dumps(summary_payload, indent=2).encode("utf-8"))

    output_buffer.seek(0)
    return output_buffer.getvalue(), summary_payload


def apply_base_styles():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Work+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --mp-pistachio: #B5D994;
                --mp-sky: #A5D4EF;
                --mp-dim: #2E3D3C;
                --mp-alabaster: #F2F3EC;
                --mp-white: #FFFFFF;
                --mp-sea: #77BFD8;
                --mp-mint: #8BCFB2;
            }
            .stApp {
                font-family: "Work Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: var(--mp-alabaster);
                color: var(--mp-dim);
            }
            .block-container { max-width: 1200px; padding-top: 4.5rem; padding-bottom: 1.5rem; }
            h1, h2, h3 { font-family: "Work Sans", sans-serif; color: var(--mp-dim) !important; letter-spacing: 0.01em; }
            .stTextInput input, .stTextArea textarea, .stMultiSelect [data-baseweb="select"], .stNumberInput input {
                background-color: var(--mp-white) !important;
                color: var(--mp-dim) !important;
                border: 1px solid #C4C8C0 !important;
                border-radius: 10px !important;
                caret-color: #2B8E9C !important;
            }
            .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
                border: 1px solid #2B8E9C !important;
                box-shadow: 0 0 0 3px rgba(165, 212, 239, 0.55) !important;
                outline: none !important;
            }
            [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; border-bottom: 1px solid #CFCFC7; }
            [data-baseweb="tab"] {
                background: #E8EAE2 !important;
                border-radius: 8px 8px 0 0 !important;
                color: var(--mp-dim) !important;
                font-weight: 600 !important;
            }
            [aria-selected="true"][data-baseweb="tab"] { background: var(--mp-pistachio) !important; color: var(--mp-dim) !important; }
            [data-baseweb="tag"] { background: var(--mp-sky) !important; color: var(--mp-dim) !important; }
            .stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
                border-radius: 10px; font-weight: 600;
                border: 1px solid #5D9FA7 !important;
                background: linear-gradient(90deg, var(--mp-pistachio) 0%, var(--mp-sky) 100%) !important;
                color: var(--mp-dim) !important; transition: all .15s ease-in-out;
            }
            .stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
                border-color: #74B3C0 !important;
                background: linear-gradient(90deg, var(--mp-mint) 0%, var(--mp-sea) 100%) !important;
                box-shadow: 0 0 0 2px rgba(119, 191, 216, 0.35);
            }
            [data-testid="stMetricValue"] { color: var(--mp-dim) !important; font-weight: 700; }
            .mp-brand-card {
                border: 1px solid #CBD0C7; border-left: 6px solid var(--mp-pistachio); border-radius: 12px;
                padding: 1rem 1.2rem; margin-bottom: 1rem;
                background: linear-gradient(90deg, rgba(165,212,239,.35) 0%, rgba(242,243,236,1) 35%);
            }
            .mp-kicker { text-transform: uppercase; letter-spacing: .16em; font-size: 0.74rem; color: var(--mp-dim); margin-bottom: 0.35rem; font-weight: 600; }
            .mp-section { background: rgba(255,255,255,0.55); border: 1px solid #CFD6CC; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="mp-brand-card">
          <div class="mp-kicker">Modern Practice • Performance & Schema Intelligence</div>
          <h1 style="margin:0; font-size: 2rem;">Performance Audit Studio</h1>
          <p style="margin: .45rem 0 0 0; font-size: 1rem;">Sitemap-driven PageSpeed audits and AI schema generation in one branded workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("API capabilities used in this app", expanded=False):
        st.markdown(
            """
            This app uses **PageSpeed Insights API v5** (`runPagespeed`) and exposes controls for the supported query parameters:
            - `strategy`: `mobile` and/or `desktop`
            - `category`: `performance`, `accessibility`, `best-practices`, `seo`
            - `locale`: localization code for formatted output
            - `url`: page URL being audited
            - `key`: your API key (optional in UI, recommended for quota)
            """
        )


def init_ui_state():
    defaults = {
        "perf_api_key": os.getenv("PSI_API_KEY", ""),
        "perf_client_url": "",
        "perf_strategies": ["mobile"],
        "perf_categories": ["performance", "accessibility", "best-practices", "seo"],
        "perf_locale": "en-US",
        "perf_include_field_data": True,
        "perf_sitemap_only": True,
        "perf_max_urls": 50,
        "perf_timeout_seconds": 60,
        "perf_max_workers": 4,
        "perf_retries": 2,
        "perf_include_pattern": "",
        "perf_exclude_pattern": "",
        "perf_min_perf": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_performance_controls() -> Tuple[bool, Dict]:
    st.markdown("<div class='mp-section'>", unsafe_allow_html=True)
    st.subheader("Audit Setup")
    with st.form("performance_audit_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        c1.text_input("PageSpeed API Key", key="perf_api_key", type="password")
        c2.text_input("Client Website URL", key="perf_client_url", placeholder="https://example.com")

        c3, c4, c5 = st.columns([1.2, 1.2, 0.8])
        c3.multiselect("Device Strategy", ["mobile", "desktop"], key="perf_strategies")
        c4.multiselect("Lighthouse Categories", ["performance", "accessibility", "best-practices", "seo"], key="perf_categories")
        c5.text_input("Locale", key="perf_locale")

        with st.expander("Advanced Settings", expanded=False):
            a1, a2 = st.columns(2)
            a1.toggle("Include CrUX field-data columns", key="perf_include_field_data")
            a2.toggle("Use sitemap URLs only (no fallback crawl)", key="perf_sitemap_only")
            a3, a4, a5 = st.columns(3)
            a3.slider("Max URLs to audit", min_value=5, max_value=500, step=5, key="perf_max_urls")
            a4.slider("Request timeout (seconds)", min_value=10, max_value=180, step=5, key="perf_timeout_seconds")
            a5.slider("Parallel workers", min_value=1, max_value=10, key="perf_max_workers")
            st.slider("Retries per request", min_value=1, max_value=5, key="perf_retries")

        st.subheader("Report Filters")
        f1, f2, f3 = st.columns(3)
        f1.text_input("Include URL regex", key="perf_include_pattern")
        f2.text_input("Exclude URL regex", key="perf_exclude_pattern")
        f3.slider("Minimum performance score", 0, 100, key="perf_min_perf")
        run_clicked = st.form_submit_button("▶️ Run Audit", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    params = {
        "api_key": st.session_state["perf_api_key"],
        "client_url": st.session_state["perf_client_url"],
        "strategies": st.session_state["perf_strategies"],
        "categories": st.session_state["perf_categories"],
        "locale": st.session_state["perf_locale"],
        "include_field_data": st.session_state["perf_include_field_data"],
        "sitemap_only": st.session_state["perf_sitemap_only"],
        "max_urls": st.session_state["perf_max_urls"],
        "timeout_seconds": st.session_state["perf_timeout_seconds"],
        "max_workers": st.session_state["perf_max_workers"],
        "retries": st.session_state["perf_retries"],
        "include_pattern": st.session_state["perf_include_pattern"],
        "exclude_pattern": st.session_state["perf_exclude_pattern"],
        "min_perf": st.session_state["perf_min_perf"],
    }
    return run_clicked, params


def prepare_dashboard_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error_flag"] = out.get("error", pd.Series([None] * len(out))).notna()
    score = pd.to_numeric(out.get("performance_score"), errors="coerce")
    out["performance_score"] = score
    out["score_bucket"] = pd.cut(
        score,
        bins=[-1, 49, 89, 100],
        labels=["Red (0-49)", "Orange (50-89)", "Green (90-100)"],
    ).astype("string")
    out["score_bucket"] = out["score_bucket"].fillna("Unknown")

    def _path_segment(url: str) -> str:
        parsed = urllib.parse.urlparse(str(url))
        parts = [p for p in parsed.path.split("/") if p]
        return parts[0] if parts else "home"

    out["path_segment"] = out.get("url_requested", pd.Series([""] * len(out))).apply(_path_segment)
    return out


def render_dashboard_toolbar(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("### Dashboard Toolbar")
    t1, t2, t3, t4, t5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.2])
    sort_by = t1.selectbox("Sort by", ["performance_score", "lcp_seconds", "tbt_ms", "cls", "url_requested"], key="dash_sort_by")
    metric_focus = t2.selectbox("Metric Focus", ["Performance", "LCP", "TBT", "CLS"], key="dash_metric_focus")
    strategy_options = ["All"] + sorted([s for s in df.get("strategy", pd.Series([])).dropna().unique().tolist()])
    strategy_filter = t3.selectbox("Strategy", strategy_options, key="dash_strategy_filter")
    score_bucket_filter = t4.selectbox("Score Bucket", ["All", "Green (90-100)", "Orange (50-89)", "Red (0-49)", "Unknown"], key="dash_bucket_filter")
    problems_only = t5.toggle("Show only problem pages", key="dash_problems_only")

    filtered = df.copy()
    if strategy_filter != "All" and "strategy" in filtered.columns:
        filtered = filtered[filtered["strategy"] == strategy_filter]
    if score_bucket_filter != "All":
        filtered = filtered[filtered["score_bucket"] == score_bucket_filter]
    if problems_only:
        filtered = filtered[
            filtered["error_flag"]
            | (pd.to_numeric(filtered.get("lcp_seconds"), errors="coerce") > 2.5)
            | (pd.to_numeric(filtered.get("cls"), errors="coerce") > 0.10)
            | (pd.to_numeric(filtered.get("tbt_ms"), errors="coerce") > 200)
            | (pd.to_numeric(filtered.get("performance_score"), errors="coerce") < 50)
        ]

    ascending = sort_by in {"performance_score"}
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=ascending, na_position="last")
    st.caption(f"Metric focus: **{metric_focus}**")
    return filtered


def render_audit_summary(df: pd.DataFrame):
    st.subheader("Audit Summary")
    score = pd.to_numeric(df.get("performance_score"), errors="coerce")
    lcp = pd.to_numeric(df.get("lcp_seconds"), errors="coerce")
    cls = pd.to_numeric(df.get("cls"), errors="coerce")
    error_count = int(df.get("error_flag", pd.Series([False] * len(df))).sum())

    green = int((score >= 90).sum())
    orange = int(((score >= 50) & (score < 90)).sum())
    red = int((score < 50).sum())

    cards = st.columns(9)
    cards[0].metric("URLs Audited", len(df))
    cards[1].metric("Avg Perf", round(score.mean(), 1) if score.notna().any() else "n/a")
    cards[2].metric("Median Perf", round(score.median(), 1) if score.notna().any() else "n/a")
    cards[3].metric("Green Zone (90+)", green)
    cards[4].metric("Orange (50-89)", orange)
    cards[5].metric("Red (<50)", red)
    cards[6].metric("Avg LCP", round(lcp.mean(), 2) if lcp.notna().any() else "n/a")
    cards[7].metric("Avg CLS", round(cls.mean(), 3) if cls.notna().any() else "n/a")
    cards[8].metric("Error Count", error_count)


def render_score_overview(df: pd.DataFrame):
    st.subheader("Score Overview")
    numeric_df = df[~df["error_flag"]].copy()
    if numeric_df.empty:
        st.info("No successful rows available for score overview charts.")
        return

    c1, c2 = st.columns(2)
    hist = (
        alt.Chart(numeric_df)
        .mark_bar(color="#77BFD8")
        .encode(x=alt.X("performance_score:Q", bin=alt.Bin(maxbins=20), title="Performance Score"), y=alt.Y("count():Q", title="Pages"))
        .properties(height=280, title="Performance Score Distribution")
    )
    c1.altair_chart(hist, use_container_width=True)

    buckets = numeric_df.groupby("score_bucket", dropna=False).size().reset_index(name="count")
    bucket_chart = (
        alt.Chart(buckets)
        .mark_bar()
        .encode(
            x=alt.X("score_bucket:N", title="Score Bucket", sort=["Red (0-49)", "Orange (50-89)", "Green (90-100)", "Unknown"]),
            y=alt.Y("count:Q", title="Pages"),
            color=alt.Color("score_bucket:N", scale=alt.Scale(domain=["Red (0-49)", "Orange (50-89)", "Green (90-100)", "Unknown"], range=["#E88B8B", "#EFC78A", "#B5D994", "#CFD6CC"])),
        )
        .properties(height=280, title="Performance Score Buckets")
    )
    c2.altair_chart(bucket_chart, use_container_width=True)

    avg_values = pd.DataFrame(
        {
            "category": ["Performance", "Accessibility", "Best Practices", "SEO"],
            "avg_score": [
                pd.to_numeric(numeric_df.get("performance_score"), errors="coerce").mean(),
                pd.to_numeric(numeric_df.get("accessibility_score"), errors="coerce").mean(),
                pd.to_numeric(numeric_df.get("best_practices_score"), errors="coerce").mean(),
                pd.to_numeric(numeric_df.get("seo_score"), errors="coerce").mean(),
            ],
        }
    ).dropna()
    cat_chart = (
        alt.Chart(avg_values)
        .mark_bar(color="#8BCFB2")
        .encode(x=alt.X("category:N", title="Category"), y=alt.Y("avg_score:Q", title="Average Score"))
        .properties(height=320, title="Category Average Scores")
    )
    st.altair_chart(cat_chart, use_container_width=True)


def _top_n_chart(df: pd.DataFrame, metric_col: str, title: str, n: int = 10):
    if metric_col not in df.columns:
        st.info(f"{title}: metric unavailable.")
        return
    sample = df[~df["error_flag"]][["url_requested", metric_col]].copy()
    sample[metric_col] = pd.to_numeric(sample[metric_col], errors="coerce")
    sample = sample.dropna().sort_values(metric_col, ascending=False).head(n)
    if sample.empty:
        st.info(f"{title}: no data.")
        return
    chart = (
        alt.Chart(sample)
        .mark_bar(color="#A5D4EF")
        .encode(
            x=alt.X(f"{metric_col}:Q", title=metric_col.replace("_", " ").upper()),
            y=alt.Y("url_requested:N", sort="-x", title="URL"),
            tooltip=["url_requested", metric_col],
        )
        .properties(height=320, title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def render_core_metrics(df: pd.DataFrame):
    st.subheader("Core Performance Metrics")
    c1, c2 = st.columns(2)
    with c1:
        _top_n_chart(df, "lcp_seconds", "Top 10 Slowest Pages by LCP")
    with c2:
        _top_n_chart(df, "tbt_ms", "Top 10 Highest TBT Pages")
    _top_n_chart(df, "cls", "Top 10 Worst CLS Pages")

    numeric = df[~df["error_flag"]].copy()
    if {"performance_score", "lcp_seconds"}.issubset(numeric.columns):
        s1, s2 = st.columns(2)
        scatter_lcp = (
            alt.Chart(numeric.dropna(subset=["performance_score", "lcp_seconds"]))
            .mark_circle(size=70, color="#77BFD8")
            .encode(
                x=alt.X("lcp_seconds:Q", title="LCP (seconds)"),
                y=alt.Y("performance_score:Q", title="Performance Score"),
                tooltip=["url_requested", "strategy", "performance_score", "lcp_seconds"],
            )
            .properties(height=300, title="Performance Score vs LCP")
        )
        s1.altair_chart(scatter_lcp, use_container_width=True)
        if "tbt_ms" in numeric.columns:
            scatter_tbt = (
                alt.Chart(numeric.dropna(subset=["performance_score", "tbt_ms"]))
                .mark_circle(size=70, color="#8BCFB2")
                .encode(
                    x=alt.X("tbt_ms:Q", title="TBT (ms)"),
                    y=alt.Y("performance_score:Q", title="Performance Score"),
                    tooltip=["url_requested", "strategy", "performance_score", "tbt_ms"],
                )
                .properties(height=300, title="Performance Score vs TBT")
            )
            s2.altair_chart(scatter_tbt, use_container_width=True)


def render_strategy_comparison(df: pd.DataFrame):
    strategies = sorted(df.get("strategy", pd.Series([])).dropna().unique().tolist())
    if not {"mobile", "desktop"}.issubset(set(strategies)):
        return
    st.subheader("Strategy Comparison")
    numeric = df[~df["error_flag"]].copy()
    agg = (
        numeric.groupby("strategy", dropna=False)[["performance_score", "lcp_seconds", "tbt_ms", "cls"]]
        .mean(numeric_only=True)
        .reset_index()
        .round(2)
    )
    st.dataframe(agg, use_container_width=True, hide_index=True)
    melt = agg.melt(id_vars="strategy", value_vars=["performance_score", "lcp_seconds", "tbt_ms", "cls"], var_name="metric", value_name="value")
    chart = (
        alt.Chart(melt)
        .mark_bar()
        .encode(x="metric:N", y="value:Q", color="strategy:N", xOffset="strategy:N", tooltip=["strategy", "metric", "value"])
        .properties(height=300, title="Average Metrics by Strategy")
    )
    st.altair_chart(chart, use_container_width=True)


def render_url_insights(df: pd.DataFrame):
    st.subheader("URL Insights")
    urls = sorted(df.get("url_requested", pd.Series([])).dropna().unique().tolist())
    if not urls:
        st.info("No URLs available.")
        return
    search = st.text_input("Search URL", key="dash_url_search", placeholder="Type part of a URL")
    if search:
        urls = [u for u in urls if search.lower() in u.lower()]
    selected = st.selectbox("Select URL", urls, key="dash_url_select")
    page_rows = df[df["url_requested"] == selected].copy()
    if page_rows.empty:
        return
    row = page_rows.iloc[0]
    score = pd.to_numeric(pd.Series([row.get("performance_score")]), errors="coerce").iloc[0]
    status = "Strong" if pd.notna(score) and score >= 90 else "Needs Improvement" if pd.notna(score) and score >= 50 else "Critical"

    st.markdown(f"**Status:** `{status}`")
    metrics = ["performance_score", "accessibility_score", "best_practices_score", "seo_score", "lcp_seconds", "cls", "tbt_ms", "fcp_seconds", "speed_index_seconds", "tti_seconds", "inp_ms_lab"]
    details = {m: row.get(m) for m in metrics if m in row.index}
    details["strategy"] = row.get("strategy")
    details["url_final"] = row.get("url_final")
    st.json(details)


def render_problem_pages(df: pd.DataFrame):
    st.subheader("Problem Pages")
    base = df.copy()
    base["lcp_fail"] = pd.to_numeric(base.get("lcp_seconds"), errors="coerce") > 2.5
    base["cls_fail"] = pd.to_numeric(base.get("cls"), errors="coerce") > 0.10
    base["tbt_fail"] = pd.to_numeric(base.get("tbt_ms"), errors="coerce") > 200

    st.markdown("**Worst Performing Pages**")
    worst = base.sort_values("performance_score", ascending=True, na_position="last").head(10)
    st.dataframe(worst[["url_requested", "strategy", "performance_score", "lcp_seconds", "cls", "tbt_ms", "error"]], use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Pages with Errors**")
        errors = base[base["error_flag"]]
        st.dataframe(errors[["url_requested", "strategy", "error"]], use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Threshold Failures**")
        st.write(f"LCP > 2.5s: **{int(base['lcp_fail'].sum())}**")
        st.write(f"CLS > 0.10: **{int(base['cls_fail'].sum())}**")
        st.write(f"TBT > 200ms: **{int(base['tbt_fail'].sum())}**")

    f1, f2, f3 = st.columns(3)
    f1.dataframe(base[base["lcp_fail"]][["url_requested", "strategy", "lcp_seconds"]], use_container_width=True, hide_index=True)
    f2.dataframe(base[base["cls_fail"]][["url_requested", "strategy", "cls"]], use_container_width=True, hide_index=True)
    f3.dataframe(base[base["tbt_fail"]][["url_requested", "strategy", "tbt_ms"]], use_container_width=True, hide_index=True)


def render_quick_insights(df: pd.DataFrame):
    st.subheader("Quick Insights")
    notes = []
    counts = df["score_bucket"].value_counts()
    if not counts.empty:
        notes.append(f"Most audited pages fall in **{counts.idxmax()}**.")
    metric_means = {
        "LCP": pd.to_numeric(df.get("lcp_seconds"), errors="coerce").mean(),
        "TBT": pd.to_numeric(df.get("tbt_ms"), errors="coerce").mean(),
        "CLS": pd.to_numeric(df.get("cls"), errors="coerce").mean(),
    }
    if pd.notna(metric_means["LCP"]) and metric_means["LCP"] > 2.5:
        notes.append("**LCP** is above the recommended threshold on average.")
    if {"mobile", "desktop"}.issubset(set(df.get("strategy", pd.Series([])).dropna().unique())):
        perf_by_strategy = df.groupby("strategy")["performance_score"].mean()
        if "mobile" in perf_by_strategy and "desktop" in perf_by_strategy and perf_by_strategy["mobile"] < perf_by_strategy["desktop"]:
            notes.append("Mobile scores are materially worse than desktop.")
    seg = df.groupby("path_segment")["performance_score"].mean().dropna().sort_values()
    if not seg.empty:
        notes.append(f"Weakest path segment is **/{seg.index[0]}** (avg score {seg.iloc[0]:.1f}).")
    for note in notes[:4]:
        st.write(f"- {note}")


def render_downloads(df: pd.DataFrame, timestamp: str):
    st.subheader("Downloads")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download CSV report", data=csv_bytes, file_name=f"psi_audit_{timestamp}.csv", mime="text/csv", use_container_width=True)
    d2.download_button("⬇️ Download JSON report", data=json_bytes, file_name=f"psi_audit_{timestamp}.json", mime="application/json", use_container_width=True)


def render_performance_results(filtered: pd.DataFrame, timestamp: str):
    dashboard_df = prepare_dashboard_data(filtered)
    dashboard_df = render_dashboard_toolbar(dashboard_df)

    render_audit_summary(dashboard_df)
    render_score_overview(dashboard_df)
    render_core_metrics(dashboard_df)
    render_strategy_comparison(dashboard_df)
    render_url_insights(dashboard_df)
    render_problem_pages(dashboard_df)
    render_quick_insights(dashboard_df)

    st.subheader("Full Results")
    st.dataframe(dashboard_df, use_container_width=True, height=500)
    render_downloads(dashboard_df, timestamp)


def run_performance_audit_workflow(params: Dict):
    if not params["strategies"]:
        st.error("Select at least one strategy.")
        return
    if not params["categories"]:
        st.error("Select at least one category.")
        return

    try:
        base_url = normalize_base_url(params["client_url"])
    except ValueError as err:
        st.error(str(err))
        return

    st.info(f"Starting discovery for **{base_url}**")
    sitemaps = discover_sitemaps(base_url, timeout_seconds=params["timeout_seconds"])
    all_urls: List[str] = []

    if sitemaps:
        st.success(f"Discovered {len(sitemaps)} sitemap(s)")
        with st.expander("View discovered sitemap URLs"):
            st.write(sitemaps)
        for sm in sitemaps:
            all_urls.extend(parse_sitemap_urls(sm, timeout_seconds=params["timeout_seconds"]))

    unique_urls = sorted({u for u in all_urls if u.startswith(("http://", "https://"))})
    if not unique_urls and not params["sitemap_only"]:
        st.warning("No sitemap URLs found. Running fallback same-site crawl from homepage.")
        unique_urls = crawl_fallback_urls(base_url, timeout_seconds=params["timeout_seconds"], max_urls=params["max_urls"])
    if not unique_urls:
        st.error("No URLs discovered. Try disabling sitemap-only mode or verify robots/sitemap accessibility.")
        return

    unique_urls = unique_urls[: params["max_urls"]]
    st.write(f"URLs queued for audit: **{len(unique_urls)}**")

    config = AuditConfig(
        api_key=params["api_key"].strip(),
        strategies=params["strategies"],
        categories=params["categories"],
        locale=params["locale"].strip() or "en-US",
        max_urls=params["max_urls"],
        timeout_seconds=params["timeout_seconds"],
        max_workers=params["max_workers"],
        retries=params["retries"],
        include_field_data=params["include_field_data"],
    )

    progress = st.progress(0.0, text="Preparing...")
    df = run_audit(unique_urls, config, progress)
    if df.empty:
        st.error("Audit returned no rows.")
        return

    filtered = apply_custom_filters(df, params["include_pattern"], params["exclude_pattern"], params["min_perf"])
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    st.session_state["perf_last_filtered"] = filtered
    st.session_state["perf_last_timestamp"] = timestamp
    render_performance_results(filtered, timestamp)


def render_performance_tab():
    run_clicked, params = render_performance_controls()
    if run_clicked:
        run_performance_audit_workflow(params)
    elif "perf_last_filtered" in st.session_state:
        render_performance_results(st.session_state["perf_last_filtered"], st.session_state.get("perf_last_timestamp", datetime.utcnow().strftime("%Y%m%d_%H%M%S")))
    else:
        st.info("Configure **Audit Setup** and click **Run Audit** to generate a performance dashboard.")


def render_schema_tab():
    schema_tab = st.container()
    with schema_tab:
        st.subheader("AI Schema Doc Generator")
        st.caption("Discovers sitemap URLs, generates sanitized JSON-LD per page, and downloads schema/error JSON files in a ZIP.")
        schema_root = st.text_input("Root website for schema generation", placeholder="https://example.com", key="schema_root")
        schema_openai_key = st.text_input("OpenAI API Key", value="", type="password", key="schema_key")
        s1, s2, s3, s4 = st.columns(4)
        schema_model = s1.text_input("Model", value="gpt-5-mini", key="schema_model")
        schema_max_urls = s2.number_input("Max URLs", min_value=1, max_value=1000, value=50, step=1, key="schema_max_urls")
        schema_pause = s3.number_input("Pause between requests (sec)", min_value=0.0, max_value=10.0, value=0.5, step=0.1, key="schema_pause")
        schema_retries = s4.number_input("Retries per URL", min_value=1, max_value=6, value=3, step=1, key="schema_retries")
        schema_timeout = st.slider("Request timeout (seconds)", min_value=10, max_value=180, value=60, step=5, key="schema_timeout")

        run_schema = st.button("🧩 Generate Schema JSON ZIP", type="primary", use_container_width=True, key="run_schema")
        if run_schema:
            if not schema_openai_key.strip():
                st.error("OpenAI API key is required. Please paste your key to run schema generation.")
                return
            try:
                normalized_root = normalize_base_url(schema_root)
            except ValueError as err:
                st.error(str(err))
                return

            schema_progress = st.progress(0.0, text="Preparing schema generation...")
            with st.spinner("Generating schema JSON files. This may take a while for many URLs."):
                zip_bytes, summary = generate_schema_json_zip(
                    root_site=normalized_root,
                    openai_api_key=schema_openai_key.strip(),
                    model=schema_model.strip() or "gpt-5-mini",
                    timeout_seconds=int(schema_timeout),
                    max_urls=int(schema_max_urls),
                    pause_seconds=float(schema_pause),
                    max_retries=int(schema_retries),
                    progress_placeholder=schema_progress,
                )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("URLs Processed", summary["total_urls"])
            k2.metric("Success", summary["succeeded"])
            k3.metric("Cached", summary["cached"])
            k4.metric("Failed", summary["failed"])

            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download schema_json.zip",
                data=zip_bytes,
                file_name=f"schema_json_{stamp}.zip",
                mime="application/zip",
                use_container_width=True,
            )


def render():
    st.set_page_config(page_title="Performance Audit Studio", page_icon="🚀", layout="wide")
    apply_base_styles()
    init_ui_state()
    render_header()

    performance_tab_container, schema_tab_container = st.tabs(["Performance Audit", "Schema Doc Generator"])
    with performance_tab_container:
        render_performance_tab()
    with schema_tab_container:
        render_schema_tab()


if __name__ == "__main__":
    render()
