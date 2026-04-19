"""
Web search fallback for Gyan Engine.

When the KB doesn't have an answer, search Wikipedia + DuckDuckGo,
validate via source agreement, and learn the answer.

This is the self-evolution loop: every miss makes the system smarter.
"""

import json
import re
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


def search_wikipedia(query: str, timeout: int = 10) -> list[dict]:
    """Search Wikipedia for relevant articles."""
    results = []

    # Method 1: Wikipedia REST API
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query)}"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Gyan/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            if data.get("extract"):
                results.append({
                    "text": data["extract"],
                    "source": f"wikipedia:{data.get('title', query)}",
                    "title": data.get("title", ""),
                })
    except Exception:
        pass

    # Method 2: Wikipedia search API
    try:
        params = urllib.parse.urlencode({
            "action": "query", "list": "search",
            "srsearch": query, "srlimit": "3",
            "format": "json", "utf8": "1",
        })
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Gyan/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            for item in data.get("query", {}).get("search", []):
                snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
                if snippet:
                    results.append({
                        "text": snippet,
                        "source": f"wikipedia:{item['title']}",
                        "title": item["title"],
                    })
    except Exception:
        pass

    return results


def search_duckduckgo(query: str, timeout: int = 10) -> list[dict]:
    """Search DuckDuckGo Instant Answers API."""
    results = []
    try:
        params = urllib.parse.urlencode({
            "q": query, "format": "json", "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Gyan/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())

            # Abstract
            if data.get("Abstract"):
                results.append({
                    "text": data["Abstract"],
                    "source": f"ddg:{data.get('AbstractSource', 'unknown')}",
                    "title": data.get("Heading", ""),
                })

            # Answer
            if data.get("Answer"):
                results.append({
                    "text": str(data["Answer"]),
                    "source": "ddg:instant",
                    "title": query,
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:2]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "text": topic["Text"],
                        "source": "ddg:related",
                        "title": topic.get("FirstURL", ""),
                    })
    except Exception:
        pass

    return results


def web_search(query: str, timeout: int = 10) -> list[dict]:
    """
    Search multiple sources in parallel.
    Returns list of {text, source, title}.
    """
    all_results = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(search_wikipedia, query, timeout): "wikipedia",
            pool.submit(search_duckduckgo, query, timeout): "duckduckgo",
        }
        for future in as_completed(futures, timeout=timeout + 5):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception:
                pass

    return all_results


def extract_best_answer(results: list[dict], max_len: int = 500) -> Optional[str]:
    """Pick the best answer from web search results."""
    if not results:
        return None

    # Prefer longer, more informative answers
    results.sort(key=lambda r: len(r["text"]), reverse=True)

    best = results[0]["text"]
    if len(best) > max_len:
        # Truncate at sentence boundary
        truncated = best[:max_len]
        last_period = truncated.rfind(".")
        if last_period > max_len // 2:
            best = truncated[:last_period + 1]
        else:
            best = truncated + "..."

    return best


def count_source_agreement(results: list[dict], answer: str) -> int:
    """Count how many sources agree on the answer."""
    if not answer:
        return 0
    answer_lower = answer.lower()
    count = 0
    seen_sources = set()
    for r in results:
        source_type = r["source"].split(":")[0]
        if source_type in seen_sources:
            continue
        if answer_lower in r["text"].lower() or r["text"].lower() in answer_lower:
            count += 1
            seen_sources.add(source_type)
    return count


# ─── Integration with GyanEngine ──────────────────────────────

def query_with_fallback(engine, question: str, context: str = "") -> dict:
    """
    Query engine, fall back to web search on miss, learn the answer.
    This is the self-evolution loop.
    """
    result = engine.query(question, context)

    if result["answer"] and result["confidence"] > 0.3:
        return result

    # Web search fallback
    web_results = web_search(question)
    if not web_results:
        return result

    answer = extract_best_answer(web_results)
    if not answer:
        return result

    # Source agreement determines initial weight
    agreement = count_source_agreement(web_results, answer)
    weight = 0.5 if agreement >= 2 else 0.3

    # Learn the answer — INSERT INTO
    engine.learn(question, answer, source="web-search", weight=weight)

    return {
        "answer": answer,
        "confidence": weight,
        "hops": 0,
        "timeMs": result["timeMs"],
        "source": "web-search",
        "agreement": agreement,
        "learned": True,
    }
