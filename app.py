import os
import re
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

APP_ID = os.getenv("OMI_APP_ID", "")
APP_SECRET = os.getenv("OMI_APP_SECRET", "")
SERP_API_KEY = ""
SERP_API_URL = "https://serpapi.com/search.json"
RESEND_API_KEY = ""
RESEND_EMAIL_URL = "https://api.resend.com/emails"
RESEND_FROM = ""
RESEND_TO = ""

GREETING_KEYWORDS = {
    "nice",
    "great",
    "good",
    "pleased",
    "pleasure",
    "lovely",
    "happy",
    "wonderful",
}
MISHEARD_TOKEN_MAP = {"9": "nice"}
BANK_STATES: Dict[str, Dict[str, object]] = {}


async def send_omi_notification(uid: str, message: str) -> None:
    if not APP_ID or not APP_SECRET:
        raise RuntimeError("OMI_APP_ID or OMI_APP_SECRET not set")

    url = f"https://api.omi.me/v2/integrations/{APP_ID}/notification"
    params = {"uid": uid, "message": message}
    headers = {
        "Authorization": f"Bearer {APP_SECRET}",
        "Content-Type": "application/json",
    }

    print(f"[notify] Sending OMI notification to uid={uid} message='{message}'")

    async with httpx.AsyncClient(timeout=10) as http:
        resp = await http.post(url, params=params, headers=headers, content=b"")
        resp.raise_for_status()

    print("[notify] OMI notification sent successfully")


async def send_profiles_email(first_name: str, last_name: str, profiles_by_platform: Dict[str, List[Dict[str, str]]]) -> None:
    print(f"[email] Preparing email for {first_name} {last_name}")

    lines: List[str] = []
    total_profiles = 0

    for platform, profiles in profiles_by_platform.items():
        if not profiles:
            continue

        lines.append(f"{platform} Profiles:")
        for idx, profile in enumerate(profiles, start=1):
            snippet = profile.get("snippet", "").strip()
            snippet_text = snippet if snippet else "No snippet provided."
            lines.append(
                f"  {idx}. {profile.get('title', '').strip()}\n     {profile.get('link', '').strip()}\n     {snippet_text}"
            )
        lines.append("")
        total_profiles += len(profiles)

    if total_profiles == 0:
        lines.append("No profiles were found automatically across LinkedIn, Instagram, or Twitter.")
        lines.append(f"Tried searching for: {first_name} {last_name}.")
        lines.append("Consider adjusting the spelling or searching manually.")

    body_text = "\n".join(line for line in lines if line is not None)
    subject = (
        f"Social profiles for {first_name} {last_name}"
        if total_profiles
        else f"Social profiles for {first_name} {last_name} (not found)"
    )
    payload = {
        "from": RESEND_FROM,
        "to": [RESEND_TO],
        "subject": subject,
        "text": body_text,
    }
    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=10) as http:
        resp = await http.post(RESEND_EMAIL_URL, json=payload, headers=headers)
        resp.raise_for_status()

    print("[email] Email sent via Resend")


def _looks_like_name(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][a-z]+(?:['-][A-Z][a-z]+)*", token))


def extract_name_from_segments(uid: str, segments: List[Dict]) -> Optional[Tuple[str, str]]:
    state = BANK_STATES.get(uid)
    if state is None:
        state = {"step": 0, "tokens": [], "first": None}
        BANK_STATES[uid] = state

    for segment in segments:
        text = segment.get("text", "")
        if not text:
            continue

        tokens = _normalize_tokens(text)
        if not tokens:
            continue

        print(f"[detect] Tokens={tokens}")

        for token in tokens:
            match = _advance_detection_state(uid, state, token)
            if match:
                BANK_STATES.pop(uid, None)
                return match

    print("[detect] No name detected")
    return None
def _advance_detection_state(uid: str, state: Dict[str, object], token: str) -> Optional[Tuple[str, str]]:
    normalized = MISHEARD_TOKEN_MAP.get(token.lower(), token.lower())
    step = int(state.get("step", 0))
    tokens: List[str] = list(state.get("tokens", []))

    def update(new_step: int, new_tokens: Optional[List[str]] = None) -> None:
        state["step"] = new_step
        state["tokens"] = list(new_tokens) if new_tokens is not None else tokens
        if new_step < 4:
            state["first"] = None
        _log_detection_state(uid, state)

    def reset() -> None:
        state["step"] = 0
        state["tokens"] = []
        state["first"] = None
        _log_detection_state(uid, state)

    if step == 0:
        if normalized in GREETING_KEYWORDS:
            update(1, [token])
        elif normalized == "to":
            update(2, [token])
        else:
            reset()
        return None

    if step == 1:
        if normalized == "to":
            update(2, list(state["tokens"]) + [token])
        elif normalized in GREETING_KEYWORDS:
            update(1, [token])
        else:
            reset()
            return _advance_detection_state(uid, state, token)
        return None

    if step == 2:
        if normalized == "meet":
            update(3, list(state["tokens"]) + [token])
        elif normalized in GREETING_KEYWORDS:
            update(1, [token])
        elif normalized == "to":
            update(2, [token])
        else:
            reset()
        return None

    if step == 3:
        if normalized == "you":
            update(4, list(state["tokens"]) + [token])
        elif normalized in GREETING_KEYWORDS:
            update(1, [token])
        elif normalized == "to":
            update(2, [token])
        else:
            reset()
        return None

    if step == 4:
        if _looks_like_name(token):
            collected = list(state["tokens"]) + [token]
            state["first"] = token
            update(5, collected)
        elif normalized in GREETING_KEYWORDS:
            update(1, [token])
        elif normalized == "to":
            update(2, [token])
        else:
            reset()
        return None

    if step == 5:
        if _looks_like_name(token):
            collected = list(state["tokens"]) + [token]
            first_token = state.get("first")
            reset()
            print(f"[detect] Match tokens={collected}")
            if isinstance(first_token, str):
                return first_token, token
            return None
        elif normalized in GREETING_KEYWORDS:
            update(1, [token])
        elif normalized == "to":
            update(2, [token])
        else:
            reset()
        return None

    reset()
    return None


def _log_detection_state(uid: str, state: Dict[str, object]) -> None:
    step = int(state.get("step", 0))
    if step == 0:
        return
    tokens = state.get("tokens", [])
    print(f"[detect] uid={uid} step={step} tokens={tokens}")


async def fetch_profiles(query: str, platform: str) -> List[Dict[str, str]]:
    params = {
        "engine": "google",
        "q": query,
        "num": 3,
        "api_key": SERP_API_KEY,
    }

    print(f"[serp] Fetching {platform} profiles with query='{query}'")

    profiles: List[Dict[str, str]] = []

    async with httpx.AsyncClient(timeout=10) as http:
        resp = await http.get(SERP_API_URL, params=params)
        resp.raise_for_status()

    payload = resp.json()
    results = payload.get("organic_results", [])
    print(f"[serp] {platform} query returned {len(results)} organic result(s)")

    for result in results:
        link = result.get("link")
        if not link:
            continue

        profile = {
            "title": result.get("title", ""),
            "link": link,
            "snippet": result.get("snippet", ""),
        }
        profiles.append(profile)

    return profiles


def _normalize_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text)

@app.post("/")
@app.post("/omi/webhook")
async def omi_webhook(request: Request):
    uid = request.query_params.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="uid missing")

    data = await request.json()
    segments = data.get("segments", [])

    print(f"[webhook] Received request for uid={uid} with {len(segments)} segment(s)")

    for segment in segments:
        text = segment.get("text", "")
        print(f"[segment] text='{text}'")

    match = extract_name_from_segments(uid, segments)
    if not match:
        print("[webhook] No matching segment found")
        return {"ok": True}

    first_name, last_name = match
    print(f"[webhook] Detected name: {first_name} {last_name}")

    platform_queries = {
        "LinkedIn": f"{first_name} {last_name} linkedin profile san francisco bay area tech",
        "Instagram": f"{first_name} {last_name} instagram profile san francisco bay area tech",
        "Twitter": f"{first_name} {last_name} twitter profile san francisco bay area tech",
    }

    profiles_by_platform: Dict[str, List[Dict[str, str]]] = {}
    total_profiles = 0

    for platform, query in platform_queries.items():
        try:
            platform_profiles = await fetch_profiles(query, platform)
        except httpx.HTTPError as exc:
            print(f"[error] SerpAPI request failed for {platform}: {exc}")
            raise HTTPException(status_code=502, detail="Failed to search profiles") from exc

        profiles_by_platform[platform] = platform_profiles
        total_profiles += len(platform_profiles)

    try:
        await send_profiles_email(first_name, last_name, profiles_by_platform)
    except HTTPException as exc:
        print(f"[error] Email not sent: {exc.detail}")
        raise
    except httpx.HTTPError as exc:
        print(f"[error] Resend request failed: {exc}")
        raise HTTPException(status_code=502, detail="Failed to send email") from exc

    print(f"[webhook] Email dispatched with {total_profiles} profile(s) across platforms")

    msg = (
        "Found profiles, check your email"
        if total_profiles
        else "No profiles found. Check your email for details."
    )
    try:
        await send_omi_notification(uid, msg)
    except RuntimeError as exc:
        print(f"[error] OMI configuration error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        detail = "Failed to send OMI notification"
        if status_code == 429:
            detail = "Rate limited while sending OMI notification"
        print(f"[error] OMI returned status {status_code}: {exc}")
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        print(f"[error] OMI transport failure: {exc}")
        raise HTTPException(status_code=502, detail="Failed to send OMI notification") from exc

    print("[webhook] Processing complete for request")

    return {"ok": True}
