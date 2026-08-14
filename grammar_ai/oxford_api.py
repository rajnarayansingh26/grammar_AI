import os
import requests
from functools import lru_cache
from dotenv import load_dotenv

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

BASE_URL = os.getenv(
    "OXFORD_BASE_URL",
    "https://od-api-sandbox.oxforddictionaries.com/api/v2"
)

APP_ID = os.getenv("OXFORD_APP_ID")
APP_KEY = os.getenv("OXFORD_APP_KEY")

LANGUAGE = "en-gb"

# ============================================================
# CHECK CREDENTIALS
# ============================================================

if not APP_ID:
    raise ValueError(
        "OXFORD_APP_ID is missing from .env"
    )

if not APP_KEY:
    raise ValueError(
        "OXFORD_APP_KEY is missing from .env"
    )


# ============================================================
# HEADERS
# ============================================================

HEADERS = {
    "app_id": APP_ID,
    "app_key": APP_KEY,
    "Accept": "application/json"
}


# ============================================================
# GENERIC API REQUEST
# ============================================================

def _request(endpoint, params=None):

    url = f"{BASE_URL}{endpoint}"

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=10
        )

        # Successful request
        if response.status_code == 200:
            return response.json()

        # Word/data not found
        if response.status_code == 404:
            return {
                "error": "not_found",
                "status_code": 404,
                "message": "No Oxford data found."
            }

        # Authentication error
        if response.status_code == 401:
            return {
                "error": "authentication_failed",
                "status_code": 401,
                "message": "Check Oxford App ID and App Key."
            }

        # Rate limit
        if response.status_code == 429:
            return {
                "error": "rate_limit",
                "status_code": 429,
                "message": "Oxford API rate limit reached."
            }

        # Other errors
        return {
            "error": "api_error",
            "status_code": response.status_code,
            "message": response.text
        }

    except requests.exceptions.Timeout:

        return {
            "error": "timeout",
            "message": "Oxford API request timed out."
        }

    except requests.exceptions.ConnectionError:

        return {
            "error": "connection_error",
            "message": "Could not connect to Oxford API."
        }

    except requests.exceptions.RequestException as e:

        return {
            "error": "request_error",
            "message": str(e)
        }


# ============================================================
# 1. WORDS API
# ============================================================

@lru_cache(maxsize=500)
def get_word(word):

    endpoint = f"/words/{LANGUAGE}"

    params = {
        "q": word.strip().lower()
    }

    return _request(
        endpoint,
        params
    )


# ============================================================
# 2. SEARCH API
# ============================================================

@lru_cache(maxsize=500)
def search_word(word):

    endpoint = f"/search/{LANGUAGE}"

    params = {
        "q": word.strip().lower()
    }

    return _request(
        endpoint,
        params
    )


# ============================================================
# 3. LEMMAS API
# ============================================================

@lru_cache(maxsize=500)
def get_lemma(word):

    endpoint = (
        f"/lemmas/"
        f"{LANGUAGE}/"
        f"{word.strip().lower()}"
    )

    return _request(endpoint)


# ============================================================
# 4. INFLECTIONS API
# ============================================================

@lru_cache(maxsize=500)
def get_inflections(word):

    endpoint = (
        f"/inflections/"
        f"{LANGUAGE}/"
        f"{word.strip().lower()}"
    )

    return _request(endpoint)


# ============================================================
# 5. THESAURUS API
# ============================================================

@lru_cache(maxsize=500)
def get_thesaurus(word):

    endpoint = (
        f"/thesaurus/"
        f"{LANGUAGE}/"
        f"{word.strip().lower()}"
    )

    return _request(endpoint)


# ============================================================
# 6. SENTENCES API
# ============================================================

@lru_cache(maxsize=500)
def get_sentences(word):

    endpoint = (
        f"/sentences/"
        f"{LANGUAGE}/"
        f"{word.strip().lower()}"
    )

    return _request(endpoint)


# ============================================================
# 7. TRANSLATION API
# ============================================================

@lru_cache(maxsize=500)
def translate_word(
    word,
    source_language="en",
    target_language="hi"
):

    endpoint = (
        f"/translations/"
        f"{source_language}/"
        f"{target_language}/"
        f"{word.strip().lower()}"
    )

    return _request(endpoint)


# ============================================================
# 8. GET ALL BASIC INFORMATION
# ============================================================

def analyze_word(word):

    """
    Get the most useful Oxford information
    about a word.

    NOTE:
    We don't automatically call every API here
    to avoid unnecessary API usage.
    """

    result = {
        "word": word,
        "dictionary": get_word(word),
        "lemma": get_lemma(word),
        "inflections": get_inflections(word)
    }

    return result


# ============================================================
# 9. COMPLETE WORD ANALYSIS
# ============================================================

def complete_word_analysis(word):

    """
    Get extended information about a word.

    This calls more Oxford endpoints, so use it
    selectively rather than for every word.
    """

    result = {

        "word": word,

        "dictionary": get_word(word),

        "search": search_word(word),

        "lemma": get_lemma(word),

        "inflections": get_inflections(word),

        "thesaurus": get_thesaurus(word),

        "sentences": get_sentences(word)

    }

    return result


# ============================================================
# 10. SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("OXFORD DICTIONARIES API TEST")
    print("=" * 60)

    word = input(
        "\nEnter a word to test: "
    ).strip()

    if not word:

        print("Please enter a word.")

    else:

        print(
            f"\nSearching Oxford for: {word}"
        )

        result = get_word(word)

        print("\nAPI Response:")
        print(result)