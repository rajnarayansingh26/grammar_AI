
from flask import Flask, render_template, request, jsonify

from grammar_model import analyze_grammar
from oxford_api import (
    get_word,
    get_lemma,
    get_inflections,
    get_thesaurus,
    get_sentences
)

import re


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# EXTRACT WORDS
# ============================================================

def extract_words(text):

    return re.findall(
        r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b",
        text
    )


# ============================================================
# OXFORD INFORMATION
# ============================================================

def get_oxford_information(word):

    """
    Get useful Oxford information for a word.

    We call only selected APIs so that we don't
    unnecessarily consume API requests.
    """

    result = {
        "word": word,
        "dictionary": None,
        "lemma": None,
        "inflections": None,
        "thesaurus": None,
        "sentences": None
    }

    try:
        result["dictionary"] = get_word(word)
    except Exception as e:
        result["dictionary"] = {
            "error": str(e)
        }

    try:
        result["lemma"] = get_lemma(word)
    except Exception as e:
        result["lemma"] = {
            "error": str(e)
        }

    try:
        result["inflections"] = get_inflections(word)
    except Exception as e:
        result["inflections"] = {
            "error": str(e)
        }

    try:
        result["thesaurus"] = get_thesaurus(word)
    except Exception as e:
        result["thesaurus"] = {
            "error": str(e)
        }

    try:
        result["sentences"] = get_sentences(word)
    except Exception as e:
        result["sentences"] = {
            "error": str(e)
        }

    return result


# ============================================================
# EXTRACT SIMPLE DICTIONARY INFORMATION
# ============================================================

def extract_dictionary_info(data):

    if not data:
        return {}

    if "error" in data:
        return {
            "error": data.get("message", "No data")
        }

    output = {
        "word": data.get("word"),
        "definitions": [],
        "examples": [],
        "pronunciation": [],
        "parts_of_speech": []
    }

    for result in data.get("results", []):

        for lexical_entry in result.get(
            "lexicalEntries",
            []
        ):

            lexical_category = lexical_entry.get(
                "lexicalCategory",
                {}
            ).get("text")

            if lexical_category:
                output["parts_of_speech"].append(
                    lexical_category
                )

            for entry in lexical_entry.get(
                "entries",
                []
            ):

                for pronunciation in entry.get(
                    "pronunciations",
                    []
                ):

                    phonetic = pronunciation.get(
                        "phoneticSpelling"
                    )

                    if phonetic:
                        output[
                            "pronunciation"
                        ].append(phonetic)

                for sense in entry.get(
                    "senses",
                    []
                ):

                    for definition in sense.get(
                        "definitions",
                        []
                    ):

                        output[
                            "definitions"
                        ].append(definition)

                    for example in sense.get(
                        "examples",
                        []
                    ):

                        example_text = example.get(
                            "text"
                        )

                        if example_text:
                            output[
                                "examples"
                            ].append(example_text)

    # Remove duplicates

    output["definitions"] = list(
        dict.fromkeys(
            output["definitions"]
        )
    )[:5]

    output["examples"] = list(
        dict.fromkeys(
            output["examples"]
        )
    )[:5]

    output["parts_of_speech"] = list(
        dict.fromkeys(
            output["parts_of_speech"]
        )
    )

    output["pronunciation"] = list(
        dict.fromkeys(
            output["pronunciation"]
        )
    )

    return output


# ============================================================
# FIND WORDS THAT CHANGED
# ============================================================

def get_changed_words(grammar_result):

    changed_words = []

    for change in grammar_result.get(
        "changes",
        []
    ):

        original = change.get(
            "original",
            ""
        )

        corrected = change.get(
            "corrected",
            ""
        )

        original_words = extract_words(
            original
        )

        corrected_words = extract_words(
            corrected
        )

        for word in original_words:
            if word.lower() not in [
                x.lower()
                for x in changed_words
            ]:
                changed_words.append(word)

        for word in corrected_words:
            if word.lower() not in [
                x.lower()
                for x in changed_words
            ]:
                changed_words.append(word)

    return changed_words


# ============================================================
# MAIN GRAMMAR API
# ============================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "error": "No JSON data received."
            }), 400

        sentence = data.get(
            "sentence",
            ""
        ).strip()

        if not sentence:

            return jsonify({
                "success": False,
                "error": "Please enter a sentence."
            }), 400

        # ====================================================
        # STEP 1
        # ML GRAMMAR CORRECTION
        # ====================================================

        grammar_result = analyze_grammar(
            sentence
        )

        # ====================================================
        # STEP 2
        # FIND CHANGED WORDS
        # ====================================================

        changed_words = get_changed_words(
            grammar_result
        )

        # Limit Oxford calls
        # This is especially useful with Sandbox.
        changed_words = changed_words[:5]

        # ====================================================
        # STEP 3
        # OXFORD ANALYSIS
        # ====================================================

        oxford_results = {}

        for word in changed_words:

            # Skip very short words
            if len(word) <= 2:
                continue

            oxford_data = get_oxford_information(
                word
            )

            oxford_results[word] = {
                "dictionary":
                    extract_dictionary_info(
                        oxford_data["dictionary"]
                    ),

                "lemma":
                    oxford_data["lemma"],

                "inflections":
                    oxford_data["inflections"],

                "thesaurus":
                    oxford_data["thesaurus"],

                "sentences":
                    oxford_data["sentences"]
            }

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        return jsonify({

            "success": True,

            "original":
                grammar_result["original"],

            "corrected":
                grammar_result["corrected"],

            "changed":
                grammar_result["changed"],

            "similarity":
                grammar_result["similarity"],

            "changes":
                grammar_result["changes"],

            "changed_words":
                changed_words,

            "oxford":
                oxford_results

        })

    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "service": "Grammar AI"
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000,
        debug=True
    )
