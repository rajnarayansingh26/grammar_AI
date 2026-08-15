from flask import Flask, render_template, request, jsonify
from grammar_model import analyze_grammar
from oxford_api import get_word

import re
import gc


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# MEMORY / REQUEST LIMITS
# ============================================================

# Maximum characters accepted from the user.
# Adjust if required.
MAX_CHARACTERS = 5000

# Maximum number of words sent to Oxford API.
MAX_OXFORD_WORDS = 3


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
# EXTRACT USEFUL OXFORD DATA
# ============================================================

def extract_dictionary_info(data):

    if not data:
        return {}

    if "error" in data:

        return {
            "error": data.get(
                "message",
                "Oxford data unavailable"
            )
        }

    result = {

        "word": data.get(
            "word",
            ""
        ),

        "definitions": [],

        "examples": [],

        "parts_of_speech": [],

        "pronunciation": []

    }


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    for dictionary_result in data.get(
        "results",
        []
    ):

        lexical_entries = (
            dictionary_result.get(
                "lexicalEntries",
                []
            )
        )


        for lexical_entry in lexical_entries:

            # ----------------------------------------------
            # PART OF SPEECH
            # ----------------------------------------------

            category = lexical_entry.get(
                "lexicalCategory",
                {}
            ).get(
                "text"
            )

            if category:

                result[
                    "parts_of_speech"
                ].append(
                    category
                )


            # ----------------------------------------------
            # ENTRIES
            # ----------------------------------------------

            for entry in lexical_entry.get(
                "entries",
                []
            ):

                # ------------------------------------------
                # PRONUNCIATION
                # ------------------------------------------

                for pronunciation in entry.get(
                    "pronunciations",
                    []
                ):

                    phonetic = pronunciation.get(
                        "phoneticSpelling"
                    )

                    if phonetic:

                        result[
                            "pronunciation"
                        ].append(
                            phonetic
                        )


                # ------------------------------------------
                # SENSES
                # ------------------------------------------

                for sense in entry.get(
                    "senses",
                    []
                ):

                    # Definitions

                    for definition in sense.get(
                        "definitions",
                        []
                    ):

                        result[
                            "definitions"
                        ].append(
                            definition
                        )


                    # Examples

                    for example in sense.get(
                        "examples",
                        []
                    ):

                        example_text = example.get(
                            "text"
                        )

                        if example_text:

                            result[
                                "examples"
                            ].append(
                                example_text
                            )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    result[
        "definitions"
    ] = list(
        dict.fromkeys(
            result["definitions"]
        )
    )[:3]


    result[
        "examples"
    ] = list(
        dict.fromkeys(
            result["examples"]
        )
    )[:3]


    result[
        "parts_of_speech"
    ] = list(
        dict.fromkeys(
            result["parts_of_speech"]
        )
    )[:3]


    result[
        "pronunciation"
    ] = list(
        dict.fromkeys(
            result["pronunciation"]
        )
    )[:2]


    return result


# ============================================================
# FIND CHANGED WORDS
# ============================================================

def get_changed_words(grammar_result):

    changed_words = []

    changes = grammar_result.get(
        "changes",
        []
    )


    for change in changes:

        original = change.get(
            "original",
            ""
        )

        corrected = change.get(
            "corrected",
            ""
        )


        words = (
            extract_words(original)
            +
            extract_words(corrected)
        )


        for word in words:

            word_lower = word.lower()

            # Ignore very short words

            if len(word_lower) <= 2:
                continue


            # Avoid duplicates

            if word_lower not in [
                x.lower()
                for x in changed_words
            ]:

                changed_words.append(
                    word
                )


            # Stop after maximum

            if len(changed_words) >= MAX_OXFORD_WORDS:

                return changed_words


    return changed_words


# ============================================================
# ANALYZE
# ============================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    try:

        # ====================================================
        # GET REQUEST
        # ====================================================

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No input received."

            }), 400


        sentence = data.get(
            "sentence",
            ""
        )


        if not isinstance(
            sentence,
            str
        ):

            return jsonify({

                "success": False,

                "error":
                    "Input must be text."

            }), 400


        sentence = sentence.strip()


        # ====================================================
        # EMPTY INPUT
        # ====================================================

        if not sentence:

            return jsonify({

                "success": False,

                "error":
                    "Please enter some text."

            }), 400


        # ====================================================
        # LENGTH LIMIT
        # ====================================================

        if len(sentence) > MAX_CHARACTERS:

            return jsonify({

                "success": False,

                "error":
                    f"Text is too long. "
                    f"Maximum {MAX_CHARACTERS} characters."

            }), 400


        # ====================================================
        # MACHINE LEARNING
        # ====================================================

        grammar_result = analyze_grammar(
            sentence
        )


        # ====================================================
        # COPY ONLY SMALL RESULTS
        # ====================================================

        original = grammar_result.get(
            "original",
            ""
        )

        corrected = grammar_result.get(
            "corrected",
            ""
        )

        changes = grammar_result.get(
            "changes",
            []
        )

        similarity = grammar_result.get(
            "similarity",
            0
        )

        changed = grammar_result.get(
            "changed",
            False
        )


        # ====================================================
        # FIND CHANGED WORDS
        # ====================================================

        changed_words = get_changed_words(
            grammar_result
        )


        # ====================================================
        # OXFORD API
        # ====================================================

        oxford_results = {}


        for word in changed_words:

            try:

                dictionary_data = get_word(
                    word
                )


                useful_data = (
                    extract_dictionary_info(
                        dictionary_data
                    )
                )


                oxford_results[
                    word
                ] = useful_data


                # Delete large raw API response

                del dictionary_data


            except Exception as e:

                oxford_results[
                    word
                ] = {

                    "error":
                        str(e)

                }


        # ====================================================
        # CREATE RESPONSE
        # ====================================================

        response_data = {

            "success": True,

            "original": original,

            "corrected": corrected,

            "changed": changed,

            "similarity": similarity,

            "changes": changes,

            "changed_words":
                changed_words,

            "oxford":
                oxford_results

        }


        # ====================================================
        # CLEAN TEMPORARY MEMORY
        # ====================================================

        del grammar_result

        gc.collect()


        return jsonify(
            response_data
        )


    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )


        gc.collect()


        return jsonify({

            "success": False,

            "error":
                "An error occurred while "
                "processing the text."

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "running",

        "service":
            "Grammar AI"

    })


# ============================================================
# APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=10000,

        debug=False,

        threaded=False

    )