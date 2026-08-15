import re
import difflib
import torch

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "rajnarayansingh26/grammar-corrector"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading Grammar AI...")

tokenizer = T5Tokenizer.from_pretrained(
    MODEL_NAME
)

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)

model.eval()

print(f"Model loaded on {DEVICE}")


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_text(text):

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# SPLIT PARAGRAPH INTO SENTENCES
# ============================================================

def split_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# CORRECT ONE SENTENCE
# ============================================================

def correct_sentence(sentence):

    sentence = preprocess_text(
        sentence
    )

    if not sentence:
        return ""

    input_text = (
        "grammar: " +
        sentence
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=128,
        truncation=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_length=128,

            num_beams=3,

            no_repeat_ngram_size=2,

            early_stopping=True
        )

    corrected = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return corrected.strip()


# ============================================================
# CORRECT PARAGRAPH
# ============================================================

def correct_grammar(text):

    text = preprocess_text(text)

    if not text:
        return ""

    sentences = split_sentences(
        text
    )

    corrected_sentences = []

    for sentence in sentences:

        corrected = correct_sentence(
            sentence
        )

        if corrected:
            corrected_sentences.append(
                corrected
            )

        # Free unused GPU memory if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return " ".join(
        corrected_sentences
    )


# ============================================================
# TOKENIZE
# ============================================================

def tokenize(text):

    return re.findall(
        r"\b[\w']+\b|[.,!?;:]",
        text
    )


# ============================================================
# FIND CHANGES
# ============================================================

def find_changes(
    original,
    corrected
):

    original_tokens = tokenize(
        original
    )

    corrected_tokens = tokenize(
        corrected
    )

    matcher = difflib.SequenceMatcher(
        None,
        original_tokens,
        corrected_tokens
    )

    changes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            continue

        old_text = " ".join(
            original_tokens[i1:i2]
        )

        new_text = " ".join(
            corrected_tokens[j1:j2]
        )

        changes.append({

            "type": tag,

            "original": old_text,

            "corrected": new_text

        })

    return changes


# ============================================================
# SIMILARITY
# ============================================================

def calculate_similarity(
    original,
    corrected
):

    original_tokens = tokenize(
        original
    )

    corrected_tokens = tokenize(
        corrected
    )

    if not original_tokens:
        return 0.0

    matcher = difflib.SequenceMatcher(
        None,
        original_tokens,
        corrected_tokens
    )

    return round(
        matcher.ratio() * 100,
        2
    )


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_grammar(text):

    text = preprocess_text(
        text
    )

    corrected = correct_grammar(
        text
    )

    changes = find_changes(
        text,
        corrected
    )

    similarity = calculate_similarity(
        text,
        corrected
    )

    return {

        "original": text,

        "corrected": corrected,

        "changes": changes,

        "similarity": similarity,

        "changed":
            text.lower()
            != corrected.lower()

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\nGrammar AI")

    text = input(
        "\nEnter text:\n"
    )

    result = analyze_grammar(
        text
    )

    print("\nOriginal:")
    print(result["original"])

    print("\nCorrected:")
    print(result["corrected"])

    print(
        "\nSimilarity:",
        result["similarity"],
        "%"
    )

    print("\nChanges:")

    for change in result["changes"]:

        print(
            change["original"],
            "→",
            change["corrected"]
        )