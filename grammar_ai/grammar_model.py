import re
import difflib
import torch

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "rajnarayansingh26/grammar-corrector"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Grammar AI model...")

tokenizer = T5Tokenizer.from_pretrained(
    MODEL_NAME
)

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)

model.eval()

print(
    f"Grammar AI loaded on: {DEVICE}"
)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = text.strip()

    # Remove unnecessary spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# GRAMMAR CORRECTION
# ============================================================

def correct_grammar(text):

    text = preprocess_text(text)

    if not text:
        return ""

    # T5 grammar prompt
    input_text = (
        "grammar: "
        + text
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_length=256,

            num_beams=5,

            no_repeat_ngram_size=2,

            early_stopping=True
        )

    corrected_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return corrected_text.strip()


# ============================================================
# TOKENIZE SENTENCE
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
# CALCULATE CONFIDENCE
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
# GRAMMAR ANALYSIS
# ============================================================

def analyze_grammar(text):

    text = preprocess_text(text)

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

        "changed": text.lower()
        != corrected.lower()

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "MACHINE LEARNING GRAMMAR AI"
    )

    print("=" * 60)

    while True:

        sentence = input(
            "\nEnter sentence "
            "(type exit to stop): "
        )

        if sentence.lower() == "exit":
            break

        result = analyze_grammar(
            sentence
        )

        print(
            "\nOriginal:"
        )

        print(
            result["original"]
        )

        print(
            "\nCorrected:"
        )

        print(
            result["corrected"]
        )

        print(
            "\nSimilarity:",
            result["similarity"],
            "%"
        )

        print(
            "\nChanges:"
        )

        for change in result["changes"]:

            print(
                f"{change['original']} "
                f"→ "
                f"{change['corrected']}"
            )