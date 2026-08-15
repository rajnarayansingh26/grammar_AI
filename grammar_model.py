import re
import difflib
import torch

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "rajnarayansingh26/grammar-corrector"

# Render deployment = CPU
DEVICE = torch.device("cpu")

# Keep input/output small to reduce RAM usage
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 128

# Model objects are loaded only when needed
tokenizer = None
model = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global tokenizer
    global model

    if tokenizer is not None and model is not None:
        return

    print("Loading Grammar AI...")

    # Explicitly request the SentencePiece tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME,
        legacy=True,
        use_fast=False
    )

    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True
    )

    model.to(DEVICE)

    model.eval()

    print("Grammar AI loaded successfully on CPU")


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_text(text):

    if not text:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# SPLIT PARAGRAPH
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

    load_model()

    sentence = preprocess_text(sentence)

    if not sentence:
        return ""

    input_text = "grammar: " + sentence

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=2,
            do_sample=False,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    corrected = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Delete temporary tensors
    del inputs
    del outputs

    return corrected.strip()


# ============================================================
# CORRECT PARAGRAPH
# ============================================================

def correct_grammar(text):

    text = preprocess_text(text)

    if not text:
        return ""

    sentences = split_sentences(text)

    corrected_sentences = []

    for sentence in sentences:

        corrected = correct_sentence(
            sentence
        )

        if corrected:
            corrected_sentences.append(
                corrected
            )

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

    if not text:
        return {
            "original": "",
            "corrected": "",
            "changes": [],
            "similarity": 100.0,
            "changed": False
        }

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
