import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

VAGUE_WORDS = ["fast", "easy", "simple", "efficient", "user-friendly", "flexible"]

def analyze_requirement(text):
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    sentences = list(doc.sents)

    vague_count = sum(1 for w in words if w in VAGUE_WORDS)
    passive_count = sum(1 for token in doc if token.dep_ == "auxpass")
    has_criteria = any(w in words for w in ["shall", "must", "criteria", "acceptance"])

    return {
        "vague_ratio": vague_count / max(len(words), 1),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "passive_voice_score": passive_count / max(len(sentences), 1),
        "missing_criteria": 0 if has_criteria else 1
    }

def compute_ambiguity_score(metrics):
    score = (0.4 * metrics["vague_ratio"] +
             0.3 * metrics["avg_sentence_length"]/50 +
             0.2 * metrics["passive_voice_score"] +
             0.1 * metrics["missing_criteria"])
    return min(score, 1.0)
