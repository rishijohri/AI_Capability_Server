"""Objectivity detection using spaCy NLP pipeline for filtering factual content."""

import spacy

# Load small English model (dependency parsing, POS tagging, NER)
_nlp = spacy.load("en_core_web_sm")


def _is_question(doc) -> bool:
    """Detect questions via dependency parse and POS tags."""
    # Direct question mark
    if doc.text.strip().endswith("?"):
        return True
    # Sentence starts with a WH-word (what, who, where, when, why, how)
    first_token = doc[0] if len(doc) > 0 else None
    if first_token and first_token.tag_ in ("WP", "WP$", "WRB", "WDT"):
        return True
    # Aux-initial (inverted) question: "Is this ...", "Can you ...", "Do they ..."
    if first_token and first_token.dep_ == "aux" and first_token.head.pos_ == "VERB":
        return True
    return False


def _is_imperative(doc) -> bool:
    """Detect imperative / command sentences via dependency parse."""
    if len(doc) == 0:
        return False
    first_token = doc[0]
    # Imperatives start with a base-form verb (VB) that is the root
    if first_token.tag_ == "VB" and first_token.dep_ == "ROOT":
        return True
    # "Please ..." followed by a verb
    if first_token.lower_ == "please" and len(doc) > 1 and doc[1].tag_ == "VB":
        return True
    return False


def _is_conversational(doc) -> bool:
    """Detect greetings, thanks, and filler using NER + lexical cues."""
    lowered = doc.text.strip().lower()
    # Short conversational tokens
    greetings = {
        "hi", "hello", "hey", "thanks", "thank", "thank you", "ok", "okay", "sure",
        "yes", "no", "bye", "goodbye", "welcome", "sorry",
    }
    # Check against the full text and each sentence
    texts_to_check = [lowered]
    for sent in doc.sents:
        texts_to_check.append(sent.text.strip().lower())
    for text in texts_to_check:
        first_words = text.split()[:2]
        if first_words and first_words[0] in greetings:
            return True
        if " ".join(first_words) in greetings:
            return True
    return False


def _has_named_entities(doc) -> bool:
    """Check if the text contains named entities (a good signal for factual content)."""
    # Entities that signal factual content
    useful_labels = {
        "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY",
        "QUANTITY", "PERCENT", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
        "LANGUAGE", "FAC", "NORP",
    }
    return any(ent.label_ in useful_labels for ent in doc.ents)


def _is_declarative(doc) -> bool:
    """Check if the sentence is declarative (subject + verb structure)."""
    has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in doc)
    has_verb = any(tok.pos_ == "VERB" or tok.pos_ == "AUX" for tok in doc)
    return has_subject and has_verb


def is_objective(text: str, threshold: float = 0.3) -> tuple[bool, float]:
    """
    Determine if a text message is objective/factual enough to store as knowledge.

    Uses spaCy's dependency parser, POS tagger, and NER to classify text
    as factual vs non-factual based on linguistic structure rather than regex.

    A message is considered storable when it is:
      - A declarative sentence (subject + verb)
      - NOT a question, imperative/command, or conversational filler
      - Contains named entities OR has low TextBlob subjectivity

    Args:
        text: The text to analyze
        threshold: Subjectivity threshold (0.0-1.0) as a secondary filter.

    Returns:
        Tuple of (is_objective: bool, subjectivity_score: float)
    """
    if not text or len(text.strip()) < 20:
        return False, 1.0

    doc = _nlp(text.strip())

    # Reject questions
    if _is_question(doc):
        return False, 1.0

    # Reject imperatives / commands
    if _is_imperative(doc):
        return False, 1.0

    # Reject conversational filler
    if _is_conversational(doc):
        return False, 1.0

    # Must be declarative (subject + verb)
    if not _is_declarative(doc):
        return False, 0.9

    # If named entities present, strong factual signal — accept regardless of subjectivity
    if _has_named_entities(doc):
        return True, 0.0

    # Otherwise fall back to TextBlob subjectivity as final filter
    from textblob import TextBlob
    subjectivity = TextBlob(text.strip()).sentiment.subjectivity

    # Detect superlatives ("best", "worst", "greatest") — strong opinion signal
    if any(tok.tag_ == "JJS" or tok.tag_ == "RBS" for tok in doc):
        subjectivity = max(subjectivity, 0.5)

    return subjectivity <= threshold, subjectivity
