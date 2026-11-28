# turns a probability into a risk level and shows what pushed a message towards scam
#
# The production model is TfidfVectorizer(char_wb, 2-5 grams) -> LogisticRegression, so the
# logit is  intercept + sum_j(tfidf_j * coef_j).  Every character n-gram j lives inside one
# word (char_wb never crosses a space), so we can add the n-gram contributions up per word and
# the words then add up to the logit exactly. Positive = pushes towards SCAM, negative = NORMAL.
from collections import Counter

DEFAULT_THRESHOLDS = {"medium": 0.30, "high": 0.70}


def risk_level(prob, thresholds=None):
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if prob >= t["high"]:
        return "high"
    if prob >= t["medium"]:
        return "medium"
    return "low"


def _steps(pipe):
    return pipe.named_steps["tfidf"], pipe.named_steps["clf"]


def explain(pipe, text, top_k=8):
    vec, clf = _steps(pipe)
    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0])
    vocab = vec.vocabulary_
    analyzer = vec.build_analyzer()

    x = vec.transform([text]).tocoo()
    contrib = {int(j): float(v) * float(coef[j]) for j, v in zip(x.col, x.data)}
    doc_count = Counter(vocab[g] for g in analyzer(text) if g in vocab)

    # per word: share of every n-gram contribution that came from this word
    tokens = []
    for tok in text.split():
        grams = Counter(vocab[g] for g in analyzer(tok) if g in vocab)
        score = sum(contrib.get(j, 0.0) * c / doc_count[j] for j, c in grams.items() if doc_count[j])
        tokens.append({"text": tok, "score": round(score, 4)})

    # the same word may appear twice, keep one row for it in the top lists
    merged = {}
    for t in tokens:
        key = t["text"].lower().strip(".,!?;:\"'()[]")
        if key:
            merged[key] = merged.get(key, 0.0) + t["score"]
    ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    scam_words = [{"word": w, "score": round(s, 4)} for w, s in ranked if s > 0][:top_k]
    normal_words = [{"word": w, "score": round(s, 4)} for w, s in reversed(ranked) if s < 0][:top_k]

    # the strongest single character patterns, spaces shown as an underscore
    names = None
    grams = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    if grams:
        names = vec.get_feature_names_out()
    def show(j):
        return str(names[j]).replace(" ", "_")
    top_ngrams = [{"pattern": show(j), "score": round(s, 4)} for j, s in grams if s > 0][:top_k]
    low_ngrams = [{"pattern": show(j), "score": round(s, 4)} for j, s in reversed(grams) if s < 0][:top_k]

    return {
        "scam_words": scam_words,
        "normal_words": normal_words,
        "scam_patterns": top_ngrams,
        "normal_patterns": low_ngrams,
        "tokens": tokens,                 # in message order, used by the UI to highlight
        "intercept": round(intercept, 4),
        "logit": round(intercept + sum(contrib.values()), 4),
    }
