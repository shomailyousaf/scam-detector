# shared fixture: a tiny pipeline with the same shape as the production model
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TRAIN = [
    ("Apna OTP abhi bhejo warna account band ho jayega", 1),
    ("Win money now click http://x.com claim your prize", 1),
    ("aap ka inaam nikla hai jaldi paisay bhejein", 1),
    ("ہمیں اپنا او ٹی پی بھیجیں ورنہ اکاؤنٹ بند", 1),
    ("Send your bank account number to receive the lottery money", 1),
    ("Kal class 10 baje hai notes le aana", 0),
    ("Meeting at 5 pm see you there", 0),
    ("کل کلاس ہے نوٹس لے آنا", 0),
    ("dinner tonight? bring bread", 0),
    ("ammi ne kaha ghar jaldi aana", 0),
]


@pytest.fixture(scope="module")
def pipe():
    X, y = zip(*TRAIN)
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(C=4, class_weight="balanced", solver="liblinear", random_state=42)),
    ]).fit(list(X), list(y))
