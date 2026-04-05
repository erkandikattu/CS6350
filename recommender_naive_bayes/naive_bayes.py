import math
import re
from urllib.request import urlopen
from pyspark.sql import SparkSession

# =========================================================
# Naive Bayes Classifier using Spark MapReduce (RDD-based)
# Dataset: SMS Spam Collection
# =========================================================

# ---------------------------
# 1. Create or reuse Spark session
# ---------------------------
try:
    spark
except NameError:
    spark = SparkSession.builder.appName("NaiveBayes_SMS_Spam").getOrCreate()

sc = spark.sparkContext

# ---------------------------
# 2. Public dataset path
# ---------------------------
data_path = "https://raw.githubusercontent.com/erkandikattu/CS6350/refs/heads/main/recommender_naive_bayes/SMSSpamCollection"

# ---------------------------
# 3. Stopword list
# ---------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "is", "are", "to", "of", "in", "on", "for",
    "at", "be", "am", "it", "this", "that", "with", "as", "by", "from", "was",
    "were", "will", "would", "can", "could", "you", "your", "yours", "i", "me",
    "my", "mine", "we", "our", "ours", "he", "him", "his", "she", "her", "hers",
    "they", "them", "their", "theirs", "have", "has", "had", "do", "does", "did",
    "not", "but", "if", "so", "up", "out", "all", "just", "now", "then", "than",
    "too", "very", "been", "being", "into", "over", "under", "again", "further"
}

# ---------------------------
# 4. Text preprocessing
# ---------------------------
def preprocess(text):
    text = text.lower()
    tokens = re.findall(r"[a-z0-9']+", text)
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return tokens

def parse_line(line):
    parts = line.split("\t", 1)
    if len(parts) != 2:
        return None

    label = parts[0].strip().lower()
    message = parts[1].strip()

    if label not in {"ham", "spam"}:
        return None

    tokens = preprocess(message)
    if not tokens:
        return None

    return (label, tokens)

def load_lines(sc, path):
    """
    Supports:
    1. Public raw URL
    2. Regular Spark-readable path
    """
    if path.startswith("http://") or path.startswith("https://"):
        text = urlopen(path).read().decode("utf-8", errors="ignore")
        return sc.parallelize(text.splitlines())
    else:
        return sc.textFile(path)

# ---------------------------
# 5. Train Naive Bayes model
# ---------------------------
def train_naive_bayes(train_rdd):
    # Count documents in each class
    class_doc_counts = dict(
        train_rdd
        .map(lambda x: (x[0], 1))
        .reduceByKey(lambda a, b: a + b)
        .collect()
    )

    total_docs = sum(class_doc_counts.values())

    # Count each word occurrence by class: ((class, word), count)
    word_counts = dict(
        train_rdd
        .flatMap(lambda x: [((x[0], word), 1) for word in x[1]])
        .reduceByKey(lambda a, b: a + b)
        .collect()
    )

    # Total number of words in each class
    total_words_per_class = {}
    for (label, word), count in word_counts.items():
        total_words_per_class[label] = total_words_per_class.get(label, 0) + count

    # Vocabulary size from training set only
    vocab_size = (
        train_rdd
        .flatMap(lambda x: x[1])
        .distinct()
        .count()
    )

    # Prior probabilities P(class)
    priors = {
        label: class_doc_counts[label] / total_docs
        for label in class_doc_counts
    }

    model = {
        "priors": priors,
        "word_counts": word_counts,
        "total_words_per_class": total_words_per_class,
        "vocab_size": vocab_size,
        "classes": sorted(priors.keys())
    }

    return model

# ---------------------------
# 6. Smoothed word probability
# ---------------------------
def word_probability(word, label, model):
    count_wc = model["word_counts"].get((label, word), 0)
    total_words_in_class = model["total_words_per_class"][label]
    vocab_size = model["vocab_size"]

    # Laplace smoothing
    return (count_wc + 1) / (total_words_in_class + vocab_size)

# ---------------------------
# 7. Predict class for one document
# ---------------------------
def predict(tokens, model):
    best_label = None
    best_score = float("-inf")

    for label in model["classes"]:
        score = math.log(model["priors"][label])

        for word in tokens:
            score += math.log(word_probability(word, label, model))

        if score > best_score:
            best_score = score
            best_label = label

    return best_label

# ---------------------------
# 8. Evaluate on test set
# ---------------------------
def evaluate(test_rdd, model):
    predictions = test_rdd.map(lambda x: (x[0], predict(x[1], model))).cache()

    accuracy = predictions.map(lambda x: 1 if x[0] == x[1] else 0).mean()

    confusion_matrix = (
        predictions
        .map(lambda x: ((x[0], x[1]), 1))
        .reduceByKey(lambda a, b: a + b)
        .collect()
    )

    return accuracy, sorted(confusion_matrix), predictions

# ---------------------------
# 9. Load and preprocess data
# ---------------------------
lines = load_lines(sc, data_path)

parsed_data = (
    lines
    .map(parse_line)
    .filter(lambda x: x is not None)
    .cache()
)

# ---------------------------
# 10. Split into training and testing
# ---------------------------
train_rdd, test_rdd = parsed_data.randomSplit([0.8, 0.2], seed=42)
train_rdd = train_rdd.cache()
test_rdd = test_rdd.cache()

# ---------------------------
# 11. Train model
# ---------------------------
model = train_naive_bayes(train_rdd)

# ---------------------------
# 12. Evaluate model
# ---------------------------
accuracy, confusion_matrix, predictions = evaluate(test_rdd, model)

# ---------------------------
# 13. Output results
# ---------------------------
print("==========================================")
print("Naive Bayes Classifier using Spark MapReduce")
print("Dataset: SMS Spam Collection")
print("==========================================\n")

print("Total Documents:", parsed_data.count())
print("Training Documents:", train_rdd.count())
print("Testing Documents:", test_rdd.count())
print("Vocabulary Size:", model["vocab_size"])

print("\nClass Priors:")
for label in model["classes"]:
    print(f"{label}: {model['priors'][label]:.6f}")

print(f"\nAccuracy: {accuracy:.6f}")

print("\nConfusion Matrix Counts (true_label, predicted_label):")
for item in confusion_matrix:
    print(item)

print("\nSample Predictions:")
for item in predictions.take(10):
    print(item)
