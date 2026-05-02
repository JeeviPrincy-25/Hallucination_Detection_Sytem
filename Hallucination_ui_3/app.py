# ==========================================================
# UPDATED app.py
# Uses your NEW trained model + notebook prediction logic
# Keep UI templates same
# Added:
#   - top 10 evidence for Wikipedia content panel
#   - top 1 evidence for comparison logic
# ==========================================================

from flask import Flask, render_template, request
import re
import wikipediaapi
import spacy
import torch
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================================
# APP
# ==========================================================
app = Flask(__name__)

# ==========================================================
# LOAD NLP
# ==========================================================
nlp = spacy.load("en_core_web_sm")

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================
MODEL_PATH = r"C:\Users\HP\Downloads\College\sem6\NLP-mini\model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ==========================================================
# SBERT RETRIEVER
# ==========================================================
retriever = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================================
# WIKIPEDIA
# ==========================================================
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="HallucinationDetector/20.0"
)

# ==========================================================
# CLEAN
# ==========================================================
def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==========================================================
# GET WIKI TEXT
# ==========================================================
@lru_cache(maxsize=100)
def get_wiki_text(title):
    try:
        page = wiki.page(title)
        if page.exists():
            return page.text[:5000]
    except:
        pass
    return "No trusted source found."

# ==========================================================
# SENTENCE SPLIT
# ==========================================================
def split_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 20]

# ==========================================================
# BEST EVIDENCE
# ==========================================================
def get_best_evidence(query, wiki_text, top_k=5):

    sentences = split_sentences(wiki_text)

    if not sentences:
        return [(0, "No evidence found.")]

    q_emb = retriever.encode(query, convert_to_tensor=True)
    s_emb = retriever.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, s_emb)[0]

    ranked = []

    for i in range(len(sentences)):
        ranked.append((float(scores[i]), sentences[i]))

    ranked = sorted(ranked, reverse=True, key=lambda x: x[0])

    return ranked[:top_k]

# ==========================================================
# ENTITY SCORE
# ==========================================================
def fuzzy_entity_score(evidence, llm_output):

    ev_doc = nlp(evidence)
    llm_doc = nlp(llm_output)

    ev_ents = set(ent.text.lower() for ent in ev_doc.ents)
    llm_ents = set(ent.text.lower() for ent in llm_doc.ents)

    if len(llm_ents) == 0:
        return 0.5

    overlap = len(ev_ents & llm_ents)

    return overlap / len(llm_ents)

# ==========================================================
# MAIN PREDICTION FUNCTION
# ==========================================================
def predict_hallucination(title, query, llm_output):

    wiki_text = get_wiki_text(title)
    top1_list = get_best_evidence(query, wiki_text, top_k=1)

    score, evidence = top1_list[0]

    q = clean(query)
    llm = clean(llm_output)
    wiki = clean(evidence)

    # --------------------------
    # WHO question
    # --------------------------
    if q.startswith("who"):

        llm_doc = nlp(llm_output)
        wiki_doc = nlp(evidence)

        llm_ans = " ".join(
            ent.text.lower()
            for ent in llm_doc.ents
            if ent.label_ == "PERSON"
        )

        wiki_ans = " ".join(
            ent.text.lower()
            for ent in wiki_doc.ents
            if ent.label_ == "PERSON"
        )

    # --------------------------
    # WHEN question
    # --------------------------
    elif q.startswith("when"):

        llm_ans = " ".join(re.findall(r'\d{4}', llm_output))
        wiki_ans = " ".join(re.findall(r'\d{4}', evidence))

    # --------------------------
    # WHERE / CAPITAL
    # --------------------------
    elif q.startswith("where") or "capital" in q:

        llm_doc = nlp(llm_output)
        wiki_doc = nlp(evidence)

        llm_ans = " ".join(
            ent.text.lower()
            for ent in llm_doc.ents
            if ent.label_ in ["GPE", "LOC"]
        )

        wiki_ans = " ".join(
            ent.text.lower()
            for ent in wiki_doc.ents
            if ent.label_ in ["GPE", "LOC"]
        )

    else:
        llm_ans = llm
        wiki_ans = wiki

    # fallback
    if llm_ans.strip() == "":
        llm_ans = llm

    if wiki_ans.strip() == "":
        wiki_ans = wiki

    # --------------------------
    # partial inclusion logic
    # --------------------------
    if llm_ans in wiki_ans or wiki_ans in llm_ans:
        sim = 0.95
    else:
        emb1 = retriever.encode(llm_ans, convert_to_tensor=True)
        emb2 = retriever.encode(wiki_ans, convert_to_tensor=True)
        sim = float(util.cos_sim(emb1, emb2)[0][0])

    # --------------------------
    # prediction
    # --------------------------
    if sim >= 0.75:
        pred = "Not Hallucinated"
    elif sim >= 0.45:
        pred = "Partially Verified"
    else:
        pred = "Hallucinated"

    return {
        "Prediction": pred,
        "Confidence": round(sim * 100, 2),
        "Evidence": evidence,
        "Semantic Similarity": round(sim, 3),
        "Entity Match": round(sim, 3)
    }

# ==========================================================
# HOME
# ==========================================================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ==========================================================
# ANALYZE
# ==========================================================
@app.route("/analyze", methods=["POST"])
def analyze():

    # ======================================================
    # GET INPUTS
    # ======================================================
    topic = request.form.get("topic", "").strip()
    query = request.form.get("query", "").strip()
    llm_output = request.form.get("llm_output", "").strip()

    topic_key = topic.lower()
    AMBIGUOUS_TERMS = {

    "python": [
        {"label": "Python Programming Language", "value": "Python (programming language)"},
        {"label": "Python Snake", "value": "Pythonidae"},
        {"label": "Python Mythology", "value": "Python (mythology)"}
    ],

    "apple": [
        {"label": "Apple Company", "value": "Apple Inc."},
        {"label": "Apple Fruit", "value": "Apple"},
        {"label": "Apple Music Label", "value": "Apple Records"}
    ],

    "java": [
        {"label": "Java Programming Language", "value": "Java (programming language)"},
        {"label": "Java Island", "value": "Java"},
        {"label": "Coffee", "value": "Coffee"}
    ],

    "c": [
        {"label": "C Programming Language", "value": "C (programming language)"},
        {"label": "Vitamin C", "value": "Vitamin C"}
    ],

    "ruby": [
        {"label": "Ruby Programming Language", "value": "Ruby (programming language)"},
        {"label": "Ruby Gemstone", "value": "Ruby"}
    ],

    "go": [
        {"label": "Go Programming Language", "value": "Go (programming language)"},
        {"label": "Go Board Game", "value": "Go (game)"}
    ],

    "swift": [
        {"label": "Swift Programming Language", "value": "Swift (programming language)"},
        {"label": "Swift Bird", "value": "Swift (bird)"}
    ],

    "rust": [
        {"label": "Rust Programming Language", "value": "Rust (programming language)"},
        {"label": "Rust Chemical", "value": "Rust"}
    ],

    "r": [
        {"label": "R Programming Language", "value": "R (programming language)"},
        {"label": "Letter R", "value": "R"}
    ],

    "scala": [
        {"label": "Scala Programming Language", "value": "Scala (programming language)"},
        {"label": "Scala Theatre", "value": "Scala Theatre"}
    ],

    "perl": [
        {"label": "Perl Programming Language", "value": "Perl"},
        {"label": "Pearl Gem", "value": "Pearl"}
    ],

    "matlab": [
        {"label": "MATLAB Software", "value": "MATLAB"},
        {"label": "Math Laboratory", "value": "MATLAB"}
    ],

    "oracle": [
        {"label": "Oracle Company", "value": "Oracle Corporation"},
        {"label": "Oracle Meaning", "value": "Oracle"}
    ],

    "windows": [
        {"label": "Microsoft Windows", "value": "Microsoft Windows"},
        {"label": "Window Glass", "value": "Window"}
    ],

    "linux": [
        {"label": "Linux Operating System", "value": "Linux"},
        {"label": "Linus Linux Kernel", "value": "Linux kernel"}
    ],

    "amazon": [
        {"label": "Amazon Company", "value": "Amazon (company)"},
        {"label": "Amazon River", "value": "Amazon River"},
        {"label": "Amazon Rainforest", "value": "Amazon rainforest"}
    ],

    "tesla": [
        {"label": "Tesla Company", "value": "Tesla, Inc."},
        {"label": "Nikola Tesla", "value": "Nikola Tesla"}
    ],

    "mercury": [
        {"label": "Mercury Planet", "value": "Mercury (planet)"},
        {"label": "Mercury Element", "value": "Mercury (element)"},
        {"label": "Mercury God", "value": "Mercury (mythology)"}
    ],

    "saturn": [
        {"label": "Saturn Planet", "value": "Saturn"},
        {"label": "Saturn Car Brand", "value": "Saturn Corporation"}
    ],

    "mars": [
        {"label": "Mars Planet", "value": "Mars"},
        {"label": "Mars Company", "value": "Mars, Incorporated"}
    ],

    "jaguar": [
        {"label": "Jaguar Animal", "value": "Jaguar"},
        {"label": "Jaguar Car", "value": "Jaguar Cars"}
    ],

    "mustang": [
        {"label": "Mustang Horse", "value": "Mustang"},
        {"label": "Ford Mustang", "value": "Ford Mustang"}
    ],

    "turbo": [
        {"label": "Turbo Engine", "value": "Turbocharger"},
        {"label": "Turbo Movie", "value": "Turbo (film)"}
    ],

    "bolt": [
        {"label": "Bolt Fastener", "value": "Bolt (fastener)"},
        {"label": "Usain Bolt", "value": "Usain Bolt"}
    ],

    "delta": [
        {"label": "Delta Airline", "value": "Delta Air Lines"},
        {"label": "River Delta", "value": "River delta"}
    ],

    "meta": [
        {"label": "Meta Company", "value": "Meta Platforms"},
        {"label": "Meta Meaning", "value": "Meta"}
    ],

    "byte": [
        {"label": "Byte Computer Unit", "value": "Byte"},
        {"label": "Bite Meaning", "value": "Bite"}
    ],

    "ram": [
        {"label": "RAM Memory", "value": "Random-access memory"},
        {"label": "Ram Animal", "value": "Ram (sheep)"}
    ],

    "mouse": [
        {"label": "Computer Mouse", "value": "Computer mouse"},
        {"label": "Mouse Animal", "value": "Mouse"}
    ],

    "virus": [
        {"label": "Computer Virus", "value": "Computer virus"},
        {"label": "Biological Virus", "value": "Virus"}
    ],

    "cloud": [
        {"label": "Cloud Computing", "value": "Cloud computing"},
        {"label": "Cloud Sky", "value": "Cloud"}
    ],

    "node": [
        {"label": "Node.js", "value": "Node.js"},
        {"label": "Node Meaning", "value": "Node"}
    ],

    "react": [
        {"label": "React JS", "value": "React (software)"},
        {"label": "Reaction Meaning", "value": "Reaction"}
    ],

    "django": [
        {"label": "Django Framework", "value": "Django (web framework)"},
        {"label": "Django Film", "value": "Django Unchained"}
    ],

    "flask": [
        {"label": "Flask Framework", "value": "Flask (web framework)"},
        {"label": "Flask Bottle", "value": "Flask"}
    ],

    "panda": [
        {"label": "Pandas Library", "value": "pandas (software)"},
        {"label": "Panda Animal", "value": "Giant panda"}
    ],

    "spark": [
        {"label": "Apache Spark", "value": "Apache Spark"},
        {"label": "Spark Fire", "value": "Spark"}
    ],

    "hadoop": [
        {"label": "Hadoop Software", "value": "Apache Hadoop"},
        {"label": "Hadoop Toy", "value": "Hadoop"}
    ],

    "excel": [
        {"label": "Microsoft Excel", "value": "Microsoft Excel"},
        {"label": "Excel Meaning", "value": "Excellence"}
    ],

    "word": [
        {"label": "Microsoft Word", "value": "Microsoft Word"},
        {"label": "Word Meaning", "value": "Word"}
    ],

    "powerpoint": [
        {"label": "Microsoft PowerPoint", "value": "Microsoft PowerPoint"},
        {"label": "Presentation Meaning", "value": "Presentation"}
    ],

    "chrome": [
        {"label": "Google Chrome", "value": "Google Chrome"},
        {"label": "Chrome Metal", "value": "Chromium"}
    ],

    "edge": [
        {"label": "Microsoft Edge", "value": "Microsoft Edge"},
        {"label": "Edge Meaning", "value": "Edge"}
    ],

    "opera": [
        {"label": "Opera Browser", "value": "Opera (web browser)"},
        {"label": "Opera Music", "value": "Opera"}
    ],

    "firefox": [
        {"label": "Firefox Browser", "value": "Firefox"},
        {"label": "Red Panda", "value": "Red panda"}
    ],

    "shell": [
        {"label": "Shell Company", "value": "Shell plc"},
        {"label": "Shell Command", "value": "Shell (computing)"},
        {"label": "Shell Object", "value": "Shell"}
    ],

    "ubuntu": [
        {"label": "Ubuntu OS", "value": "Ubuntu"},
        {"label": "Ubuntu Philosophy", "value": "Ubuntu philosophy"}
    ]
}
    # ======================================================
    # 50+ AMBIGUOUS WORDS CHECK
    # Only ask popup for those words
    # ======================================================
    if topic_key in AMBIGUOUS_TERMS and request.form.get("resolved") != "yes":

        return render_template(
            "disambiguate.html",
            topic=topic,
            query=query,
            llm_output=llm_output,
            options=AMBIGUOUS_TERMS[topic_key]
        )

    # ======================================================
    # USER SELECTED RADIO OPTION
    # topic internally changed here
    # ======================================================
    selected = request.form.get("topic_choice", "").strip()

    if selected != "":
        topic = selected

    # ======================================================
    # PREDICTION
    # ======================================================
    result = predict_hallucination(topic, query, llm_output)

    # ======================================================
    # FETCH WIKIPEDIA CONTENT
    # ======================================================
    wiki_text = get_wiki_text(topic)

    # ======================================================
    # TOP 10 EVIDENCE
    # ======================================================
    top10_list = get_best_evidence(
        query + " " + llm_output,
        wiki_text,
        top_k=10
    )

    trusted_content = ""

    for i, (score, sent) in enumerate(top10_list, start=1):
        trusted_content += f"{i}. {sent}\n\n"

    # ======================================================
    # TOP 1 EVIDENCE
    # ======================================================
    top1_evidence = result["Evidence"]

    paired_rows = [{
        "llm": llm_output,
        "wiki": top1_evidence,
        "status": "match" if result["Prediction"] == "Not Hallucinated" else "mismatch",
        "weight": "high",
        "type": "main claim"
    }]

    # ======================================================
    # FINAL OUTPUT
    # ======================================================
    return render_template(
        "result.html",

        query=query,
        topic=topic,
        llm_output=llm_output,

        result=result["Prediction"],
        confidence=result["Confidence"],
        reason="Compared with trusted Wikipedia evidence.",

        trusted_content=trusted_content,
        trusted_summary=trusted_content,
        evidence=top1_evidence,

        semantic_score=result["Semantic Similarity"],
        common_words=result["Entity Match"],

        main_claim=llm_output,

        mismatch="None",
        high_miss="None",
        low_miss="None",
        high_match="None",

        paired_rows=paired_rows
    )
# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)