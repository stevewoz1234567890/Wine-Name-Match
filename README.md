# Wine Name Match

Match **scraped or free-text wine product titles** to rows in a **standard wine catalog** using text similarity. The repo includes sample catalog data and two experimental Python approaches.

## What’s in the repo

| File | Role |
|------|------|
| `similarity1.py` | NLTK-based pipeline: tokenizes text, expands tokens with WordNet lemmas, stems with Porter, removes English stopwords, then scores catalog rows against a query string with a cosine-style similarity. Prints the **top 3** catalog names for one hard-coded example query. |
| `similarity2.py` | Sketch of a **TF–IDF + character 3-grams** matcher using `sparse_dot_topn` for fast top-*N* cosine similarity (useful for finding near-duplicate names in bulk). **Not runnable as-is** in this tree: it references placeholders like `standard_wize_table` and `room_types` that need to be wired to your wine name series. |
| `wine` | Semicolon-delimited catalog export (headers in the first line). |
| `items_1` | Example scraped/listing table used by `similarity1.py`. |
| `wines1.csv` | Additional wine-related CSV (not used by the scripts by default). |

## Requirements (`similarity1.py`)

- Python 3
- `pandas`, `nltk`

Install:

```bash
pip install pandas nltk
```

Download NLTK data once (WordNet and stopwords):

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Run

From the repository root (so `wine` and `items_1` resolve):

```bash
python similarity1.py
```

The script loads the catalog from `wine`, builds vectors from **Name, Varietal, Producer, Description, Type, subappellation**, plus the literal word `wine`, then compares them to a sample query string defined in the script.

## Requirements (`similarity2.py`)

If you adapt the sketch, you will typically need:

`pandas`, `numpy`, `scipy`, `scikit-learn`, `sparse-dot-topn`

## License

See [LICENSE](LICENSE) (MIT).
