import spacy
from spacy import displacy
from pathlib import Path

nlp = spacy.load('en_core_web_sm')

def visualize(doc):
    colors = {"ORG": "green", "PERSON":"yellow"}
    options = {"colors": colors}
    displacy.serve(doc, style='ent', options=options)

def save_as_html(doc, path):
    html = displacy.render(doc, style="ent")
    html_file= open(path, "w", encoding="utf-8")
    html_file.write(html)
    html_file.close()

def main():
    text = """SpaceX is an aerospace manufacturer and space transport services company headquartered in California. It was founded in 2002 by entrepreneur and investor Elon Musk with the goal of reducing space transportation costs and enabling the colonization of Mars."""
    doc = nlp(text)
    doc.user_data["title"] = "Random Text"
    visualize(doc)
    save_as_html(doc, "./ner_vis.html")

if (__name__ == "__main__"):
    main()

