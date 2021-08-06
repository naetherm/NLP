import spacy
from spacy import displacy
from pathlib import Path


nlp = spacy.load('en_core_web_sm')

def visualize(doc, is_list=False):
    options = {
        "add_lemma": True, 
        "compact": True, 
        "color": "green", 
        "collapse_punct": True, 
        "arrow_spacing": 10, 
        "bg": "#FFFFFF",
        "font": "Times",
        "distance": 120}
    if (is_list):
        displacy.serve(list(doc.sents), style='dep', options=options)
    else:
        displacy.serve(doc, style='dep', options=options)

def save_as_image(doc, path):
    output_path = Path(path)
    svg = displacy.render(doc, style="dep", jupyter=False)
    output_path.open("w", encoding="utf-8").write(svg)


def main():
    text = "SpaceX is an aerospace manufacturer and space transport services company headquartered in California."
    
    doc = nlp(text)
    visualize(doc)
    save_as_image(doc, "./dep.svg")

if (__name__ == "__main__"):
    main()
