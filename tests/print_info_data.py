from data import importdata
from pylatex import Tabular, Document
import os

#
# Generuje informacje o danych tekstowych i zapisuje do pliku daneinfo.tex oraz daneinfo.pdf
#
path = os.path.dirname(os.path.abspath(__file__))
dataset = ['abalone0_4', 'abalone0_4_16_29', 'abalone16_29', 'balance_scale', 'breast_cancer', 'bupa', 'car', 'cmc',
           'ecoli', 'german', 'glass', 'haberman', 'heart_cleveland', 'hepatitis', 'horse_colic', 'ionosphere',
           'new_thyroid', 'postoperative', 'seeds', 'solar_flare', 'transfusion', 'vehicle', 'vertebal', 'yeastME1',
           'yeastME2', 'yeastME3']
table1 = Tabular('|c|c|c|c|c|c|')
table1.add_hline()
table1.add_row(('Nazwa danych', 'L. el.', 'L. atrybutow', "Rozklad klas", "% calosci kl. mn.", 'IR'))
table1.add_hline()
for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    row = [data]
    row.extend(importdata.print_latex(db.data, db.target))
    table1.add_row(row)
    table1.add_hline()

doc = Document("daneinfo")
doc.append(table1)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
