"""Extrae 16 tablas de aforos de la tesis oficial y comprueba sus sumas."""
from pathlib import Path
import csv
import re
import hashlib
import json
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'fuentes/Cerna_Loyola_2020.pdf'
URL = 'https://repositorio.upn.edu.pe/handle/11537/25244'

def main():
    reader = PdfReader(SOURCE)
    records, checks, detail = [], [], []
    for campaign, pages in [('ajuste', range(70, 78)), ('prueba', range(83, 91))]:
        for position, page in enumerate(pages):
            text = reader.pages[page-1].extract_text()
            table = int(re.search(r'Tabla\s+(\d+)', text).group(1))
            section = text[text.rfind('AV. '):]
            matches = re.findall(r'^([^\d\n]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$', section, re.M)
            assert len(matches) == 13, (page, len(matches))
            sums = [0]*4
            for category, *numbers in matches:
                values = list(map(int, numbers))
                sums = [a+b for a,b in zip(sums, values[:4])]
                detail.append(dict(campana=campaign,pagina=page,tabla=table,categoria=category.strip(),q1=values[0],q2=values[1],q3=values[2],q4=values[3],total=values[4]))
            published = int(re.search(r'TOTAL, VEH\.\s+(\d+)', section).group(1))
            movement = ['Andrés Zevallos S-N','Andrés Zevallos N-S','Hoyos Rubio E-O','Hoyos Rubio O-E'][position % 4]
            hour = 12 if position < 4 else 18
            checks.append(dict(pagina=page,tabla=table,total_publicado=published,total_calculado=sum(sums),correcto=sum(sums)==published))
            for interval, total in enumerate(sums):
                records.append(dict(campana=campaign,fecha='2018-10-06' if campaign=='ajuste' else '',fecha_descripcion='06/10/2018, p. 60' if campaign=='ajuste' else '13 de octubre, p. 83; año omitido en ese párrafo',movimiento=movement,hora_inicio=f'{hour:02d}:{interval*15:02d}',franja=hour,intervalo=interval,volumen=total,pagina=page,tabla=table,fuente=URL))
    for name, rows in [('aforos.csv',records),('categorias.csv',detail)]:
        with (ROOT/'datos'/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (ROOT/'datos/verificacion.json').write_text(json.dumps({'sha256_pdf':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'tablas':checks,'registros':len(records)},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(records)} intervalos; {len(checks)} tablas; discrepancias: {[c for c in checks if not c["correcto"]]}')

if __name__ == '__main__':
    main()
