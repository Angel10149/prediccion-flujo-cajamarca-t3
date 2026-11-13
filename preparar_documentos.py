from pathlib import Path
import json, csv, base64, html, sys, importlib.metadata
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import nbformat

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'resultados/modelos.json').read_text(encoding='utf-8'))

def table(doc, headers, rows):
    t=doc.add_table(rows=1, cols=len(headers));t.style='Table Grid'
    for cell,title in zip(t.rows[0].cells,headers):cell.text=title
    for row in rows:
        for cell,value in zip(t.add_row().cells,row):cell.text=str(value)
    for i,row in enumerate(t.rows):
        for cell in row.cells:
            props=cell._tc.get_or_add_tcPr()
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'DDEBF0' if i==0 else 'FFFFFF');props.append(shade)
            borders=OxmlElement('w:tcBorders')
            for side in ['top','left','bottom','right']:
                border=OxmlElement('w:'+side);border.set(qn('w:val'),'single');border.set(qn('w:sz'),'4');border.set(qn('w:color'),'D9D9D9');borders.append(border)
            props.append(borders)
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(4)
                for r in p.runs:r.font.size=Pt(9);r.bold=i==0
    return t

def build(md, target):
    doc=Document();s=doc.sections[0]
    s.page_width=Inches(8.5);s.page_height=Inches(11)
    s.top_margin=s.bottom_margin=Inches(.75);s.left_margin=s.right_margin=Inches(.8)
    for name in ['Normal','Title','Heading 1','Heading 2']:
        style=doc.styles[name];style.font.name='Calibri';style.font.color.rgb=RGBColor(0,0,0)
    doc.styles['Normal'].font.size=Pt(11)
    doc.styles['Normal'].paragraph_format.space_after=Pt(6)
    doc.styles['Normal'].paragraph_format.line_spacing=1.08
    doc.styles['Title'].font.size=Pt(20)
    doc.styles['Heading 1'].font.size=Pt(14)
    doc.styles['Heading 2'].font.size=Pt(12)
    footer=s.footer.paragraphs[0];footer.alignment=2
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    for block in md.strip().split('\n\n'):
        if block=='{{METRICAS}}':
            table(doc,['Modelo','CV MAE','Train MAE','MAE','RMSE','R²','MAPE %'],[[m['modelo'],f"{m['cv_mae']:.2f}",f"{m['train_mae']:.2f}",f"{m['MAE']:.2f}",f"{m['RMSE']:.2f}",f"{m['R2']:.3f}",f"{m['MAPE']:.2f}"] for m in data['metrics']])
        elif block.startswith('{{FIGURA_'):
            filename='predicciones.png' if 'PREDICCIONES' in block else 'errores.png'
            doc.add_picture(str(ROOT/'resultados'/filename),width=Inches(6.7))
        elif block.startswith('### '):doc.add_heading(block[4:],2)
        elif block.startswith('## '):doc.add_heading(block[3:],1)
        elif block.startswith('# '):doc.add_paragraph(block[2:],'Title')
        else:doc.add_paragraph(block.replace('**',''))
    doc.save(ROOT/target)

build((ROOT/'Paper_T3.md').read_text(encoding='utf-8'),'01_Paper_T3.docx')
for source,target in [('02_Respuestas_caso.md','02_Respuestas_caso.docx'),('03_Algoritmos_cuanticos.md','03_Algoritmos_cuanticos.docx')]:
    text=(ROOT.parent/source).read_text(encoding='utf-8-sig')
    (ROOT/source).write_text(text,encoding='utf-8')
    build(text,target)

requirements=['numpy','scikit-learn','matplotlib','pypdf']
(ROOT/'requisitos.txt').write_text('\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in requirements)+'\n',encoding='utf-8')
csv_text=(ROOT/'datos/aforos.csv').read_text(encoding='utf-8-sig')
training=(ROOT/'entrenar.py').read_text(encoding='utf-8').replace('ROOT=Path(__file__).resolve().parent','ROOT=Path.cwd()')
interface=(ROOT/'ABRIR_INTERFAZ.html').read_text(encoding='utf-8')
cells=[nbformat.v4.new_markdown_cell('# Pronóstico de aforos reales de Cajamarca\n\nEjecuta todas las celdas. El CSV de 64 registros está incorporado. Fuente de campo y datos auditables en la carpeta fuentes. La primera campaña ajusta los modelos y la segunda se reserva para prueba. La interfaz permite pronosticar y comparar con observaciones reales.'),
nbformat.v4.new_code_cell('%pip install -q '+ ' '.join((ROOT/'requisitos.txt').read_text().splitlines())),
nbformat.v4.new_code_cell("from pathlib import Path\nPath('datos').mkdir(exist_ok=True)\nPath('resultados').mkdir(exist_ok=True)\nPath('datos/aforos.csv').write_text("+repr(csv_text)+",encoding='utf-8-sig')\nprint('64 aforos históricos incorporados')"),
nbformat.v4.new_code_cell(training),
nbformat.v4.new_code_cell("from IPython.display import display, Image\ndisplay(Image('resultados/predicciones.png'))"),
nbformat.v4.new_code_cell("import html\nfrom IPython.display import HTML\npanel="+repr(interface)+"\npanel=panel.replace('<script src=\"modelos.js\"></script>','<script>'+Path('modelos.js').read_text(encoding='utf-8')+'</script>')\npanel=panel.replace('src=\"resultados/predicciones.png\"','src=\"data:image/png;base64,'+__import__('base64').b64encode(Path('resultados/predicciones.png').read_bytes()).decode()+'\"')\ndisplay(HTML('<iframe style=\"width:100%;height:1000px;border:0\" srcdoc=\"'+html.escape(panel,quote=True)+'\"></iframe>'))")]
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nbformat.write(nb,ROOT/'04_Proyecto_real_Colab.ipynb')
print('Tres documentos Word y notebook autocontenido generados.')
