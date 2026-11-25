import json, re
core = open('src/aegis_core.py').read()
import sys
srcs = sys.argv[1].split(',') if len(sys.argv)>1 else ['nb1_data','nb2_enrich','nb3_features','nb4_experiments','nb5_explain']
OUTF = sys.argv[2] if len(sys.argv)>2 else 'notebooks/kaggle_runs/AEGIS_full_pipeline.ipynb'
TITLE = sys.argv[3] if len(sys.argv)>3 else 'AEGIS-MD — full pipeline in one notebook (GPU T4 + Internet ON)'
cells=[dict(cell_type='markdown',metadata={},source=["# "+TITLE+"\n","Sections 1–5 = the five notebooks. Run cells in order; every section saves its outputs to /kaggle/working and the next section reloads them.\n"]),
       dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=('%%writefile aegis_core.py\n'+core).splitlines(True))]
for i,src in enumerate(srcs):
    txt=open(f'src/{src}.py').read()
    ch=re.split(r'^# %%(.*)$',txt,flags=re.M)
    for tag,body in zip(ch[1::2],ch[2::2]):
        body=body.strip('\n')
        if tag.strip()=='writefile-core': continue
        if i==0: body=body.replace('TIME_BUDGET_MIN = 38 ','TIME_BUDGET_MIN = 600')
        if tag.strip()=='[markdown]':
            md='\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.split('\n'))
            cells.append(dict(cell_type='markdown',metadata={},source=md.splitlines(True)))
        else:
            compile(body,src,'exec')
            cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=body.splitlines(True)))
    cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["import gc; gc.collect(); print('section done')\n"]))
nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}})
json.dump(nb,open(OUTF,'w'),indent=1); print(len(cells),'cells')
