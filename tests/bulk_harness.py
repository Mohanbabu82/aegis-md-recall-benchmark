import sys, io, json, zipfile, types
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import requests, mock_openfda
W = mock_openfda.world()
class Resp:
    def __init__(s, js=None, content=None): s._js=js; s.content=content; s.status_code=200
    def json(s): return s._js
    def raise_for_status(s): pass
def fake_get(url, params=None, timeout=None):
    if url.endswith('/download.json'):
        parts=[]
        for y in range(2014,2026):
            for q in range(1,5):
                parts.append({"display_name":f"{y} Q{q} (part 1 of 1)","file":f"bulk://{y}/{q}","size_mb":"1"})
        return Resp({"results":{"device":{"event":{"partitions":parts}}}})
    if url.startswith('bulk://'):
        y,q=map(int,url[7:].split('/')); res=[]
        for m in range(y*12+(q-1)*3, y*12+(q-1)*3+3):
            for c in W.codes: res += W.records(c,m,1000)
        b=io.BytesIO(); z=zipfile.ZipFile(b,'w'); z.writestr('x.json', json.dumps({"results":res})); z.close()
        return Resp(content=b.getvalue())
    raise RuntimeError(url)
requests.get = fake_get
src = open('nb1.py').read().replace('USE_BULK = (not API_KEY) and not MOCK','USE_BULK = True')
exec(compile(src,'nb1','exec'), {'__name__':'__main__'})
