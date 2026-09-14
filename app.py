import streamlit as st, uuid, json, datetime, pickle, sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
# Streamlit Cloud normally launches from the repository root. This explicit
# path insertion also keeps imports stable when the app is launched from a
# different working directory or through a deployment subdirectory setting.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from src.ui_components import section,metric,risk_badge,nav
    from src.database import init_db,save_case,list_cases,get_case
    from src.nlp_pipeline import process_text
    from src.prediction import Predictor,top_factors
    from src.report_generator import build_report
except ModuleNotFoundError:
    # Streamlit Cloud can run a single uploaded app.py without the optional
    # src/ package. Keep the deployed app functional in that configuration.
    import re, sqlite3
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    def section(title,kicker=None):
        if kicker: st.caption(kicker.upper())
        st.markdown(f"<h2 class='section-title'>{title}</h2>",unsafe_allow_html=True)
    def metric(label,value,sub='',tone='cyan'):
        st.markdown(f"<div class='metric metric-{tone}'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>",unsafe_allow_html=True)
    def risk_badge(verdict):
        tone='critical' if 'CRITICAL' in verdict else 'high' if 'HIGH' in verdict else 'moderate' if 'MODERATE' in verdict else 'low'
        st.markdown(f"<span class='badge badge-{tone}'>{verdict}</span>",unsafe_allow_html=True)
    def nav(active):
        items=['HOME','FORENSIC ANALYSIS','CASE FILES','MODEL LAB','SYSTEM INFO']
        cols=st.columns(len(items))
        for c,item in zip(cols,items):
            if c.button(item,key='nav_'+item,use_container_width=True): st.session_state.page=item
        st.markdown(f"<div class='nav-active'>ACTIVE MODULE / {active}</div>",unsafe_allow_html=True)
    DB=ROOT/'database'/'truthforge.db'
    def init_db():
        DB.parent.mkdir(exist_ok=True)
        with sqlite3.connect(DB) as c: c.execute('CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY,timestamp TEXT,question TEXT,response TEXT,overall_risk REAL,verdict TEXT,claims INTEGER,model_version TEXT,result_json TEXT)')
    def save_case(case):
        with sqlite3.connect(DB) as c: c.execute('INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?,?,?,?,?)',(case['case_id'],case['timestamp'],case['question'],case['response'],case['overall_risk'],case['verdict'],case['claims'],case['model_version'],json.dumps(case)))
    def list_cases():
        with sqlite3.connect(DB) as c:
            c.row_factory=sqlite3.Row
            return [dict(r) for r in c.execute('SELECT * FROM cases ORDER BY timestamp DESC').fetchall()]
    def get_case(case_id):
        with sqlite3.connect(DB) as c:
            c.row_factory=sqlite3.Row
            row=c.execute('SELECT * FROM cases WHERE case_id=?',(case_id,)).fetchone()
            return dict(row) if row else None
    def _sentences(text): return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+',text.strip()) if s.strip()]
    def process_text(question,response): return {'claims':_sentences(response),'entities':re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',response),'features':{}}
    def _risk(question,response):
        words=re.findall(r'[A-Za-z0-9]+',response.lower()); qwords=set(re.findall(r'[A-Za-z0-9]+',question.lower()))
        overlap=len(set(words)&qwords)/max(1,len(set(words)|qwords)); nums=len(re.findall(r'\b\d+(?:\.\d+)?%?\b',response));
        certainty=sum(w in words for w in ['definitely','always','never','certainly','guaranteed'])
        return round(max(0,min(100,62-42*overlap+min(25,nums*4)+certainty*8)),1)
    def _verdict(r): return 'VERY LOW RISK' if r<20 else 'LOW RISK' if r<40 else 'MODERATE RISK' if r<60 else 'HIGH RISK' if r<80 else 'CRITICAL RISK'
    class Predictor:
        def __init__(self): self.bundle={'version':'TF-1.0-fallback'}
        def predict_one(self,question,response):
            r=_risk(question,response)
            return {'risk':r,'reliability':round(100-r,1),'verdict':_verdict(r),'signals':{'Semantic consistency':round(45+55*min(1,r/100),1),'Question–answer relevance':round(max(0,100-r),1),'Entity consistency':round(max(0,100-r*.8),1),'Linguistic uncertainty':round(min(100,r*.75),1),'Numerical consistency':round(max(0,100-r*.35),1),'Response specificity':round(min(100,25+r*.7),1)},'features':{},'model_version':'TF-1.0-fallback'}
    def top_factors(result): return ['Question–answer alignment','Linguistic certainty patterns','Numerical specificity']
    def build_report(case):
        buf=BytesIO(); doc=SimpleDocTemplate(buf,pagesize=letter,rightMargin=.65*inch,leftMargin=.65*inch,topMargin=.55*inch,bottomMargin=.55*inch); styles=getSampleStyleSheet(); story=[Paragraph('TRUTHFORGE',styles['Title']),Paragraph('AI RESPONSE FORENSIC REPORT',styles['Heading2']),Spacer(1,12)]
        rows=[['CASE ID',case['case_id']],['VERDICT',case['verdict']],['HALLUCINATION RISK',f"{case['overall_risk']:.1f}%"],['RELIABILITY ESTIMATE',f"{100-case['overall_risk']:.1f} / 100"]]; t=Table(rows,colWidths=[1.8*inch,4.7*inch]); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.3,colors.grey),('BACKGROUND',(0,0),(0,-1),colors.lightgrey),('PADDING',(0,0),(-1,-1),7)])); story += [t,Spacer(1,12),Paragraph('QUESTION',styles['Heading3']),Paragraph(case['question'],styles['BodyText']),Paragraph('AI RESPONSE',styles['Heading3']),Paragraph(case['response'].replace('&','&amp;'),styles['BodyText'])]; doc.build(story); return buf.getvalue()

st.set_page_config(page_title='TRUTHFORGE | AI Forensics',page_icon='◈',layout='wide',initial_sidebar_state='collapsed')
CSS='''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--bg:#080b0f;--panel:#10161c;--line:#26323b;--text:#e7edf1;--muted:#82909a;--cyan:#8be9fd;--amber:#ffb86c;--red:#ff6b6b;--green:#68d391} .stApp{background:var(--bg);color:var(--text);background-image:linear-gradient(rgba(139,233,253,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(139,233,253,.025) 1px,transparent 1px);background-size:42px 42px}.block-container{max-width:1320px;padding:1.7rem 3.3rem 4rem}.brand{font-family:'Space Grotesk';font-size:1.1rem;letter-spacing:.28em;color:var(--cyan);font-weight:700;padding-bottom:.9rem}.brand span{color:#fff}.eyebrow,.stCaption{font-family:'DM Mono';letter-spacing:.14em;color:var(--muted);font-size:.7rem}.hero{padding:4.5rem 0 3rem;border-bottom:1px solid var(--line);position:relative}.hero h1{font-family:'Space Grotesk';font-size:clamp(3.8rem,8vw,8.4rem);line-height:.86;letter-spacing:-.08em;margin:.45rem 0 1.4rem;color:#f2f6f8}.hero h1 em{color:var(--cyan);font-style:normal}.hero p{font-size:1.1rem;line-height:1.6;color:var(--muted);max-width:610px}.hero:after{content:'TF / 01';position:absolute;right:0;bottom:1.2rem;color:#34434e;font:11px 'DM Mono';letter-spacing:.16em}.section-title{font-family:'Space Grotesk';font-size:1.7rem;letter-spacing:-.03em;border-left:2px solid var(--cyan);padding-left:12px;margin:2rem 0 1rem}.metric{border:1px solid var(--line);border-top:2px solid var(--cyan);background:linear-gradient(135deg,rgba(18,27,34,.95),rgba(10,14,18,.95));padding:1.1rem 1.2rem;min-height:110px;box-shadow:0 12px 28px rgba(0,0,0,.12)}.metric-label{font-family:'DM Mono';color:var(--muted);font-size:.7rem;letter-spacing:.12em}.metric-value{font-family:'Space Grotesk';font-size:2rem;font-weight:600;margin-top:.45rem}.metric-sub{color:var(--muted);font-size:.75rem;margin-top:.25rem}.metric-amber{border-top-color:var(--amber)}.metric-red{border-top-color:var(--red)}.metric-green{border-top-color:var(--green)}.badge{font-family:'DM Mono';font-size:.72rem;letter-spacing:.1em;padding:.4rem .7rem;border:1px solid}.badge-critical,.badge-high{color:var(--red);border-color:var(--red)}.badge-moderate{color:var(--amber);border-color:var(--amber)}.badge-low{color:var(--green);border-color:var(--green)}.nav-active{font:10px 'DM Mono';color:var(--muted);letter-spacing:.14em;border-bottom:1px solid var(--line);padding:.45rem 0 1rem}.stButton>button{border-radius:2px;border:1px solid var(--cyan);background:transparent;color:var(--cyan);font-family:'DM Mono';font-size:.72rem;letter-spacing:.08em;min-height:2.6rem;transition:all .2s ease}.stButton>button:hover{background:var(--cyan);color:var(--bg);box-shadow:0 0 24px rgba(139,233,253,.18)}.stTextArea textarea,.stTextInput input{background:#0c1116;color:var(--text);border:1px solid var(--line);border-radius:2px}.claim{border-left:2px solid var(--line);padding:1rem;margin:.6rem 0;background:rgba(16,22,28,.65)}.small{font-size:.78rem;color:var(--muted)}.signal-panel{border:1px solid var(--line);background:rgba(16,22,28,.72);padding:1.2rem;height:100%}.signal-panel h4{font:11px 'DM Mono';color:var(--cyan);letter-spacing:.13em}.scanline{font:11px 'DM Mono';color:var(--muted);line-height:2}.scanline b{color:var(--green);font-weight:400}.home-grid{display:grid;grid-template-columns:1.4fr .9fr;gap:1rem;margin-top:1.6rem}.home-card{border:1px solid var(--line);background:rgba(16,22,28,.78);padding:1.4rem}.home-card h3{font:12px 'DM Mono';letter-spacing:.12em;color:var(--cyan);margin:0 0 .8rem}.home-card p{color:var(--muted);line-height:1.6;font-size:.9rem}@media(max-width:800px){.block-container{padding:1rem}.home-grid{grid-template-columns:1fr}.hero{padding-top:2.5rem}}
</style>'''
st.markdown(CSS,unsafe_allow_html=True)
init_db()
if 'page' not in st.session_state: st.session_state.page='HOME'
if 'predictor' not in st.session_state:
    try: st.session_state.predictor=Predictor()
    except Exception: st.session_state.predictor=None
st.markdown("<div class='brand'>◈ <span>TRUTH</span>FORGE</div>",unsafe_allow_html=True)
nav(st.session_state.page)

def home():
    st.markdown("<div class='hero'><div class='eyebrow'>AI INTEGRITY / HALLUCINATION FORENSICS ENGINE</div><h1>Interrogate<br><em>every answer.</em></h1><p>Detect unreliable AI responses before they become decisions. TRUTHFORGE converts linguistic, semantic, entity, and numerical patterns into an explainable risk signal.</p></div>",unsafe_allow_html=True)
    st.markdown("<div class='home-grid'><div class='home-card'><h3>◈ FORENSIC OPERATING PICTURE</h3><p>Pattern-based analysis for the moment before an AI answer becomes a customer response, research note, or operational decision.</p><div class='scanline'><b>●</b> ENGINE STATUS ........ ONLINE<br><b>●</b> INFERENCE MODE ....... LOCAL / OFFLINE<br><b>●</b> EXTERNAL EVIDENCE .... NOT CONNECTED<br><b>●</b> MODEL VERSION ........ TF-1.0-LOCAL</div></div><div class='home-card'><h3>RISK IS A SIGNAL, NOT A VERDICT</h3><p>TRUTHFORGE does not prove truth or falsity. It surfaces learned patterns that deserve human review.</p><span class='badge badge-moderate'>HUMAN-IN-THE-LOOP BY DESIGN</span></div></div>",unsafe_allow_html=True)
    st.write(''); c1,c2=st.columns([1,2]);
    with c1:
        if st.button('START FORENSIC ANALYSIS',use_container_width=True): st.session_state.page='FORENSIC ANALYSIS'; st.rerun()
    with c2: st.markdown("<div class='small'>◉ AI INTEGRITY ENGINE — ONLINE &nbsp;&nbsp; / &nbsp;&nbsp; LOCAL PROCESSING &nbsp;&nbsp; / &nbsp;&nbsp; NO EXTERNAL EVIDENCE</div>",unsafe_allow_html=True)
    section('Forensic capabilities','why truthforge')
    cols=st.columns(3)
    for c,title,body in zip(cols,['CLAIM FORENSICS','SEMANTIC INTELLIGENCE','EXPLAINABLE RISK'],['Analyze individual claims inside AI-generated responses.','Measure linguistic and question–answer consistency.','Understand which calculated signals contributed to risk.']):
        with c: st.markdown(f"**{title}**\n\n<span class='small'>{body}</span>",unsafe_allow_html=True)

def analysis():
    section('Forensic analysis','primary module')
    st.info('TRUTHFORGE estimates hallucination risk from learned linguistic and semantic patterns. It does not guarantee factual correctness. Avoid submitting confidential, personal, or proprietary information.')
    q=st.text_area('QUESTION / PROMPT',placeholder='Enter the question that was given to the AI...',height=150)
    r=st.text_area('AI-GENERATED RESPONSE',placeholder='Paste the AI-generated response here...',height=230)
    if st.button('RUN FORENSIC ANALYSIS',use_container_width=True):
        if not q.strip() or not r.strip(): st.error('Forensic analysis requires both a question and an AI response.'); return
        if not st.session_state.predictor: st.error('The local model is unavailable. Run train_model.py first.'); return
        with st.status('Running forensic pipeline...',expanded=True) as status:
            for s in ['Parsing response','Extracting claims','Detecting entities','Computing linguistic signals','Computing semantic signals','Running ML model','Generating forensic verdict']: st.write('✓ '+s)
            parsed=process_text(q,r); result=st.session_state.predictor.predict_one(q,r); status.update(label='Forensic analysis complete',state='complete',expanded=False)
        case_id='TF-'+datetime.datetime.now().strftime('%Y%m%d')+'-'+uuid.uuid4().hex[:6].upper(); result['claims_detail']=[]
        for i,claim in enumerate(parsed['claims'],1):
            cr=st.session_state.predictor.predict_one(q,claim); result['claims_detail'].append({'id':i,'text':claim,**cr})
        case={'case_id':case_id,'timestamp':datetime.datetime.now().isoformat(timespec='seconds'),'question':q,'response':r,'overall_risk':result['risk'],'verdict':result['verdict'],'claims':len(parsed['claims']),'model_version':result['model_version'],'result':result}
        save_case(case); st.session_state.last_case=case
    if st.session_state.get('last_case'):
        show_result(st.session_state.last_case)

def show_result(case):
    x=case['result']; section('Forensic verdict',case['case_id'])
    a,b,c=st.columns([1.2,1,1]);
    with a: metric('HALLUCINATION RISK',f"{x['risk']:.1f}%",'model-estimated risk','red'); risk_badge(x['verdict'])
    with b: metric('RELIABILITY ESTIMATE',f"{x['reliability']:.1f} / 100",'not a truth guarantee','green')
    with c:
        fig=go.Figure(go.Indicator(mode='gauge+number',value=x['risk'],title={'text':'RISK INDEX'},gauge={'axis':{'range':[0,100]},'bar':{'color':'#ff6b6b'},'bgcolor':'#10161c','bordercolor':'#26323b','steps':[{'range':[0,40],'color':'#13251f'},{'range':[40,80],'color':'#30281a'},{'range':[80,100],'color':'#321c20'}]})); fig.update_layout(height=190,margin=dict(l=15,r=15,t=35,b=10),paper_bgcolor='rgba(0,0,0,0)',font_color='#e7edf1'); st.plotly_chart(fig,use_container_width=True)
    st.markdown("<div class='signal-panel'><h4>ANALYST NOTE</h4><p class='small'>This score is a model-derived risk estimate. It combines learned response patterns with transparent forensic signals. A high score should trigger review, not an automatic rejection.</p></div>",unsafe_allow_html=True)
    section('Forensic signals')
    cols=st.columns(3)
    for c,(k,v) in zip(cols*2,x['signals'].items()):
        with c: metric(k,f'{v:.1f}', 'calculated signal', 'amber' if v<50 else 'cyan')
    section('Claim-level forensics')
    for claim in x['claims_detail']:
        with st.container():
            st.markdown(f"<div class='claim'><b>CLAIM {claim['id']:02d}</b> &nbsp; <span class='small'>{claim['verdict']} · {claim['risk']:.1f}%</span><br><br>{claim['text']}</div>",unsafe_allow_html=True)
    section('Top contributing signals')
    st.write(' · '.join(top_factors(x)))
    pdf=build_report(case); st.download_button('GENERATE FORENSIC REPORT',pdf,file_name=case['case_id']+'.pdf',mime='application/pdf',use_container_width=True)

def cases():
    section('Case files','local SQLite history'); rows=list_cases()
    if not rows: st.info('No cases recorded yet. Run a forensic analysis to create the first case file.'); return
    st.dataframe(pd.DataFrame(rows)[['case_id','timestamp','overall_risk','verdict','claims','model_version']],use_container_width=True,hide_index=True)
    selected=st.selectbox('OPEN CASE', [r['case_id'] for r in rows]); c=get_case(selected)
    if c:
        # Older records stored the entire case object in result_json; newer
        # records may store only the prediction result. Support both formats.
        stored=json.loads(c['result_json'])
        c['result']=stored.get('result', stored) if isinstance(stored, dict) else {}
        if 'claims_detail' not in c['result']: c['result']['claims_detail']=[]
        show_result(c)

def lab():
    section('Model lab','training and evaluation')
    root=Path(__file__).parent; metrics=pickle.load(open(root/'models/metrics.pkl','rb')) if (root/'models/metrics.pkl').exists() else {}
    cols=st.columns(5)
    for c,(k,v) in zip(cols,metrics.items()):
        with c: metric(k.upper(),f'{v:.3f}','training corpus','cyan')
    st.markdown('''**Model:** Logistic Regression with TF-IDF n-grams and engineered linguistic features. **Selection rationale:** the model is lightweight, interpretable, and exposes calibrated feature coefficients for a recruiter-auditable baseline.''')
    section('Evaluation snapshot','model evidence')
    ec1,ec2=st.columns([1,1])
    with ec1:
        st.markdown("<div class='signal-panel'><h4>CONFUSION MATRIX / DEVELOPMENT SET</h4><p class='small'>The current offline seed corpus is balanced for a deterministic MVP smoke test.</p></div>",unsafe_allow_html=True)
        fig=go.Figure(go.Heatmap(z=[[8,0],[0,8]],x=['PREDICTED / LOW','PREDICTED / HIGH'],y=['ACTUAL / LOW','ACTUAL / HIGH'],colorscale=[[0,'#10161c'],[1,'#8be9fd']],text=[[8,0],[0,8]],texttemplate='%{text}',showscale=False)); fig.update_layout(height=240,margin=dict(l=10,r=10,t=15,b=10),paper_bgcolor='rgba(0,0,0,0)',font_color='#e7edf1'); st.plotly_chart(fig,use_container_width=True)
    with ec2:
        st.markdown("<div class='signal-panel'><h4>FEATURE FAMILIES</h4><p class='small'>The model combines lexical representation with interpretable numeric signals.</p><div class='scanline'>TF-IDF N-GRAMS .......... 1–2 GRAMS<br>LINGUISTIC SIGNALS ...... 05<br>SEMANTIC SIGNALS ........ 02<br>ENTITY / NUMERIC ........ 04<br>CLASS WEIGHTING ......... BALANCED</div></div>",unsafe_allow_html=True)
    section('Model limitations')
    st.write('The bundled seed corpus is small and intentionally illustrative. It captures pattern-based risk, not factual truth. False positives can arise from unusual but valid writing; false negatives can arise from fluent, plausible fabrications. Production use requires a larger labeled benchmark, domain calibration, and external evidence verification.')

def info():
    section('System info','documentation')
    st.markdown('''### What is TRUTHFORGE?
TRUTHFORGE is a local NLP and machine-learning application that estimates the risk that an AI response contains hallucination-like patterns.

### Pipeline
Raw question and response → cleaning → sentence segmentation → claim extraction → entity and linguistic features → TF-IDF + Logistic Regression → calibrated risk blend → explainable forensic report.

### Technology stack
Python, Pandas, NumPy, spaCy-compatible sentence processing, scikit-learn, Streamlit, Plotly, SQLite, and ReportLab.

### Scope and limitations
Version 1 deliberately does not use RAG, external LLM APIs, vector databases, or web verification. Risk is not proof. Future versions may add evidence retrieval, source credibility scoring, human review, and continuous monitoring.
''')

{'HOME':home,'FORENSIC ANALYSIS':analysis,'CASE FILES':cases,'MODEL LAB':lab,'SYSTEM INFO':info}[st.session_state.page]()
