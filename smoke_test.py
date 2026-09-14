from pathlib import Path
from src.prediction import Predictor
from src.database import init_db, save_case, list_cases
from src.report_generator import build_report
p=Predictor(); result=p.predict_one('What is the capital of France?','Paris is the capital of France.')
assert 0 <= result['risk'] <= 100 and result['verdict']
case={'case_id':'SMOKE-001','timestamp':'2026-01-01T00:00:00','question':'Q','response':'A','overall_risk':result['risk'],'verdict':result['verdict'],'claims':1,'model_version':result['model_version'],'result':result}
init_db(); save_case(case); assert any(x['case_id']=='SMOKE-001' for x in list_cases())
pdf=build_report(case); assert pdf[:4] == b'%PDF'
print('smoke ok',result['risk'],len(pdf))
