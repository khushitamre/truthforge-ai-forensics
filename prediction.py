import pickle, re
from pathlib import Path
from .feature_engineering import extract_features, signal_scores
MODEL=Path(__file__).resolve().parents[1]/"models"/"final_model.pkl"
class Predictor:
    def __init__(self):
        with open(MODEL,"rb") as f: self.bundle=pickle.load(f)
    def predict_one(self,question,response):
        feats=extract_features(question,response)
        text=self.bundle['vectorizer'].transform([question+' '+response])
        X=self.bundle['feature_union'].transform(text, feats)
        risk=float(self.bundle['model'].predict_proba(X)[0,1]*100)
        signals=signal_scores(question,response)
        # Blend learned probability with transparent signals, preserving model-driven output.
        signal_risk=(100-signals['Semantic consistency'])*.25+(100-signals['Question–answer relevance'])*.2+signals['Linguistic uncertainty']*.2+(100-signals['Entity consistency'])*.15+(100-signals['Numerical consistency'])*.1+(100-signals['Response specificity'])*.1
        risk=round(max(0,min(100,0.72*risk+0.28*signal_risk)),1)
        return {'risk':risk,'reliability':round(100-risk,1),'verdict':verdict(risk),'signals':signals,'features':feats,'model_version':self.bundle.get('version','TF-1.0')}
def verdict(r):
    return 'VERY LOW RISK' if r<20 else 'LOW RISK' if r<40 else 'MODERATE RISK' if r<60 else 'HIGH RISK' if r<80 else 'CRITICAL RISK'
def top_factors(result):
    s=result['signals']; ranked=[('Low semantic consistency',100-s['Semantic consistency']),('Low question–answer relevance',100-s['Question–answer relevance']),('Entity pattern irregularity',100-s['Entity consistency']),('Linguistic uncertainty',s['Linguistic uncertainty']),('Numerical caution signal',100-s['Numerical consistency']),('Claim specificity',s['Response specificity'])]
    return [x[0] for x in sorted(ranked,key=lambda x:x[1],reverse=True)[:3]]
