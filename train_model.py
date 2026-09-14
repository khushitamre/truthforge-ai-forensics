"""Train a lightweight local model. The seed corpus is documented in data/README.md."""
import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from src.feature_engineering import extract_features
from src.model_utils import FeatureUnionAdapter

ROOT=Path(__file__).parent
rows=[
('What is the capital of France?','Paris is the capital of France.',0),('Who wrote Hamlet?','William Shakespeare wrote Hamlet.',0),('What is water made of?','Water is composed of hydrogen and oxygen.',0),('When did Apollo 11 land?','Apollo 11 landed on the Moon in 1969.',0),('Explain photosynthesis.','Plants convert light energy into chemical energy using carbon dioxide and water.',0),('What is 2 plus 2?','Two plus two equals four.',0),('Who painted Starry Night?','Vincent van Gogh painted The Starry Night.',0),('What is the largest planet?','Jupiter is the largest planet in the Solar System.',0),
('Who founded Company X and when?','Company X was founded by Aristotle in 1842, and it currently operates on Mars.',1),('What is the capital of France?','The capital of France is Berlin, founded by 12 astronauts in 2037.',1),('Who wrote Hamlet?','Hamlet was written by Albert Einstein in 1905 and contains 42 chapters.',1),('When did Apollo 11 land?','Apollo 11 landed in 1492 and was commanded by Leonardo da Vinci.',1),('Explain photosynthesis.','Photosynthesis is a cryptocurrency protocol invented in 2001 by Newton.',1),('What is 2 plus 2?','Two plus two equals seventeen, definitely and always.',1),('Who painted Starry Night?','Starry Night was painted by Shakespeare in 12 percent of the year.',1),('What is the largest planet?','The largest planet is Mercury, which has 900 moons and no atmosphere.',1),
]
df=pd.DataFrame(rows,columns=['question','response','label'])
texts=df.question+' '+df.response
vec=TfidfVectorizer(ngram_range=(1,2),max_features=1200,stop_words='english')
Xtext=vec.fit_transform(texts)
feat_names=list(extract_features(df.question.iloc[0],df.response.iloc[0]).keys())
Xnum=pd.DataFrame([extract_features(q,r) for q,r in zip(df.question,df.response)])[feat_names]
scaler=StandardScaler(); Xnum=scaler.fit_transform(Xnum)
from scipy.sparse import hstack
X=hstack([Xtext,Xnum]).tocsr()
model=LogisticRegression(max_iter=1200,class_weight='balanced',random_state=42)
model.fit(X,df.label)
proba=model.predict_proba(X)[:,1]
metrics={'accuracy':accuracy_score(df.label,proba>.5),'precision':precision_score(df.label,proba>.5),'recall':recall_score(df.label,proba>.5),'f1':f1_score(df.label,proba>.5),'roc_auc':roc_auc_score(df.label,proba)}
bundle={'model':model,'vectorizer':vec,'feature_union':FeatureUnionAdapter(scaler,feat_names),'metrics':metrics,'feature_names':feat_names,'version':'TF-1.0-local'}
(ROOT/'models').mkdir(exist_ok=True)
with open(ROOT/'models/final_model.pkl','wb') as f: pickle.dump(bundle,f)
(ROOT/'models/metrics.pkl').write_bytes(pickle.dumps(metrics))
print('trained',metrics)
