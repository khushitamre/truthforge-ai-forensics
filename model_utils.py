from scipy.sparse import hstack
class FeatureUnionAdapter:
    def __init__(self,scaler,feature_names): self.scaler=scaler; self.feature_names=feature_names
    def transform(self,text_matrix, feats):
        arr=self.scaler.transform([[feats[k] for k in self.feature_names]])
        return hstack([text_matrix,arr]).tocsr()
