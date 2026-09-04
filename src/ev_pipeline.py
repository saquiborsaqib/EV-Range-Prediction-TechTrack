import re, numpy as np, pandas as pd
from catboost import CatBoostRegressor

class EVRangePipeline:
    def __init__(self, params=None):
        self.params=params or dict(iterations=600,depth=6,learning_rate=.04,loss_function="RMSE",l2_leaf_reg=5,random_seed=42,verbose=False)
        self.model=CatBoostRegressor(**self.params); self.columns_=None; self.cats_=None
    def _cargo(self,v):
        if pd.isna(v): return np.nan
        m=re.search(r"\d+(?:\.\d+)?",str(v)); return float(m.group()) if m else np.nan
    def transform(self,X):
        d=X.copy()
        if "cargo_volume_l" in d: d["cargo_volume_l"]=d["cargo_volume_l"].apply(self._cargo)
        d=d.drop(columns=["range_km","efficiency_wh_per_km","source_url","model","battery_type"],errors="ignore")
        d["footprint_m2"]=d["length_mm"]*d["width_mm"]/1_000_000
        d["volume_proxy_m3"]=d["length_mm"]*d["width_mm"]*d["height_mm"]/1_000_000_000
        d["battery_per_torque"]=d["battery_capacity_kWh"]/(d["torque_nm"].abs()+1)
        d["battery_per_footprint"]=d["battery_capacity_kWh"]/(d["footprint_m2"]+1e-6)
        d["performance_index"]=d["top_speed_kmh"]/(d["acceleration_0_100_s"]+.1)
        if self.columns_ is not None:
            for c in self.columns_:
                if c not in d: d[c]=np.nan
            d=d[self.columns_]
        for c in d.select_dtypes("object").columns: d[c]=d[c].fillna("Missing").astype(str)
        return d
    def fit(self,X,y):
        z=self.transform(X); self.columns_=z.columns.tolist(); z=self.transform(X)
        self.cats_=z.select_dtypes("object").columns.tolist()
        self.model.fit(z,y,cat_features=[z.columns.get_loc(c) for c in self.cats_]); return self
    def predict(self,X): return self.model.predict(self.transform(X))
