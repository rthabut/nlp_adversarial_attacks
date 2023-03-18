from sklearn.covariance import OAS, LedoitWolf, MinCovDet

estimators_dict = {"LW": LedoitWolf, "OAS": OAS, "MCD": MinCovDet}

rd_state = 42
