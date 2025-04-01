обучил 3 варианта

* crop - оставшиеся параметры kH_eff, H_all, ntg, Hgvk, minLgwc, Vdren, Sdren, Scr, Temp q_i или OFR
Почему? KH1-3 сильно коррелируют друг с другом, KHmin1-3 тож снес потому что они клоны kH_eff,
minL1-3 снес ибо сетка равномерная, Lgs снес ибо длины гс скважины одинаковая примерно у всех

обучил 3 модельки на 100 триалов оптюны, forest_modif внутри на чем обучал
1. forest q_i все тоже самое просто вместо ofr q_i
Final Model Metrics:
MSE: 66435125765.8208
RMSE: 257750.1227
MAE: 213796.9274
R²: 0.2066
Spearman Correlation: 0.4438

2. forest_q_i_crop
Final Model Metrics:
MSE: 68127876298.4285 (36.40%)
RMSE: 261013.1727 (60.33%)
MAE: 215018.1826 (49.70%)
R²: 0.1864
Spearman Correlation: 0.4237

3. forest_OFR_crop
Final Model Metrics:
MSE: 0.0009 (31.58%)
RMSE: 0.0300 (56.19%)
MAE: 0.0254 (47.49%)
R²: 0.2427
Spearman Correlation: 0.4768


их вся сравнил с базовым forest

