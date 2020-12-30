Folder containing all Notebooks and scripts used for the challenge, [RANZCR CLiP - Catheter and Line Position Challenge](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification)

### Best Model Parameters:
--------------------------

1. No Data Augmentations
2. cfg = CFG(filter_sizes=[64, 128, 256, 512], HL=[2048], epochs=30, n_folds=5) 
3. model = CNN(filter_sizes=cfg.filter_sizes, HL=cfg.HL, OL=cfg.OL, use_DP=True, DP=0.5).to(cfg.device)
4. optimizer = model.getOptimizer(lr=1e-3, wd=1e-5)
