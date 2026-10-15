# nnU-Net unstructured pruning

Pruning was conducted on the validation split listed in [tbseg_fold_0_val.csv](../tb_files_used_in_this_work/tbseg_fold_0_val.csv) (see `tb_files_used_in_this_work`), using [nnunet_prune_threshold_search.py](nnunet_prune_threshold_search.py). The threshold for per-image absolute Dice reduction that counts as a failure is 0.05 wheras the maximum fraction of images allowed to fail is 0.10 during the pruning process.

Final results on the validation set:

```
Baseline | params=174,117,264 | dice=0.86±0.20
Pruned   | params=20,258,145  | dice=0.85±0.20 

threshold=8.17e-03 | sparsity=88.37%
```
