
1. FP16 training: Training with FP16 and SyncBN on multi-GPU seems to cause NAN loss for current projects (SyncBN option for FP16 is not finished). Use normal BN instead, currently.

2. Code might give some warnings, which would not cause any trouble for normal training and testing.

   ```
   Failing to import plugin, ModuleNotFoundError("No module named 'plugin'")
   loading third party model failed cannot import name 'model_zoo' from 'third_party' (unknown location)
   ```
