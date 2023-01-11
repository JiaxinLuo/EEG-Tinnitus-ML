# preliminary verification

**assumption**: 
1. exposed to high frequency for a long time, tinn subjects would have a different perception for high frequency target, compart to cntl subjects.
2. if I am cntl, I am equally sensitive to high/low freq target. Tinn expresses unequal performance over the high/low freq target.
   1. perhaps, more sensitive to high. perhaps not sensitive to high.
3. compare 4 types of stimulus, 6 tasks all engaged.
4. finegrained, 6 tasks, 2 times as a data point. (3 types. attend high.low.passive). Do experiments independnetly (seperate out the ROI **target**.)

attent high: see 5000 long duration?

try both? 30 or 150. prediction, 150 more stable.

4 stinulus * 3 tasks * 2 times (coule be merged).

attend low: all 500hz are targets.


seed=0

| settings       | dataset split | test accuracy (10 epochs) |
| -------------- | ------------- | ------------------------- |
| (0,1)          | all_set       |                           |
| (1,0), lr=5e-4 | 5-7           | 94                        |
| (1,0), lr=5e-4 | 5, -5         | 48                        |
| (1,0), lr=5e-4 | 10, -5        | 59                        |
| (1,0), lr=1e-2 | 5-10          | 68                        |
| (1,0), lr=5e-4 | all_set       | 72                        |
| (1,0), lr=1e-4 | all_set       | 63                        |
| (1,0), lr=1e-4 | all_set       | 59                        |
| (1,0), lr=1e-2 | all_set       | 68                        |


# problems

1. ```Stand/BL_tinn_2-Stream_mcattn-2_sess1_set1_ICAREV.mat``` has (2,4) all epochs. others have 2,2
    ```
    (Pdb) sample_mat['allepochs'][0,0].shape
    (240, 358, 64)
    (Pdb) sample_mat['allepochs'][0,1].shape
    (540, 358, 64)
    (Pdb) sample_mat['allepochs'][1,0].shape
    (1, 0)
    (Pdb) sample_mat['allepochs'][1,1].shape
    (540, 358, 64)
    (Pdb) sample_mat['allepochs'][1,2].shape
    (540, 358, 64)

    ```

2. 
