# preliminary verification

Target Mark:

1 = 500 Hz; short duration standard tone

2 = 5000 Hz, short duration standard tone

3 = 500 Hz, long duration deviant tone

4 = 5000 Hz, long duration deviant tone

![image](https://github.com/NIRVANALAN/EEG-Tinnitus-ML/assets/63215082/6832562d-8ae3-4cce-b67b-6aef286e21d1)

seed=0

dataset information for a 0 - 3
```
pos in training 1344                                                                                                
pos in test 330                                           
train dataset size: 2552, test set size: 638                                                                        
```

Train split ratio = 0.8

| Instruction          | EEG                 |                           |
| -------------------- | --------------------- | ------------------------- |
| Att-HF(0) /Att-LF(1) | Dev L (3)/Dev H (4) | test accuracy (20 epochs) |
| Att-HF                    | 3                     | **599/638 (93.9%)**         |
| Att-HF                    | 4                     | 585/638 (91.7%)             |
| Att-LF                    | 3                     | **582/621 (93.7%)**         |
| Att-LF                    | 4                     | 578/621 (93.1%)             |

For Dev tone ERP, when EEG auditory stimulus is low frequency, more discrimination

Few shot validation

| Train Ratio | Att-HF(0) | Dev L (3) | test accuracy (20 epochs) |
| ----------- | --------- | ---------- | ------------------------- |
| 0.1         | 0         | 3          | 1849/2871 (64%)           |
| 0.2         | 0         | 3          | 1757/2552 (69%)           |
| 0.3         | 0         | 3          | 1656/2233 (74%)           |
| 0.5         | 0         | 3          | 1283/1595 (80%)           |
| 0.7         | 0         | 3          | 852/957 (89%)             |
| 0.9         | 0         | 3          | 309/319 (97%)             |


Async train setting, train ratio  = 0.8

| train setting | test setting | test accuracy (20 epochs) |
| ------------- | ------------ | ------------------------- |
| 0-3           | 1-4          | 403/621 (65%)             |
| 0-3           | 1-3          | 403/621 (65%)             |


Check short tone (standard), train split ratio = 0.8

```
training: 270*(17+13)*2*0.8 = 12960
test: 270*(17+13)*2*0.2 = 3240
```

| Att-HF(0) /Att-LF(1) | Std L (1)/Std H (2) | test accuracy (20 epochs) |
| -------------------- | --------------------- | ------------------------- |
| 0                    | 1                     | 2885/3195 (90.3%)           |
| 0                    | 2                     | 2999/3195 (93.9%)           |
| 1                    | 1                     | **2953/3105 (95.1%)**       |
| 1                    | 2                     | **2937/3105 (94.5%)**       |

For Std tone ERP, when INSTRUCTION is to Att-LF, more discrimination

# problems

- [ ] Do ratio study for all combinations, draw a curve.

Test set, balanced split, cnl tinn 13. randomize (cntl random 13, more random seed.)

Random split 5+5 for test, 8+8, random split 1~8 for training. Draw a curve.

5+5, random (train several random seed.)

Also do this on short tones(standard, more trial data).
