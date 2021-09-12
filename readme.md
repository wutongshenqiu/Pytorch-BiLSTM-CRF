## Pytorch-BiLSTM-CRF

PyTorch implementation of BiLSTM-CRF for named entity recognition.



## TODO List

- [x] 实现 `Vocab` 
- [x] 实现数据集处理
- [x] 实现 `BiLSTMCRF`
- [x] 实现 `trainer`
- [x] 效果测试



## Structures

```
├─data                          # dataset
├─notes                         # notations for BiLSTM-CRF
└─src                           # source code
```

## results

using `python -m tests.test_predict` to test results, the result on testing data of trained model is as follows

```
total tokens: 172601, correct tokens: 169196,accuracy: 0.98
LOC total tokens: 7271, correct tokens: 6223,accuracy: 0.86
PER total tokens: 5824, correct tokens: 5013,accuracy: 0.86
ORG total tokens: 7001, correct tokens: 6061,accuracy: 0.87
```