# Experiments
train val scores
convlstm: 0.33, 0.32 
graph_model: 0.22, 0.22


**Convlstm (50x33)**

Epoch: 100, Train_loss: 0.43118, Validation_loss: 0.43033, Took 11.151 seconds.
Train metrics: AP:0.59236, F1:0.62162, Confusion_matrix: [TN:1398, FN:90, FP:47, TP:114]
Validation metrics: AP:0.61985, F1:0.63813, Confusion_matrix: [TN:1408, FN:79, FP:50, TP:114]
train_val:93/94Early exiting from epoch: 100, Eval Loss: 0.43098.
Evaluation metrics: AP:0.59740, F1:0.62434, Confusion_matrix: [TN:1401, FN:87, FP:49, TP:113]
test:15/16Test finished, test loss: 0.42868
Test metrics: AP:0.60869, F1:0.63443, Confusion_matrix: [TN:1408, FN:80, FP:49, TP:113]


**graph-model (100x66)**

Epoch: 100, Train_loss: 0.22112, Validation_loss: 0.22268, Took 761.412 seconds.
Train metrics: AP:0.27534, F1:0.37667, Confusion_matrix: [TN:25123, FN:5427, FP:596, TP:1826]
Validation metrics: AP:0.27431, F1:0.37719, Confusion_matrix: [TN:25067, FN:5404, FP:593, TP:1822]
train_val:297/298Early exiting from epoch: 100, Eval Loss: 0.22226.
Evaluation metrics: AP:0.27453, F1:0.37666, Confusion_matrix: [TN:25085, FN:5472, FP:585, TP:1836]
test:50/51Test finished, test loss: 0.21637
Test metrics: AP:0.27286, F1:0.37418, Confusion_matrix: [TN:25381, FN:5243, FP:620, TP:1755]