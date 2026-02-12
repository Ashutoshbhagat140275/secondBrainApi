# Emotion Classifier — Evaluation Results

**Test samples:** 216  
**Overall accuracy:** 0.7269  

## Classification Report

```
              precision    recall  f1-score   support

     neutral      0.500     0.786     0.611        14
        calm      0.618     0.750     0.677        28
       happy      0.690     0.690     0.690        29
         sad      0.810     0.586     0.680        29
       angry      0.857     0.828     0.842        29
     fearful      0.792     0.655     0.717        29
   disgusted      0.800     0.828     0.814        29
   surprised      0.750     0.724     0.737        29

    accuracy                          0.727       216
   macro avg      0.727     0.731     0.721       216
weighted avg      0.743     0.727     0.729       216
```

## Confusion Matrix

```
             neutra    calm   happy     sad   angry  fearfu  disgus  surpri
   neutral       11       3       0       0       0       0       0       0
      calm        4      21       0       2       0       1       0       0
     happy        2       1      20       1       1       2       1       1
       sad        3       5       1      17       0       0       1       2
     angry        0       0       0       0      24       0       3       2
   fearful        1       2       3       1       1      19       0       2
 disgusted        1       1       0       0       2       1      24       0
 surprised        0       1       5       0       0       1       1      21
```
