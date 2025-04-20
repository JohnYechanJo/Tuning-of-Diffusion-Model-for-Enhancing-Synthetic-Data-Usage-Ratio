Just a small demo for classifier structure, it will run very fast and won't cause any memory overflow.

It might be easy to expand its scale, just by changing some details, even expand to four categories.

The structure could be more complicated to adapt to a larger scale of dataset, lets try it.

I plan to join PGD adversarial to enhance the robustness of this part of the model later.(Already have a code, just adapt).

Annotations on the code exist on every piece of code

The defualt scale is :
1. 64:64 -> 128 as a batch
2. 10 batches totally (640:640->1280)
3. 8:1:1 train/val/test

The test result:(128)

accuracy 0.984375
macro avg {'precision': 0.9848484848484849, 'recall': 0.984375, 'f1-score': 0.9843711843711844, 'support': 128.0}
weighted avg {'precision': 0.9848484848484849, 'recall': 0.984375, 'f1-score': 0.9843711843711844, 'support': 128.0}

firstly run pre-process.py

secondly run classifier_demo.py

by allergic-garlic 
   4/20