# Target

Installation Requirements
1. python
2. pip
3. numpy 
4. scipy
5. matplotlib
6. tensorflow
7. keras
8. opencv

The bill file 878355.TIF need to be present in the same folder as the python file ocr_bill.py.
Command for running:
    
    python ocr_bill.py
    
This version only supports the above mentioned bill because identification of the handwritten area is yet another pattern recognition problem which I'll cover later. 
Initially the code trains a deep neural network based model using keras implementation. 
Then the handwritten area is segmented to the corresponding rows and these are segmented to obtain the digits.
These digits are recognized using the same keras model.

# In this implementation, I've focused only on completing the pipeline for segmentation and OCR.
# Further work:

1. Digits segmentation has to be improved 
    a. to obtain the correct order of digits
    b. to merge disjoint digits to the correct label
2. Accuracy has to be improved with simple debugging.

