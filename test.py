from sklearn.metrics import confusion_matrix
import cv2


label = cv2.imread('./dataset/SegmentationClass/IMG0001_2_2.png')
label = label.reshape(-1)
pred = cv2.imread('./pred_dir/IMG0001_2_2.png')
pred = pred.reshape(-1)

out = confusion_matrix(label,pred,labels=[0,1,2,3,4])
print(out)