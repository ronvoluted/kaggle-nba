from sklearn.metrics import accuracy_score, recall_score , f1_score, precision_score
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def eval_baseline(xvar, yvar):
    mod_pred = np.full( (len(xvar),1), stats.mode(yvar)[0])
    print('Accuracy Score: ',accuracy_score(yvar,mod_pred),' F1 Score ',f1_score(yvar,mod_pred),' Precision Score ', precision_score(yvar,mod_pred),' Recall Score ', recall_score(yvar,mod_pred), '(BASELINE)')
    cm = confusion_matrix(yvar, mod_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot( cmap=plt.cm.Blues)
    return 