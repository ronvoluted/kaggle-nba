from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve, f1_score, plot_confusion_matrix
from sklearn.base import is_regressor
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

def eval_model(classifier, X_train, y_train, X_val, y_val, show=True):
  """Evaluate Model:
    This module train the model based on passed in classifier and training set, 
    and then use valuation set to evaluate the performance of trained model.
    It can select to print out or not all related metrics for investigation.

    Parameters
    ----------
    classifier : sklearn classifier
        classifier to be evaluated
    X_train : dataframe or array
        Features of training set
    y_train : dataframe or array
        Target value of training set
    X_val : dataframe or array
        Features of valuation set
    y_val : dataframe or array
        Target value of valuation set
    show : boolean
        If metric should be shown or not

    Returns
    -------
    model
        Trained model based on classifier and training set data
    roc_score_training
        ROC AUC score of training set
    roc_score_val
        ROC AUC score of valuation set
    """

  model = classifier.fit(X_train,y_train)
  roc_score_val, acc_val, f1_val, precision_val, recall_val, r2_val = get_performance(model, X_val, y_val, "Validate", False)  
  roc_score_training, acc_train, f1_train, precision_train, recall_train, r2_train  = get_performance(model, X_train, y_train, "Train", False)
  
  fig, axes = plt.subplots(nrows=1,ncols=2)

  if show:
    html = f'<div style="padding: 1rem; background: #2E3440; font-family: Ubuntu, \'Helvetica Neue\', \'Segoe UI\';"><div style="max-width: 100%; height: 2rem; display: flex;"><span style="min-width: 5rem; margin-top: 0.5rem; margin-right: 1rem;"></span><div style="width: 100%; height: 2rem; display: flex; justify-content: space-evenly; padding: 0.25rem; text-align: center; font-weight: 700"><span style="width: 6rem; margin-top: 0.25em; color: #EBCB8B">AUROC</span><span style="width: 6rem; margin-top: 0.25em; color: #A3BE8C;">Accuracy</span><span style="width: 6rem; margin-top: 0.25em; color: #BF616A">F1</span><span style="width: 6rem; margin-top: 0.25em; color: #B48EAD">Recall</span><span style="width: 6rem; margin-top: 0.25em; color: #B48EAD">Precision</span><span style="width: 6rem; margin-top: 0.25em; color: #D08770">R2</span></div></div><div style="max-width: 100%; height: 2rem; display: flex; margin-top: 0.5rem;"><div style="min-width: 5rem; margin-top: 0.5rem; margin-right: 1rem; font-weight: 700; color: ghostwhite;">Validation</div><div style="width: 100%; height: 2rem; display: flex; justify-content: space-evenly; padding: 0.25rem; border-radius: 4px; box-shadow: 0 4px 6px 2px black; text-align: center; background: #3B4252"><span style="width: 6rem; margin-top: 0.25em; color: #EBCB8B">{round(roc_score_val, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #A3BE8C">{round(acc_val, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #BF616A">{round(f1_val, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #B48EAD">{round(recall_val, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #B48EAD">{round(precision_val, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #D08770">{round(r2_val, 8)}</span></div></div><div style="max-width: 100%; height: 2rem; display: flex; margin-top: 1.5rem; margin-bottom: 1rem;"><div style="min-width: 5rem; margin-top: 0.5rem; margin-right: 1rem; font-weight: 700; color: ghostwhite;">Training</div><div style="width: 100%; height: 2rem; display: flex; justify-content: space-evenly; padding: 0.25rem; border-radius: 4px; box-shadow: 0 4px 6px 2px black; text-align: center; background: #3B4252"><span style="width: 6rem; margin-top: 0.25em; color: #EBCB8B">{round(roc_score_training, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #A3BE8C">{round(acc_train, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #BF616A">{round(f1_train, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #B48EAD">{round(recall_train, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #B48EAD">{round(precision_train, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #D08770">{round(r2_train, 8)}</span></div></div></div>'
    display(HTML(html))
    disp_val = plot_confusion_matrix(model, X_val, y_val, cmap=plt.cm.Blues,ax = axes[0],colorbar=False)
    disp_val.ax_.set_title('Confusion matrix Validate')
    disp_train = plot_confusion_matrix(model, X_train, y_train, cmap=plt.cm.Blues,ax = axes[1],colorbar=False)
    disp_train.ax_.set_title('Confusion matrix Train')
    plt.tight_layout() 
    plt.show()
    print(model)

  return model, roc_score_training, roc_score_val

def get_performance(mod, xvar, yvar, runtype, show):
  """Evaluate the performance of the model:
    This function uses valuation set to evaluate the performance of trained model.
    It can select to print out or not all related metrics for investigation.

    Parameters
    ----------
    mod : sklearn classifier
        Trained model
    xvar : dataframe or array
        Features
    yvar : dataframe or array
        Target value
    runtype : string
        Enum : [Train, Validate] 
        This value is used to label printed metrics for better understanding
    show : boolean
        If metric should be shown or not

    TODO: Add more relavent scores 
    TODO: Have a switch to select / remove metrics
    TODO: More Plots/Graphs for clear indications

    Returns
    -------
    mod_roc_score
        ROC AUC score of dataset
        
    """

  if is_regressor(mod):
    convert_ratio = np.vectorize(lambda x: 1 if x > 0.5 else 0)
    mod_pred_proba = mod.predict(xvar)
    mod_pred = convert_ratio(mod_pred_proba)
  else:
    mod_pred = mod.predict(xvar)
    mod_pred_proba = mod.predict_proba(xvar)[:, 1]

  mod_roc_score = roc_auc_score(yvar, mod_pred_proba)

  accuracy = accuracy_score(yvar,mod_pred)
  f1 = f1_score(yvar,mod_pred)
  precision = precision_score(yvar,mod_pred)
  recall = recall_score(yvar,mod_pred)
  r2 = mod.score(xvar, yvar)
  
  if show:
    html = f'<div style="padding: 1rem; background: #2E3440; font-family: Ubuntu, \'Helvetica Neue\', \'Segoe UI\';"><div style="max-width: 100%; height: 2rem; display: flex;"><span style="min-width: 5rem; margin-top: 0.5rem; margin-right: 1rem;"></span><div style="width: 100%; height: 2rem; display: flex; justify-content: space-evenly; padding: 0.25rem; text-align: center; font-weight: 700"><span style="width: 6rem; margin-top: 0.25em; color: #EBCB8B">AUROC</span><span style="width: 6rem; margin-top: 0.25em; color: #A3BE8C;">Accuracy</span><span style="width: 6rem; margin-top: 0.25em; color: #BF616A">F1</span><span style="width: 6rem; margin-top: 0.25em; color: #B48EAD">Recall</span><span style="width: 6rem; margin-top: 0.25em; color: #B48EAD">Precision</span><span style="width: 6rem; margin-top: 0.25em; color: #D08770">R2</span></div></div><div style="max-width: 100%; height: 2rem; display: flex; margin-top: 0.5rem;"><div style="min-width: 5rem; margin-top: 0.5rem; margin-right: 1rem; font-weight: 700; color: ghostwhite;">Validation</div><div style="width: 100%; height: 2rem; display: flex; justify-content: space-evenly; padding: 0.25rem; border-radius: 4px; box-shadow: 0 4px 6px 2px black; text-align: center; background: #3B4252"><span style="width: 6rem; margin-top: 0.25em; color: #EBCB8B">{round(mod_roc_score, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #A3BE8C">{round(accuracy, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #BF616A">{round(f1, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #B48EAD">{round(recall, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #B48EAD">{round(precision, 8)}</span><span style="width: 6rem; margin-top: 0.25em; color: #D08770">{round(r2, 8)}</span></div></div></div>'
    display(HTML(html))
    disp = plot_confusion_matrix(mod, xvar, yvar, cmap=plt.cm.Blues)
    disp.ax_.set_title('Confusion matrix '+runtype)
  
  return mod_roc_score, accuracy, f1, precision, recall, r2
