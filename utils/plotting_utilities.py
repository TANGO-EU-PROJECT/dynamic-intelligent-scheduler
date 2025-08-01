import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_curve


def plot_roc(y_true, y_probas, title='ROC Curves',
                   plot_micro=True, plot_macro=False, class_labels=None, classes_to_plot=None,
                   ax=None, figsize=None, cmap='Paired',
                   title_fontsize="large", text_fontsize="large", save_file="ROC.pdf"):
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)
    
    classes = np.unique(y_true)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=20)

    fpr_dict = dict()
    tpr_dict = dict()

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true, probas[:, i],
                                                pos_label=classes[i])
        if to_plot:
            roc_auc = auc(fpr_dict[i], tpr_dict[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            if class_labels is None:
                ax.plot(fpr_dict[i], tpr_dict[i], lw=3, color=color,
                        label='ROC of class {0} (area = {1:0.2f})'
                              ''.format(classes[i], roc_auc))
            else:
                ax.plot(fpr_dict[i], tpr_dict[i], lw=3, color=color,
                        label='ROC of class {0} (area = {1:0.2f})'
                        ''.format(class_labels[i], roc_auc))

    if plot_micro:
        binarized_y_true = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            binarized_y_true = np.hstack(
                (1 - binarized_y_true, binarized_y_true))
        fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), probas.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                label='micro-average ROC'
                      '(area = {0:0.2f})'.format(roc_auc),
                 linestyle='dashdot', linewidth=3)

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[x] for x in range(len(classes))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        roc_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr,
                label='macro-average ROC'
                      '(area = {0:0.2f})'.format(roc_auc),
                linestyle=':', linewidth=3)

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=18)
    ax.set_ylabel('True Positive Rate', fontsize=18)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=18)
    plt.savefig(save_file, bbox_inches='tight')  
    return ax


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]


def precision_recall_threshold(p, r, y_test, y_scores, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    
    # Use predict_proba to obtain the probabilities of target class 
    p, r, thresholds = precision_recall_curve(y_test, y_scores)
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.5, 1.01]);
    plt.xlim([0.5, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)
    
    
def create_feature_heatmap(df, save_file="heatmap.pdf"):
    corrDF=  df[['Threads', 'Global Memory Loads', 'Local Memory Loads',
       'Private Memory Loads', 'Total Loops',
       'Parallel Loops', 'If Statements', 'Switch Statements',
       'Cast Operations', 'Vector Operations', 'Total Integer Operations',
       'Total Float Operations', 'Integer Math Functions']]
    corrDF.columns = ["THR", "GML", "LML", "PML", "TL", "PL", "IF", "SWT", "CST", "VOP", "TIO", "TFO", "IMF"]
    
    corrmat = corrDF.corr()

    c = corrmat.abs()
    s = c.unstack()
    so = s.sort_values(kind="quicksort", ascending=False)
    sns_plot = sns.heatmap(corrmat, vmin=-1, vmax=1, square=True)
    fig = sns_plot.get_figure()
    fig.savefig(save_file)
    
def classifier_corrplot(df):
    corrmat = df.corr()
    c = corrmat.abs()
    s = c.unstack()
    so = s.sort_values(kind="quicksort", ascending=False)
    return so

def plot_feature_importances(train_cols, indices):
    plt.figure()
    plt.title('Top feature importances')
    plt.bar(
        range(len(train_cols)), 
        importances[indices],
        yerr=std[indices], 
    )
    plt.xticks(range(len(train_cols)), indices)
    plt.show()