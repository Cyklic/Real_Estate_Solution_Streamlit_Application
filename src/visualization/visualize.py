
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def plot_violin(data):
    # box plot of beds
    # Create subplots
    fig, axs = plt.subplots(nrows=3, figsize=(10, 8))  # 3 rows, 1 column

    # Plot violin plots on each subplot
    sns.violinplot(x=data.beds, orient='h', ax=axs[0])
    axs[0].set_title('Beds')

    sns.violinplot(x=data.sqft, orient='h', ax=axs[1])
    axs[1].set_title('Sqft')

    sns.violinplot(x=data.lot_size, orient='h', ax=axs[2])
    axs[2].set_title('Lot Size')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save to file
    plt.savefig("combined_violin_plots.png")  # You can use .jpg, .svg, etc.

    # Optional: Show plot
    plt.show()

# def plot_feature_importance(lrmodel, x):
#     """
#     Plot a bar chart showing the feature importances.
    
#     Args:
#         feature_names (list): List of feature names.
#         feature_importances (list): List of feature importance values.
#     """
#     fig, ax = plt.subplots()
#     ax = sns.barplot(x=lrmodel.feature_importances_, y=x.columns)
#     plt.title("Feature importance chart")
#     plt.xlabel("Importance")
#     plt.ylabel("Feature")
#     plt.tight_layout()
#     fig.savefig("feature_importance.png")

# def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix'):
#     """
#     Plot the confusion matrix for the given true and predicted labels.
    
#     Args:
#         y_true (numpy.ndarray): Array of true labels.
#         y_pred (numpy.ndarray): Array of predicted labels.
#         classes (list): List of class labels.
#         normalize (bool, optional): Whether to normalize the confusion matrix. Default is False.
#         title (str, optional): Title for the plot. Default is 'Confusion Matrix'.
#     """
#     cm = confusion_matrix(y_true, y_pred)
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.xlabel('Predicted', fontsize=12)
#     plt.ylabel('Actual', fontsize=12)
#     plt.title(title, fontsize=16)
#     plt.show()