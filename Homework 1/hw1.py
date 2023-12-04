
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
import seaborn as sns
import load_data as ld
import joblib
import numpy as np
import pandas as pd


def file_extractor(file):
    data = []
    with open(f'./{file}.txt','r') as f:
        data = [tuple(line[:-1].split(",")) for line in f.readlines()[1:-1]]
    for _ in sorted(data,key=lambda x: -float(x[-1])):
        print(_)

def model_testing(model, X_train, X_test, y_train, y_test, data_points_csv):
    model.fit(X_train,y_train)
    #print( model.score(X_test,y_test)) #0.973-1000 / 0.98815-100

    # Predictions csv
    blind_test_csv(data_points_csv, model.predict(data_points_csv))


    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test,y_pred)
    return  accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='macro'),\
            recall_score(y_test, y_pred, average='macro'),f1_score(y_test, y_pred, average='macro'),\
            conf_matrix


def hw1():
    pre_process,    pp,                                 result_file_name,    dataset,       selected_model=\
    True,          MinMaxScaler(feature_range=(0,1)),   "",                  "dataset2",    1

    models = [
        (
        LogisticRegression(C=1, penalty='l2', solver='newton-cg'), #pre
        LogisticRegression(C=0.001, penalty='l2', solver='saga'), #nopre
        LogisticRegression()
        ),
        (
        RandomForestClassifier(bootstrap= True, criterion= 'entropy', max_depth= None, max_features= 'log2', 
                                min_samples_leaf= 4, min_samples_split= 2, n_estimators= 200), #p
        RandomForestClassifier(bootstrap= True, criterion= 'entropy', max_depth= 10, max_features= 'log2', 
                                min_samples_leaf= 2, min_samples_split= 10, n_estimators= 100), #no pre
        RandomForestClassifier()
        ),
        (
        SVC(C=0.1, class_weight='balanced', degree=2, gamma=0.1, kernel='linear'), #pre
        SVC(C=1, class_weight=None, degree=2, gamma=0.1, kernel='poly'), #no pre
        SVC()
        ),
        (
        KNeighborsClassifier(metric='euclidean', n_neighbors=13, p=1, weights='distance'), #pre
        KNeighborsClassifier(metric='euclidean', n_neighbors=7, p=1, weights='uniform'), #no pre
        #KNeighborsClassifier(metric='euclidean', n_neighbors= int(np.sqrt(len(X_train))), p=1, weights='uniform'), 
        KNeighborsClassifier()
        )
    ]

    
    
    #==================================================== Data loading and preprocessing ====================================================   
    feature_vec, classes = ld.load_data(f"./{dataset}.csv") # Load data
    data_points_csv,_ = ld.load_data(f"./blind_test2.csv")
    '''
    balance = dict()
    for i in range(len(feature_vec)):
        if classes[i] in balance: balance[classes[i]]+=1
        else: balance[classes[i]]=1
    print(sorted(balance.items()))
    #(0, 5000), (1, 5000), (2, 5000), (3, 5000), (4, 5000), (5, 5000), (6, 5000), (7, 5000), \
    #(8, 5000), (9, 5000) Dataset1&2
    '''
    if pre_process: feature_vec = pp.fit_transform(feature_vec) # Apply preprocessing
    X_train, X_test, y_train, y_test = train_test_split(feature_vec, classes, test_size=0.6) #split into train,set,validate (validate is 20%)
    #========================================================================================================================================

    #==================================================== Model and Performance Analysis ====================================================
    model = models[selected_model][0]
    #model = models[selected_model][2] #base model

    if not pre_process:
        print("Model for no preprocessing")
        model = models[selected_model][1]

    a,p,r,f1,cm = model_testing(model, X_train, X_test, y_train, y_test,data_points_csv)

    '''
    print(f'Dataset: {dataset}\tPreprocessing active: {pre_process}')
    print(f'\item \\textbf {{Accuracy}}: {str(a)[:7]}\n\
            \item \\textbf {{Precision}}: {str(p)[:7]}\n\
            \item \\textbf {{Recall}}: {str(r)[:7]}\n\
            \item \\textbf {{F1 Score}}: {str(f1)[:7]}')
    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    '''
    

    #========================================================================================================================================

    #==================================================== Hyperparameter research via GS ====================================================
    #param_grid = {}
    #clf = GridSearchCV(model, param_grid, verbose=3, cv=5)  # You can adjust the number of folds in cv
    #joblib.dump(clf, f'{result_file_name}.pkl') # Save grid search results
    #========================================================================================================================================

def blind_test_csv(data_points, predictions):
    #df = pd.DataFrame({'x': data_points, 'y': predictions})
    # Write to the CSV file
    #df.to_csv(f'd1_{1849661}.csv', index=False)

    with open('d2_1849661.csv','w') as f:
        for i in range(len(predictions)):
            f.write(f'{data_points[i].tolist()},{predictions[i]}\n')


if __name__ == '__main__':
    hw1()