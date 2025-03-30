import numpy as np

def metric_average_forgetting(task, accuracy_matrix):
    result = 0
    F_list = []
    
    for i in range(task-1):
        acc_values = [accuracy_matrix[t][i] for t in range(task-1)]
        max_acc = max(acc_values) if acc_values else 0.0
        
        current_acc = accuracy_matrix[task][i]

        F_list.append(max_acc - current_acc)
    
    result = np.mean(F_list) if F_list else 0
    
    return result