import numpy as np

def generate_permuted_array(rows, cols=100):
    array = np.zeros((rows, cols), dtype=int)
    permutations = set()
    
    for i in range(rows):
        while True:
            perm = tuple(np.random.permutation(cols))
            if perm not in permutations:  # Đảm bảo mỗi hàng là duy nhất
                permutations.add(perm)
                array[i] = perm
                break
    
    return array

num_rows = 100
permuted_array = generate_permuted_array(num_rows)

np.save("/media/tuannl1/heavy_weight/FCL/PFLlib/dataset/class_order_cifar100.npy", permuted_array)

print(permuted_array)