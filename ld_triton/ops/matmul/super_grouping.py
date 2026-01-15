# Adapted from https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

import torch

def cdiv(a, b):
    return (a + b - 1) // b


def get_pidmn(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, pid):
    num_pid_m = cdiv(M, BLOCK_SIZE_M)
    num_pid_n = cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    return pid_m, pid_n


if __name__ == '__main__':
    M, N = 2048, 2048
    BLOCK_SIZE_M, BLOCK_SIZE_N = 128, 128
    num_pid_m = cdiv(M, BLOCK_SIZE_M)
    num_pid_n = cdiv(N, BLOCK_SIZE_N)
    GROUP_SIZE_M = 8
    super_grouping = torch.empty((num_pid_m, num_pid_n), dtype=torch.int)
    for i in range(num_pid_m * num_pid_n):
        idx = get_pidmn(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, i)
        super_grouping[idx] = i
    print(super_grouping)

'''
tensor([[  0,   8,  16,  24,  32,  40,  48,  56,  64,  72,  80,  88,  96, 104, 112, 120],
        [  1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,  97, 105, 113, 121],
        [  2,  10,  18,  26,  34,  42,  50,  58,  66,  74,  82,  90,  98, 106, 114, 122],
        [  3,  11,  19,  27,  35,  43,  51,  59,  67,  75,  83,  91,  99, 107, 115, 123],
        [  4,  12,  20,  28,  36,  44,  52,  60,  68,  76,  84,  92, 100, 108, 116, 124],
        [  5,  13,  21,  29,  37,  45,  53,  61,  69,  77,  85,  93, 101, 109, 117, 125],
        [  6,  14,  22,  30,  38,  46,  54,  62,  70,  78,  86,  94, 102, 110, 118, 126],
        [  7,  15,  23,  31,  39,  47,  55,  63,  71,  79,  87,  95, 103, 111, 119, 127],
        [128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248],
        [129, 137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249],
        [130, 138, 146, 154, 162, 170, 178, 186, 194, 202, 210, 218, 226, 234, 242, 250],
        [131, 139, 147, 155, 163, 171, 179, 187, 195, 203, 211, 219, 227, 235, 243, 251],
        [132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252],
        [133, 141, 149, 157, 165, 173, 181, 189, 197, 205, 213, 221, 229, 237, 245, 253],
        [134, 142, 150, 158, 166, 174, 182, 190, 198, 206, 214, 222, 230, 238, 246, 254],
        [135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239, 247, 255]], dtype=torch.int32)
'''
