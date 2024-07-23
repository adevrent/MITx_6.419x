import numpy as np

W_fh = 0
W_fx = 0
b_f = -100
W_ih = 0
W_ix = 100
b_i = 100
W_ch = -100
W_cx = 50
b_c = 0
W_oh = 0
W_ox = 100
b_o = 0

h_t = 0
C_t = 0

x_arr = np.array([1, 1, 0, 1, 1])

sigmoid = lambda z: 1 / (1 + np.exp(-z))

def f_func(W_h, W_x, b, h, x):
    term1 = np.dot(W_h, h) + np.dot(W_x, x) + b
    return sigmoid(term1)

h_arr = []
for x in x_arr:
    f = f_func(W_fh, W_fx, b_f, h_t, x)
    i = f_func(W_ih, W_ix, b_i, h_t, x)
    o = f_func(W_oh, W_ox, b_o, h_t, x)
    C_t = f * C_t + i * np.tanh(np.dot(W_ch, h_t) + np.dot(W_cx, x) + b_c)
    h_t = o * np.tanh(C_t)
    h_arr.append(np.int64(np.round(h_t)))

print("h_arr", h_arr)