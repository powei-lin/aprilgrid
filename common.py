from functools import wraps
from time import perf_counter
import numpy as np
from unionfind import Unionfind, unionfind_connect

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def image_u8_decimate(img: np.ndarray, ffactor: float):
    height, width = img.shape

    if ffactor == 1.5:
        swidth = width // 3 * 2
        sheight = height // 3 * 2

        decim = np.zeros((sheight, swidth), np.uint8)

        y = 0
        sy = 0
        while (sy < sheight):
            x = 0
            sx = 0
            while (sx < swidth):

                # // a b c
                # // d e f
                # // g h i
                a = img[y+0, (x+0)]
                b = img[y+0, (x+1)]
                c = img[y+0, (x+2)]

                d = img[y+1, (x+0)]
                e = img[y+1, (x+1)]
                f = img[y+1, (x+2)]

                g = img[y+2, (x+0)]
                h = img[y+2, (x+1)]
                i = img[y+2, (x+2)]

                decim[(sy+0), (sx + 0)] = (4*a+2*b+2*d+e)/9
                decim[(sy+0), (sx + 1)] = (4*c+2*b+2*f+e)/9

                decim[(sy+1), (sx + 0)] = (4*g+2*d+2*h+e)/9
                decim[(sy+1), (sx + 1)] = (4*i+2*f+2*h+e)/9

                x += 3
                sx += 2

            y += 3
            sy += 2

        return decim

#     int factor = (int) ffactor;

#     int swidth = 1 + (width - 1)/factor;
#     int sheight = 1 + (height - 1)/factor;
#     image_u8_t *decim = image_u8_create(swidth, sheight);
#     int sy = 0;
#     for (int y = 0; y < height; y += factor) {
#         int sx = 0;
#         for (int x = 0; x < width; x += factor) {
#             decim->buf[sy*decim->stride + sx] = im->buf[y*im->stride + x];
#             sx++;
#         }
#         sy++;
#     }
#     return decim;
# }

# again use coprime dimensions for debugging safety
def max_pool(arr, block_size: int, _max=True):
    
    h, w = arr.shape  # pretend we only have this
    hs, r0 = divmod(h, block_size)
    ws, r1 = divmod(w, block_size)
    pooled = arr[:h-r0, :w-r1].reshape(hs, block_size,
                        ws, block_size)
    if _max:
        return pooled.max((1, 3))
    else:
        return pooled.min((1, 3))

def do_unionfind_first_line(uf: Unionfind, im: np.ndarray, w: int):
    y = 0
    # 將像素值不等於 127 的索引提取出來
    idx = np.where(im[y, 1:w-1] != 127)[0] + 1

    # 取出對應的像素值
    vals = im[y, idx]

    # 計算左側像素的索引
    left_idx = idx - 1

    # 找出左側像素與當前像素值相等的索引
    eq_idx = np.where(im[y, left_idx] == vals)[0]

    # 聯合左側像素
    for i in eq_idx:
        unionfind_connect(uf, y * w + idx[i], y * w + left_idx[i])

def do_unionfind_line2(uf: Unionfind, im: np.ndarray, w: int, y: int):
    assert(y > 0)

    v_m1_m1 = 0
    v_0_m1 = im[(y - 1), 0]
    v_1_m1 = im[(y - 1), 1]
    v_m1_0 = 0
    v = im[y, 0]

    for x in range(1, w - 1):
        v_m1_m1 = v_0_m1
        v_0_m1 = v_1_m1
        v_1_m1 = im[(y - 1), x + 1]
        v_m1_0 = v
        v = im[y, x]

        if (v == 127):
            continue

        # (dx, dy) pairs for 8 connectivity:
        # (-1, -1)    (0, -1)    (1, -1)
        # (-1, 0)    (REFERENCE)
        dx, dy = -1, 0
        if (im[(y + dy), x + dx] == v):
            unionfind_connect(uf, y*w + x, (y + dy)*w + x + dx)

        if (x == 1 or not ((v_m1_0 == v_m1_m1) and (v_m1_m1 == v_0_m1))):
            dx, dy = 0, -1
            if (im[(y + dy), x + dx] == v):
                unionfind_connect(uf, y*w + x, (y + dy)*w + x + dx)

        if (v == 255):
            if (x == 1 or not (v_m1_0 == v_m1_m1 or v_0_m1 == v_m1_m1)):
                dx, dy = -1, -1
                if (im[(y + dy), x + dx] == v):
                    unionfind_connect(uf, y*w + x, (y + dy)*w + x + dx)
            if (not (v_0_m1 == v_1_m1)):
                dx, dy = 1, -1
                if (im[(y + dy), x + dx] == v):
                    unionfind_connect(uf, y*w + x, (y + dy)*w + x + dx)