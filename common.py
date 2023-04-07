import numpy as np

def image_u8_decimate(img: np.ndarray, ffactor: float):
    height, width = img.shape

    if ffactor == 1.5:
        swidth = width // 3 * 2
        sheight = height // 3 * 2

        decim = np.zeros((sheight, swidth), np.uint8)

        y = 0
        sy = 0;
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
