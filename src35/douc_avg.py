import mnist_loader
import numpy as np


def get_averages():
    a = mnist_loader.load_data_wrapper()
    sums = np.zeros((10,2))

    for pic_tup in a[0]:
        picture = pic_tup[0]
        value_vector = pic_tup[1]
        value = np.where(value_vector == 1)[0]
        avg = np.average(picture)
        sums[value,0] = sums[value,0] + 1
        sums[value,1] = sums[value,1] + avg

    avg = (sums[:, 0] /sums[:, 1])
    return avg

# print(get_averages())

def guess_number(pic_avg):
    pass


exit()












def show_image():
    tup = next(a[1])
    pic = tup[0]
    for i in range(28):
        pix_row = pic[i*28:(i+1)*28]
        for pixel in pix_row:
            if pixel > 0.9:
                print('OO', end='')
            else:
                print(' ', end='')
        print('')