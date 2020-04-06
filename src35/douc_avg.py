import mnist_loader
import numpy as np



all_pics = mnist_loader.load_data_wrapper()

def get_averages(training_pics):
    """
    training_pics = training set with correct value as vector
    return one dim array 
    """
    sums = np.zeros((10,2))

    for pic_tup in training_pics:
        picture = pic_tup[0]
        # transform because it is vector in all_pics[0]
        value = np.where(pic_tup[1] == 1)[0]

        avg = np.average(picture)
        sums[value,0] = sums[value,0] + 1
        sums[value,1] = sums[value,1] + avg

    return sums[:, 1] / sums[:, 0]

AVGS = np.array([0.17330245, 0.07609738, 0.14833457, 0.141417, 0.12145503,
0.12799294, 0.13696827, 0.11464137, 0.15035786, 0.12250709])

def sorted_avgs(avgs):
    """
    avgs = numpy array with avg darkness values for numbers 0-9
    returns 2 dimensional sorted array with value and number
    """
    # reshape so that can be appended
    avgs = avgs.reshape((1, 10))
    order = np.array([i for i in range(10)])
    order = order.reshape((1, 10))
    # appended to preserve numbers with values after sorting
    table = np.append(avgs, order, axis=0)

    #create sorting key that can be applied to the table
    sorting_key = table[0,:].argsort()
    avgs = table[:, sorting_key]
    # print(avgs.shape)
    avgs = avgs.transpose()

    new_limits = []
    for i,j in zip(avgs[:-1], avgs[1:]):
        nl = (i[0] + j[0]) / 2
        val = i[1]
        new_limits.append((nl, val))
    new_limits.append([1, avgs[len(avgs)-1, 1]])

    return np.array(new_limits)


def guess_number(avg, sorted_avgs):
    """
    avg = average from pic form zip file  
    sorted_avgs = sorted averages calculated on training set  
    returns guessed number  
    """
    # avg = np.average(pic)
    for i in range(len(sorted_avgs)):
        upper_limit = sorted_avgs[i,0]
        n = sorted_avgs[i,1]
        if avg < upper_limit:
            return n
    return n


def gess_all_numbers(pics, sorted_avgs):
    """
    pics = dataset with pictures
    sorted_avgs = 2dim array with average vlaues 
    returns average matching % as float <0,1>
    """
    result = [] # prvni prvek = avg, druhy = co to je, treti = odhad
    for pair in pics:
        avg = np.average(pair[0])
        guess = guess_number(np.average(pair[0]), sorted_avgs)
        result.append((avg, pair[1], guess))
 
    return result
    




# x = next(all_pics[0])
# pic = x[0]
# number = np.where(x[1] == 1)[0]

# print("should be", number)
# print("pic avg", np.average(pic))
# sorted_a = sorted_avgs(avgs)
# print(sorted_a)
# print("----")
# print(gess_number(pic, sorted_avgs(avgs)))

# exit()


def show_image(a):
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

if __name__ == "__main__":
    result = np.array(gess_all_numbers(all_pics[1], sorted_avgs(AVGS)))
    result = np.where(result[:,1] == result[:,2], 1, 0)
    print("match       ", np.average(result))
    print("# of matches", np.sum(result))