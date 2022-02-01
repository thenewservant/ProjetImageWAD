import canny

im = plt.imread("../coinimages/fullpng/many.png")
#im = plt.imread("../coinimages/fullpng/4coinsflash.png")
width = im.shape[1]
height = im.shape[0]

low, high, blurfactor = 0,8,10
outim = np.zeros(im.shape)
keep = canny.canny_edge_detector(im, low, high, blurfactor)
outim[keep[:,0], keep[:,1]] = [1.,0,0,1.]
filename = "l{}h{}b{}mult2.png".format(low, high, blurfactor)
image.imsave(filename,outim)
