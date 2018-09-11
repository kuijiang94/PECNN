import tensorflow as tf

def _PS(X, r, n_out_channel):
        if n_out_channel >= 1:
            assert int(X.get_shape()[-1]) == (r ** 2) * n_out_channel, _err_log
            bsize, a, b, c = X.get_shape().as_list()
            bsize = tf.shape(X)[0] # Handling Dimension(None) type for undefined batch dim
            # X = tf.cast(X, tf.int32)
            Xs = tf.split(X, r, 3) #b*h*w*r*r dtype
            # Xs = tf.split(X, r, 3)
            Xr = tf.concat(Xs, 2) #b*h*(r*w)*r
            X = tf.reshape(Xr, (bsize, r*a, r*b, n_out_channel)) # b*(r*h)*(r*w)*c
        else:
            print(_err_log)
        return X
def down_sample(X):
    bs, a, b, c = X.get_shape().as_list()
    b = (int)(b)
    a= (int)(a)
    c = (int)(c)
    OFF = 1
    scale = 3
    #img0 = tf.Variable(tf.zeros([bs, a, b, c]))
    img1 = tf.slice(X,[0,0,0,0],[bs, a, b-OFF, c])
    img11 = tf.slice(X,[0,0,0,0],[bs, a, OFF, c])
    image1 = tf.concat([img1,img11],2)
    #image1 = tf.concat([img11,img1],2)
    img2 = tf.slice(X,[0,0,0,0],[bs, a-OFF, b, c])
    img22 = tf.slice(X,[0,0,0,0],[bs, OFF, b, c])
    image2 = tf.concat([img2,img22],1)
    #image2 = tf.concat([img22,img2],1)
    img3 = tf.slice(X,[0,0,OFF,0],[bs, a, b-OFF, c])
    img33 = tf.slice(X,[0,0,b-OFF,0],[bs, a, OFF, c])
    image3 = tf.concat([img3,img33],2)
    img4 = tf.slice(X,[0,OFF,0,0],[bs, a-OFF, b, c])
    img44 = tf.slice(X,[0,a-OFF,0,0],[bs, OFF, b, c])
    image4 = tf.concat([img4,img44],1)
    
    '''def offset(img, x, y=None):
        x = (int)(x)
        if y is None:
            y = x
        else:
            y = (int)(y)
        if x < 0:
            for i in range(img.shape[1] + x):
                img0[:, i-x, :, :] = img[:, i, :, :]
            for i in range(-x):
                img0[:, i, :, :] = img[:, i, :, :]
        else:
            for i in range(img.shape[1] - x):
                img0[:, i, :, :] = img[:, i + x, :, :]
            for i in range(img.shape[1] - x, img.shape[1], 1):
                img0[:, i, :, :] = img[:, i, :, :]

        if y < 0:
            for i in range(img.shape[2] + y):
                img0[:, :, i-y, :] = img[:, :, i: :]
            for i in range(-y):
                img0[:, :, i, :] = img[:, :, i: :]
        else:
            for i in range(img.shape[2] - y):
                img0[:, :, i, :] = img[:, :, i + y, :]
            for i in range(img.shape[2] - y, img.shape[2], 1):
                img0[:, :, i, :] = img[:, :, i, :]

        return img0'''

    #img1 = offset(X, (OFF), (0))
    #image1 = image1.resize((bs, a//scale, b//scale, c), interpolation=cv2.INTER_CUBIC)
    image1 = tf.image.resize_images(image1, [a//scale, b//scale], method=1)
    #img2 = offset(X, (-OFF), (0))
    image2 = tf.image.resize_images(image2, [a//scale, b//scale], method=1)
    #image2 = cv2.resize(image2, (bs, a//scale, b//scale, c), interpolation=cv2.INTER_CUBIC)
    #img3 = offset(X, (0), (OFF))
    image3 = tf.image.resize_images(image3, [a//scale, b//scale], method=1)
    #image3 = cv2.resize(image3, (bs, a//scale, b//scale, c), interpolation=cv2.INTER_CUBIC)
    #img4 = offset(X, (0), (-OFF))
    image4 = tf.image.resize_images(image4, [a//scale, b//scale], method=1)
    #image4 = cv2.resize(image4, (bs, a//scale, b//scale, c), interpolation=cv2.INTER_CUBIC)
    image5 = tf.image.resize_images(X, [a//scale, b//scale], method=1)
    #image5 = cv2.resize(X, (bs, a//scale, b//scale, c), interpolation=cv2.INTER_CUBIC)
    image = tf.concat([image1,image2,image3,image4,image5],3)
    return image