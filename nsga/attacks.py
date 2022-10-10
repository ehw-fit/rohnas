import tensorflow as tf
import numpy as np
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
import time
#checkpoint_path =  args.save_dir + '/best_weights_2.h5'

def resize(data_set, size):
    X_temp = []
    import scipy
    for i in range(data_set.shape[0]):
        resized = scipy.misc.imresize(data_set[i], (size, size))
        X_temp.append(resized)
    X_temp = np.array(X_temp, dtype=np.float32) / 255.
    return X_temp

def get_next_batch(x, y, start, end):
    """
    Fetch the next batch of input images and labels
    :param x: all input images
    :param y: all labels
    :param start: first image number
    :param end: last image number
    :return: batch of images and their corresponding labels
    """
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch




def PGD_linf(model,x_test,y_true, max_epsilon, sess = None, args = None, cache_file = "."):
    #with tf.Session() as sess:
    input_size = x_test.shape[0]
    #model.load_weights(checkpoint_path)
    #pred, a1 = model.predict(x_test)
    # acc =  np.mean(np.equal(np.argmax(pred,axis = 1), np.argmax(y_true,axis = 1)))

    # print("The normal validation accuracy is: {}".format(acc))
    batch_size =  200 #args.batch_size
    model.summary()
    #model.compile()
    wrap = KerasModelWrapper(model)
    #wrap.summary()
    #xxxxx
    attack = ProjectedGradientDescent(wrap, sess=sess)
    print("gg graphs: ", attack.graphs.keys())
    all_acc = []
    all_loss = []

    print("X shape", x_test.shape)
    print("Y shape", y_true.shape)

    c_size = "-".join(map(str, x_test.shape[1:]))
    for eps in max_epsilon:
            #print(eps)
            #cache_name = cache_file.format(size=c_size, eps="{:g}".format(eps))
            #print("cache:",  cache_name)
            #
            #try:
            #    x_adv_all = np.load(cache_name)["val"]
            #    print("### using cache for adv samples")
            #except FileNotFoundError:
            start = time.time()
            x_adv_all = np.zeros([0] + list(x_test.shape[1:]))
            for i in range(input_size//batch_size):
                x_test1 = x_test[i*batch_size:batch_size*(i+1)]
                y_true1 = y_true[i*batch_size:batch_size*(i+1)]
                PGD_params = {'eps':eps,
                            'eps_iter':eps/10,
                            'nb_iter':10,
                            'y': y_true1,
                            'ord':np.inf,
                            'clip_min':0,
                            'clip_max':1,
                            'y_target':None,
                            'rand_init':None
                            }
                #print("gg graphs: ", attack.graphs.keys())
    #            print("xtest1 shape", x_test1.shape)
                adv_x  = attack.generate_np(x_test1, **PGD_params)
                x_adv_all = np.concatenate((x_adv_all, adv_x))
                #print(i)
            x_adv_all = np.asarray(x_adv_all)
            #np.savez(cache_name, val=x_adv_all)

            #x_adv_all = resize(x_adv_all,32)
            #y, decoder = model.predict(x_adv_all)
            #y = model.predict(x_adv_all)
            #print(y)
            #y_pred_adv = np.argmax(y, axis=1)
            #plot_adv_samples_CW(x_test,adv_x,np.argmax(y_test, axis=-1),y_pred,np.argmax(y_target, axis=-1))

            #print(adv_x.shape)
            #print('y_true',np.argmax(y_true, axis = 1))
            #print('y_pred',y_pred_adv)
            # print(mean)
            # print(obestl2)
            #plt.imshow(adv_x[0].reshape((40, 40)))
            #plt.show()
            
            #a = model.evaluate(x_adv_all, [y_true, x_test])
            a = model.evaluate(x_adv_all, y_true) #, x_test])
            print(a)
            acc_adv = a[1] 
            loss_adv = a[0] 
            ##### VM y_pred, decoder = model.predict(x_test)
            #y_pred = model.predict(x_test)
            print("Epsilon={0}, Test loss: {1:.4f}, Test accuracy: {2:.01%}, Time: {3:.2f} sec".format(eps , loss_adv, acc_adv, time.time() - start))
            # plot_adv_samples(x_test, adv_x,
            #                     np.argmax(y_true, axis=1), y_pred_adv.astype(int), np.argmax(y_true, axis=1),
            #                     10, eps, n_samples_per_class=5)
            all_acc = np.append(all_acc, acc_adv)
            all_loss = np.append(all_loss, loss_adv)
            #f.write("eps:{:.4g} loss:{:.4g} acc:{:.4g}\n".format(eps,loss_adv,acc_adv)) 
    #f.close()
    return max_epsilon, all_acc, all_loss

