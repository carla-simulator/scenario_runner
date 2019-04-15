import dlib

from configs import g_conf



def adjust_learning_rate(optimizer, num_iters):
    """
    Adjusts the learning rate every epoch based on the selected schedule
    """
    cur_iters = num_iters
    minlr = 0.0000001
    scheduler = "normal"
    learning_rate = g_conf.LEARNING_RATE
    decayinterval = g_conf.LEARNING_RATE_DECAY_INTERVAL
    decaylevel = g_conf.LEARNING_RATE_DECAY_LEVEL
    if scheduler == "normal":
        while cur_iters >= decayinterval:
            learning_rate = learning_rate * decaylevel
            cur_iters = cur_iters - decayinterval
        learning_rate = max(learning_rate, minlr)

    for param_group in optimizer.param_groups:
        print("New Learning rate is ", learning_rate)
        param_group['lr'] = learning_rate


def adjust_learning_rate_auto(optimizer, loss_window):
    """
    Adjusts the learning rate every epoch based on the selected schedule
    """
    minlr = 0.0000001
    learning_rate = g_conf.LEARNING_RATE
    thresh = g_conf.LEARNING_RATE_THRESHOLD
    decaylevel = g_conf.LEARNING_RATE_DECAY_LEVEL
    n = 1000
    start_point = 0
    print ("Loss window ", loss_window)
    while n < len(loss_window):
        print ("Startpoint ", start_point, " N  ", n)
        steps_no_decrease = dlib.count_steps_without_decrease(loss_window[start_point:n])
        steps_no_decrease_robust = dlib.count_steps_without_decrease_robust(loss_window[start_point:n])
        print ("no decrease, ", steps_no_decrease, " robust", steps_no_decrease_robust)
        if steps_no_decrease > thresh and steps_no_decrease_robust > thresh:
            start_point = n
            learning_rate = learning_rate * decaylevel

        n += 1000

    learning_rate = max(learning_rate, minlr)

    for param_group in optimizer.param_groups:
        print("New Learning rate is ", learning_rate)
        param_group['lr'] = learning_rate
