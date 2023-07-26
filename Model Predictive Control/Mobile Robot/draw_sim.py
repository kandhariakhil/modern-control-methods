import matplotlib.pyplot as plt
import numpy as np

def Draw_MPC_point_stabilization(t, xx, xx1, u_cl, xs, N, rob_diam):
    line_width = 1.5
    fontsize_labels = 14

    # Simulate robots
    x_r_1 = []
    y_r_1 = []

    r = rob_diam / 2  # obstacle radius
    ang = 0.0
    xp = []
    yp = []
    while ang < 2 * np.pi:
        xp.append(r * np.cos(ang))
        yp.append(r * np.sin(ang))
        ang += 0.005

    fig = plt.figure(500)
    fig.set_size_inches(10, 10)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    for k in range(xx.shape[1]):
        h_t = 0.14
        w_t = 0.09

        x1 = xs[0]
        y1 = xs[1]
        th1 = xs[2]
        x1_tri = [x1 + h_t * np.cos(th1), x1 + (w_t / 2) * np.cos((np.pi / 2) - th1), x1 - (w_t / 2) * np.cos((np.pi / 2) - th1)]
        y1_tri = [y1 + h_t * np.sin(th1), y1 - (w_t / 2) * np.sin((np.pi / 2) - th1), y1 + (w_t / 2) * np.sin((np.pi / 2) - th1)]
        plt.fill(x1_tri, y1_tri, 'g')

        x1 = xx[0, k, 0]
        y1 = xx[1, k, 0]
        th1 = xx[2, k, 0]
        x_r_1.append(x1)
        y_r_1.append(y1)
        x1_tri = [x1 + h_t * np.cos(th1), x1 + (w_t / 2) * np.cos((np.pi / 2) - th1), x1 - (w_t / 2) * np.cos((np.pi / 2) - th1)]
        y1_tri = [y1 + h_t * np.sin(th1), y1 - (w_t / 2) * np.sin((np.pi / 2) - th1), y1 + (w_t / 2) * np.sin((np.pi / 2) - th1)]

        plt.plot(x_r_1, y_r_1, '-r', linewidth=line_width)
        if k < xx.shape[1]:
            plt.plot(xx1[:N, 0, k], xx1[:N, 1, k], 'r--*')

        plt.fill(x1_tri, y1_tri, 'r')
        plt.plot(x1 + xp, y1 + yp, '--r')

        plt.xlabel('$x$-position (m)', fontsize=fontsize_labels)
        plt.ylabel('$y$-position (m)', fontsize=fontsize_labels)
        plt.axis([-0.2, 1.8, -0.2, 1.8])
        plt.grid(True)
        plt.pause(0.1)
        plt.clf()

    plt.figure()
    plt.subplot(211)
    plt.stairs(t, u_cl[:, 0], 'k', linewidth=line_width)
    plt.axis([0, t[-1], -0.35, 0.75])
    plt.ylabel('v (rad/s)')
    plt.grid(True)

    plt.subplot(212)
    plt.stairs(t, u_cl[:, 1], 'r', linewidth=line_width)
    plt.axis([0, t[-1], -0.85, 0.85])
    plt.xlabel('time (seconds)')
    plt.ylabel('$\omega$ (rad/s)')
    plt.grid(True)

    plt.show()
