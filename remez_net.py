import argparse

import scipy.interpolate
import torch.nn.functional as F
from pylab import *
from pylab import plot
from sklearn.metrics import mean_squared_error
from torch import optim, nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from utils import *


class osc_loss(nn.Module):

    def __init__(self):
        super(osc_loss, self).__init__()

    def forward(self, res):
        a = np.empty((len(res),))
        a[::2] = 1
        a[1::2] = -1
        b = torch.mul(torch.FloatTensor(a).cuda(), res)

        loss = 0
        for _ in b:
            if _ < 0:
                loss -= _

        return loss


class osc_e_loss(nn.Module):

    def __init__(self):
        super(osc_e_loss, self).__init__()

    def forward(self, res, e):
        a = np.empty((len(res),))
        a[::2] = 1
        a[1::2] = -1
        b = torch.mul(torch.FloatTensor(a).cuda(), e)

        loss = F.mse_loss(res, b)

        return torch.sum(loss)


class poly_loss(nn.Module):

    def __init__(self):
        super(poly_loss, self).__init__()

    def forward(self, res):
        a = np.empty((len(res),))
        a[::2] = 1
        a[1::2] = -1
        b = torch.mul(torch.FloatTensor(a).cuda(), res)

        loss = 0
        for _ in b:
            if _ < 0:
                loss -= _

        return loss


class rational_net(nn.Module):
    def __init__(self, m_orders, n_orders):
        super(rational_net, self).__init__()

        self.m_orders = m_orders
        self.n_orders = n_orders

        a = np.empty((1, self.m_orders + 1))
        a[:, ::2] = 1
        a[:, 1::2] = -1
        self.weight_nu = Parameter(torch.FloatTensor(a))
        a = np.ones((1, self.m_orders))
        a[:, ::2] = 1
        a[:, 1::2] = -1
        self.weight_de = Parameter(torch.FloatTensor(a))

        # a = [[720/117649, -36/343, 232/343, -15/7, 25/7, -3, 1]]
        # b = [[1, -1, 1, -1, 1, -1, 1]]

        # Ground truth
        # self.weight_nu = Parameter(torch.FloatTensor(np.array(
        #     [[4.99999778e-01, - 4.74122263e+00, 1.91124430e+01,
        #       - 4.23571888e+01, 5.52049474e+01, - 4.08096792e+01, 1.35848690e+01]])))
        #
        # self.weight_de = Parameter(torch.FloatTensor(np.array([[
        #     1.0, - 7.48465143e+00, 2.33253432e+01,
        #     - 3.88944169e+01, 3.74874439e+01, - 2.16505504e+01, 7.20516856e+00]])))

        # (0.48651161789894104x^0+-1.5099493265151978x^1+1.109924077987671x^2+-0.6512492895126343x^3+1.515868067741394x^4+-0.4229825437068939x^5+1.622999668121338x^6)
        # /(1+-1.375675916671753x^1+0.7969802618026733x^2+0.020870914682745934x^3+2.3738303184509277x^4+0.1153063252568245x^5+1.4852185249328613x^6)

        nn.init.xavier_normal_(self.weight_de)
        nn.init.xavier_normal_(self.weight_nu)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_nu.size(1))

        self.weight_nu.data.uniform_(-stdv, stdv)
        self.weight_de.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        support = torch.squeeze(torch.mm(self.weight_nu, x))
        support_de = torch.squeeze(torch.mm(self.weight_de, x[1:])) + torch.ones(
            (self.m_orders + self.n_orders + 2)).cuda()
        output = torch.div(support, support_de)

        return output


class remez_net(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.gpu_mode = args.gpu_mode
        self.m = args.m
        self.n = args.n
        self.max_mn = args.m if args.m >= args.n else args.n
        self.zero_num = self.n + self.m + 2
        self.rat = rational_net(args.m, args.n)
        self.mse_list = []
        self.loss_mse = []
        self.loss_osc = []
        self.loss_mm = []

        # networks init
        self.rat = rational_net(self.max_mn, self.max_mn)
        self.E = Variable(to_cuda(torch.FloatTensor([1e-8])), requires_grad=True)
        self.optimizer = optim.Adam(list(self.rat.parameters()) + [self.E],
                                    lr=args.lr, weight_decay=5e-4,
                                    betas=(args.beta1, args.beta2))
        self.mse = nn.MSELoss()
        self.osc = osc_loss()
        self.poly = poly_loss()
        self.osc_e_loss = osc_e_loss()

        # model
        self.rat = to_cuda(self.rat)

        print('---------- Networks architecture -------------')
        print_network(self.rat)

    def chk_stop_condition(self):
        pass

    def plot_metric(self, x, y, outer, inner, xnodes, ynodes):
        # calculate criterion and show performance
        de = self.rat.weight_de.data.cpu().numpy()
        nu = self.rat.weight_nu.data.cpu().numpy()
        c = np.concatenate((nu, de), axis=1)[0]
        y_plot = rational(x, c, self.m, self.n)

        matplotlib.style.use('seaborn')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        # current ref
        # y_ref = rational(xnodes, c, self.m, self.n)
        # plt.scatter(xnodes, y_ref)
        plot(x, y_plot, markerfacecolor='salmon', markeredgewidth=1, markevery=slice(40, len(y_plot), 70),
             linestyle=':', marker='o', color='crimson', linewidth=3, label='fit')  # fit result
        plot(x, y, markerfacecolor='lime', markeredgewidth=1, markevery=slice(30, len(y_plot), 70), linestyle='-.',
             marker='v', color='dodgerblue', linewidth=3, label='func')  # ground truth
        plt.legend(loc="best", prop={'size': 30})
        offset = mean_squared_error(y, y_plot)
        self.mse_list.append(offset)
        if len(self.mse_list) > 10:
            cur_mse = self.mse_list[-10:]
            self.mse_list = cur_mse
        else:
            cur_mse = self.mse_list
        plt.title('[{}-{}]error: {}$\pm${}'.format(outer, inner, mean(cur_mse), std(cur_mse)))
        # plt.show()
        plt.savefig('{}_{}.png'.format(outer, inner))
        plt.close()

        # pk.dump(x, open('x_{}.pk'.format(opt), 'wb'))
        # pk.dump(y_plot, open('y_{}.pk'.format(opt), 'wb'))

        # plt.scatter(xnodes, y_ref - ynodes)  # current ref
        # plot(x, y_plot - y, label='err')  # err result
        # plt.show()

        res = y_plot - y
        print("res: {}/{}".format(np.max(np.abs(res)), np.min(np.abs(res))))
        print(rat_func_str(nu, de))

        return res

    def train(self, x, y):

        # nn.module train mode
        self.rat.train()

        # option 1: equal interval
        intv = 2 / (self.zero_num + 2)
        rand_init = np.random.uniform(-intv, intv)
        x_init = np.linspace(0, 2, self.zero_num + 2)[1:-1] + rand_init

        # option 2: chebyshev nodes
        # xylen = len(x)
        # xyindex = np.arange(0, self.zero_num) + 1
        # xyindex = .5 + .5 * np.cos(np.pi * (2 * xyindex - 1) / self.zero_num)
        # xyindex *= (xylen - 1)

        xnodes = np.zeros(self.zero_num)
        ynodes = np.zeros(self.zero_num)
        y_interp = scipy.interpolate.interp1d(x, y)  # interpolate
        y_exterp = extrap1d(y_interp)

        def pick():

            for i in range(x_init.shape[0]):
                xnodes[i] = x_init[i]
                ynodes[i] = y_exterp(x_init[i])

            # plot(xnodes, ynodes)
            # scatter(xnodes, ynodes)
            # show()
            # exit()
            return xnodes, ynodes

        xnodes, ynodes = pick()

        start_time = time.time()
        for _e in range(3):  # pick at most 3 times

            for _ in range(self.epoch):
                # nn optimize until convergence

                support = torch.FloatTensor(poly_recur(xnodes, orders=self.max_mn)).cuda()
                label = torch.FloatTensor(ynodes).cuda()
                yp = self.rat(support)

                res = yp - label
                loss1 = self.osc_e_loss(res, self.E)  # res - E
                loss2 = self.osc(res) * 1  # ocs constraint?
                loss3 = torch.max(torch.abs(res))

                loss = loss1 + loss2 + loss3

                self.loss_mse.append(loss1.item())
                self.loss_osc.append(loss2.item())
                self.loss_mm.append(loss3.item())

                if _ % 100 == 0:
                    print("epoch: {} loss {:.4f} = {:.8f} + {:.8f} + {:.8f}, E:{}, res:{}".format(_, loss, loss1, loss2,
                                                                                                  loss3,
                                                                                                  self.E.item(), res))
                    self.plot_metric(x, y, _e, _, xnodes, ynodes)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            res = self.plot_metric(x, y, _e, _, xnodes, ynodes)

            if self.chk_stop_condition():
                # check the diff between max / min of res convergence
                break
            else:
                print("repick the references")
                xnodes, ynodes = pick()

                # for i in range(self.zero_num - 1):
                #     a = min(res[xyindex[i]:xyindex[i + 1] + 1])
                #     b = max(res[xyindex[i]:xyindex[i + 1] + 1])
                #     i1 = xyindex[i] + nonzero(res[xyindex[i]:xyindex[i + 1] + 1] == a)[0][0]
                #     i2 = xyindex[i] + nonzero(res[xyindex[i]:xyindex[i + 1] + 1] == b)[0][0]
                #     if res[xyindex[i]] * a > 0:
                #         xyindex[i] = i1
                #     # print xyindex[i]
                #     if res[xyindex[i + 1]] * a > 0:
                #         xyindex[i + 1] = i1
                #     # print xyindex[i+1]
                #     if res[xyindex[i]] * b > 0:
                #         xyindex[i] = i2
                #     # print xyindex[i]
                #     if res[xyindex[i + 1]] * b > 0:
                #         xyindex[i + 1] = i2
                #     # print xyindex[i+1]
                #
                # xnodes = np.zeros(self.zero_num)
                # ynodes = np.zeros(self.zero_num)
                # for i in range(self.zero_num):
                #     xnodes[i] = x[xyindex[i]]
                #     ynodes[i] = y[xyindex[i]]

        end_time = time.time() - start_time
        print(time.strftime("%H:%M:%S", time.gmtime(end_time)))

        # pk.dump(self.loss_mse, open('{}_loss_mse.pk'.format(opt), 'wb'))
        # pk.dump(self.loss_osc, open('{}_loss_osc.pk'.format(opt), 'wb'))
        # pk.dump(self.loss_mm, open('{}_loss_mm.pk'.format(opt), 'wb'))

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        torch.save(self.G.state_dict(), os.path.join(self.save_dir, 'rat.pkl'))

    def load(self):
        self.rat.load_state_dict(torch.load(os.path.join(self.save_dir, 'rat.pkl')))


if __name__ == '__main__':
    desc = "RemezNet for rational approximation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--gpu_mode', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--batch', default=64, type=int, help='batch size. Default: 64')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate. Default: 0.1')
    parser.add_argument('--m', default=7, type=int, help='')
    parser.add_argument('--n', default=7, type=int, help='')
    parser.add_argument('--epoch', type=int, default=10000)
    args = parser.parse_args()

    # gen synthetic data
    # opt = 5
    # x = np.linspace(0, 1, 500)
    # y = func(x, opt=opt)
    #
    # rez = remez_net(args)
    # rez.train(x, y, opt)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = gen_data()

    e, U = LA.eigh(adj.A)
    x_hat = np.dot(U.T, features)
    y_hat = np.dot(U.T, labels)
    y = np.divide(y_hat, x_hat)

    # plot(e[idx_train], y[idx_train])
    # savefig('target2approx.png')
    # exit()

    e = e[idx_train]
    y = y[idx_train].flatten()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    rez = remez_net(args)
    rez.train(e, y)
