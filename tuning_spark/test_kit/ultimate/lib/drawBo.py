from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

x = np.linspace(-2, 10, 10000).reshape(-1, 1)
y = target(x)

plt.plot(x, y);
plt.show()

optimizer = BayesianOptimization(target, {'x': (-2, 10)}, random_state=27)

optimizer.maximize(init_points=2, n_iter=0, kappa=5)

def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(17, 10))
    steps = len(optimizer.space)
    # fig.suptitle(
    #     'Gaussian Process and Utility Function After {} Steps'.format(steps),
    #     fontdict={'size':30}
    # )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    #acq = plt.subplot(gs[1])
    for res in optimizer.res:
        print(res)

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=6, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=12, label=u'Observations', color='r')
    axis.plot(x, mu, '--', linewidth=4, color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    # axis.set_ylabel('f(x)', fontdict={'size':20})
    # axis.set_xlabel('x', fontdict={'size':20})

    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    axis.plot(x, 0.5*utility, label='Utility Function', color='purple',linewidth=4)
    axis.plot(x[np.argmax(utility)], 0.5*np.max(utility), '*', markersize=23,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=2)

    legend=axis.legend(loc=3, ncol=3, mode= 'expand', fontsize=22, bbox_to_anchor=(0., 1.02, 1., .102), borderaxespad=0.)
    plt.xticks([])
    plt.yticks([])

plot_gp(optimizer, x, y)

optimizer.maximize(init_points=0, n_iter=1, kappa=5)
plot_gp(optimizer, x, y)
# plt.savefig('./ob3.pdf')

optimizer.maximize(init_points=0, n_iter=1, kappa=5)
plot_gp(optimizer, x, y)
plt.savefig('./obtest.pdf')

optimizer.maximize(init_points=0, n_iter=1, kappa=5)
plot_gp(optimizer, x, y)
#plt.savefig('./ob5.pdf')

optimizer.maximize(init_points=0, n_iter=1, kappa=5)
plot_gp(optimizer, x, y)
#plt.savefig('./ob6.pdf')
