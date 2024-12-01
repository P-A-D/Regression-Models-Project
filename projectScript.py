import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from tensorflow.python.tpu.ops.gen_xla_ops import xla_sparse_dense_matmul_grad_with_adagrad_momentum_and_static_buffer_size
from tf_keras.src.engine.base_layer_utils import infer_init_val_and_dtype


def aic(sse, n, k):
    return n*np.log(sse) - n*np.log(n) + 2*k

def sbc(sse, n, k):
    return n*np.log(sse) - n*np.log(n) + k*np.log(n)

def ck(sse_model, mse_full, n, k):
    return sse_model/mse_full - n + 2*k

def dffits(index, t_vector, h_ii_vector):
    return t_vector[index]*np.sqrt(h_ii_vector[index]/(1-h_ii_vector[index]))

def cooks_distance(index, e_vector, h_ii_vector, mse, p):
    return (e_vector[index])**2/p/mse*h_ii_vector[index]/(1 - h_ii_vector[index])**2

def check_multicollinearity(df, colnames=None):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    vif = [variance_inflation_factor(scaled, i) for i in range(scaled.shape[1])]
    if colnames is not None:
        return dict(zip(colnames, vif))
    return dict(zip(range(len(vif)), vif))

def fit_model(x, y, mse_full=None):
    n, p = x.shape
    p += 1
    model = LinearRegression(n_jobs=-1)
    model.fit(x, y)
    beta_hat = [model.intercept_, *model.coef_]
    y_hat = model.predict(x)
    e = y - y_hat
    sse = (e ** 2).sum()
    sst = (y ** 2).sum() - (1 / n) * np.linalg.multi_dot([y.T, np.ones((n, n)), y])
    ssr = sst - sse
    mse = sse/(n-p)
    r2 = ssr / sst * 100
    r2_adj = (1 - (n-1)/(n-p)*sse/sst)*100
    aic_met = aic(sse, n, p)
    sbc_met = sbc(sse, n, p)
    ck_met = None
    if mse_full is not None:
        ck_met = ck(sse, mse_full, n, p)
    return list(x.columns), beta_hat, y_hat, e, sse, sst, ssr, mse, r2, r2_adj, aic_met, sbc_met, p, ck_met

def model_selection(x, y):
    # preparing the full model
    output_names = ['features', 'beta_hat', 'sse', 'sst', 'ssr', 'mse', 'r2', 'r2_adj', 'aic_met', 'sbc_met', 'k', 'ck_met']
    models = []
    features, beta_hat, _, _, sse, sst, ssr, mse, r2, r2_adj, aic_met, sbc_met, p, ck_met = fit_model(x, y)
    results = dict(zip(output_names, (features, beta_hat, sse, sst, ssr, mse, r2, r2_adj, aic_met, sbc_met, p, ck_met)))
    mse_full = results['mse']
    results['ck_met'] = x.shape[1] + 1
    models.append(results)

    # preparing the combinations
    cols = list(x.columns)
    asd = [list(itertools.combinations(cols, i)) for i in range(1, len(cols))]
    combinations = []
    for i in asd:
        combinations += i
    combinations = [list(i) for i in combinations]

    # fitting models
    for comb in combinations:
        features, beta_hat, _, _, sse, sst, ssr, mse, r2, r2_adj, aic_met, sbc_met, p, ck_met = fit_model(x[comb], y, mse_full=mse_full)
        results = dict(zip(output_names, (features, beta_hat, sse, sst, ssr, mse, r2, r2_adj, aic_met, sbc_met, p, ck_met)))
        models.append(results)
    return pd.DataFrame(models)

def outlier_detection(x, y, alpha=0.05):
    x = x.copy()
    n, p = x.shape
    p += 1
    x.insert(0, 'x0', np.ones(n))
    xtx = np.linalg.multi_dot([x.T, x])
    xtx_inverse = np.linalg.inv(xtx)
    h = np.linalg.multi_dot([x, xtx_inverse, x.T])
    y_hat = np.dot(h, y)
    e = y - y_hat
    sse = np.dot(e, e)
    mse = sse / (n - p)

    # standardized residuals / internally studentized
    one_minus_hii = 1 - h.diagonal()
    r = e / np.sqrt(mse * one_minus_hii)

    # studentized deleted residuals / externally studentized
    t = list(e * np.sqrt((n - p - 1) / (sse * one_minus_hii - e ** 2)))

    # y outliers
    crit = stats.t.ppf(q=1 - alpha / (2 * n), df=n - p - 1)
    outlier_y_index = np.where(crit < np.abs(t))[0]

    # x outliers
    h_ii = h.diagonal()
    critical_value = 2 * p / n
    outlier_x_index = np.where(h_ii > critical_value)[0]

    # index of all outliers
    outlier_indices = list(set(list(outlier_y_index) + list(outlier_x_index)))

    # DFFITS
    dffits_values = [np.abs(dffits(i, t, h_ii)) for i in outlier_indices]
    dffits_crit = 2 * np.sqrt(p / (n - p))
    where = np.where(np.array(dffits_values) > dffits_crit)[0]
    dffits_influential_indices = (np.array(outlier_indices))[where]

    # Cook's
    e = list(e)
    cook = [cooks_distance(i, e, h_ii, mse, p) for i in outlier_indices]
    cook_crit = stats.f.ppf(0.5, p, n - p)
    cook_outliers = np.where(np.array(cook) > cook_crit)[0]
    cook_influential_indices = (np.array(outlier_indices))[cook_outliers]

    influential_outlier_indices = list(set(list(dffits_influential_indices) + list(cook_influential_indices)))
    return outlier_x_index, outlier_y_index, influential_outlier_indices, r, t

def breusch_pagan(x, y, alpha=0.05):
    x = x.copy()
    n, p = x.shape
    p += 1
    model = LinearRegression()
    model.fit(x, y)
    e = y - model.predict(x)
    e2 = e ** 2
    sse = e2.sum()

    model = LinearRegression()
    model.fit(x, e2)
    beta_hat = np.array([model.intercept_, *model.coef_])
    x.insert(0, 'x0', np.ones(n))

    ssr_star = np.linalg.multi_dot([beta_hat.T, x.T, e2]) - np.linalg.multi_dot([e2.T, np.ones((n, n)), e2])/n

    # getting the test statistic
    x0 = n ** 2 * ssr_star / (2 * sse ** 2)
    x_crit = stats.chi2.ppf(q=1 - alpha, df=p - 1)  # q=p-1=the number of predictor variables
    print(f"Test statistic = {x0}, critical value = {x_crit}")

def inferences(x, y, alpha=0.05):
    x = x.copy()
    n, p = x.shape
    p += 1
    model = LinearRegression()
    model.fit(x, y)
    beta_hat = np.array([model.intercept_, *model.coef_])
    x.insert(0, 'x0', np.ones(n))
    beta_covar_matrix = np.linalg.inv(np.dot(x.T, x))
    critical_value = stats.t.ppf(1-alpha/2, df=n - p)
    test_statistics = beta_hat/np.sqrt(beta_covar_matrix.diagonal())
    print(f"Critical value = {critical_value}\nTest statistics: {test_statistics}")
    significance = np.abs(test_statistics) > critical_value
    asd = ['significant', 'insignificant']
    is_significant = [f'beta_{i} is {asd[0] if significance[i] else asd[1]}' for i in range(p)]
    confidence_intervals = [f"beta_{i}: {beta_hat[i]:.5f} +- {(np.sqrt(beta_covar_matrix.diagonal())*critical_value)[i]:.5f}" for i in range(p)]
    return is_significant, confidence_intervals

if __name__ == '__main__':
    data = pd.read_csv("temp.csv")

    # removing the longitude, latitude, and ocean_proximity
    data = data.drop(['Unnamed: 0'], axis=1)
    data = data.dropna()
    data = data.reset_index(drop=True)
    # defining x and y
    y = data['medv']
    x = data.drop(['medv'], axis=1)
    n, p = x.shape
    p += 1
    a, b = inferences(x, y, alpha=0.05)
    pass