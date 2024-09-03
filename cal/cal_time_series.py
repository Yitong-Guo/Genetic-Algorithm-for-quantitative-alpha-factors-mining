import torch


def rolling_window(x, window=10):
    """
    Obtain a 3D tensor of all data of a terminal
    Shape=(number of windows, number of tokens, window)
    """
    x = x.unfold(0, window, 1)
    x_copy = x.clone()

    for i in range(1, window):
        tar = torch.full((x[0].shape[0], x[0].shape[1]), float('nan'))
        tar[:, i:] = x[0][:, :window - i]
        x_copy = torch.cat([tar.unsqueeze(0), x_copy], dim=0)

    return x_copy


def ts_max(x1, window=10):
    """Rolling calculate the maximum value of the sequence for each token's past d day"""
    x = rolling_window(x=x1, window=window)

    mask_x = torch.isnan(x)
    mean_x = x.nanmean(dim=2, keepdim=True)
    # Change the nan to the average value of the row, that will not affect selecting the maximum and minimum values
    x[mask_x] = mean_x.repeat(1, 1, window)[mask_x]

    max_x, _ = torch.max(x, dim=2)

    return max_x


def ts_min(x1, window=10):
    """Rolling calculate the minimum value of the sequence for each token's past d day"""
    x = rolling_window(x=x1, window=window)

    mask_x = torch.isnan(x)
    mean_x = x.nanmean(dim=2, keepdim=True)
    x[mask_x] = mean_x.repeat(1, 1, window)[mask_x]

    min_x, _ = torch.min(x, dim=2)

    return min_x


def ts_argmax(x1, window=10):
    """Rolling calculate the index of maximum value of the sequence for each token's past d day, then divide window"""
    x = rolling_window(x=x1, window=window)

    mask_x = torch.isnan(x)
    mean_x = x.nanmean(dim=2, keepdim=True)
    x[mask_x] = mean_x.repeat(1, 1, window)[mask_x]

    max_x, argmax_x = torch.max(x, dim=2)
    argmax_x = argmax_x.float()

    mask_argmax_x = torch.isnan(max_x)
    argmax_x[mask_argmax_x] = torch.nan

    return argmax_x / window + 1


def ts_argmin(x1, window=10):
    """Rolling calculate the index of minimum value of the sequence for each token's past d day, then divide window"""
    x = rolling_window(x=x1, window=window)

    mask_x = torch.isnan(x)
    mean_x = x.nanmean(dim=2, keepdim=True)
    x[mask_x] = mean_x.repeat(1, 1, window)[mask_x]

    min_x, argmin_x = torch.min(x, dim=2)
    argmin_x = argmin_x.float()

    mask_argmin_x = torch.isnan(min_x)
    argmin_x[mask_argmin_x] = torch.nan

    return argmin_x / window + 1


def ts_sum(x1, window=10):
    """Rolling calculate the sum of the sequence for each token's past d day"""
    x = rolling_window(x=x1, window=window)
    sum_x = x.nansum(dim=2, keepdim=True)

    mask_x = ~torch.isnan(x)
    count_x = mask_x.sum(dim=2, keepdim=True).float()
    mask_x0 = (count_x == 0) | (count_x == 1)
    count_x[mask_x0] = torch.nan
    count_x[~mask_x0] = 1  # This operation is to empty the position that was originally a null value

    mean_x = sum_x / count_x
    return mean_x.reshape(-1, mean_x.shape[1])


def ts_mean(x1, window=10):
    """Rolling calculate the mean value of the sequence for each token's past d day"""
    x = rolling_window(x=x1, window=window)
    sum_x = x.nansum(dim=2, keepdim=True)

    mask_x = ~torch.isnan(x)
    count_x = mask_x.sum(dim=2, keepdim=True).float()
    mask_x0 = (count_x == 0) | (count_x == 1)
    count_x[mask_x0] = torch.nan

    mean_x = sum_x / count_x
    return mean_x.reshape(-1, mean_x.shape[1])


def ts_weighted_mean(x1, x2, window=10):
    """In rolling windows, calculate the mean of x1 with normalized x2 as the weight"""
    x = rolling_window(x1, window=window)
    y = rolling_window(x2, window=window)

    sum_y = y.nansum(dim=2, keepdim=True)
    weight = y / sum_y

    weighted_x = x * weight
    weighted_mean_x = weighted_x.nansum(dim=2, keepdim=True)

    mask_x = ~torch.isnan(x)
    count_x = mask_x.sum(dim=2, keepdim=True).float()
    mask_x0 = (count_x == 0) | (count_x == 1)
    count_x[mask_x0] = torch.nan
    count_x[~mask_x0] = 1

    weighted_mean_x = weighted_mean_x / count_x

    return weighted_mean_x.reshape(-1, weighted_mean_x.shape[1])


def ts_stddev(x1, window=10):
    """Rolling calculate the std value of the sequence for each token's past d day"""
    x = rolling_window(x=x1, window=window)

    mean_x = x.nanmean(dim=2, keepdim=True)
    delta_x = x - mean_x  #

    mask_x = ~torch.isnan(delta_x)
    count_x = mask_x.sum(dim=2, keepdim=True).float()
    mask_x0 = (count_x == 0) | (count_x == 1)
    count_x[mask_x0] = torch.nan

    std_x = ((delta_x ** 2).nansum(dim=2, keepdim=True) / (count_x - 1)) ** 0.5 + 0.01
    return std_x.reshape(-1, std_x.shape[1])


def ts_var(x1, window=10):
    """Rolling calculate the var value of the sequence for each token's past d day"""
    x = rolling_window(x=x1, window=window)

    mean_x = x.nanmean(dim=2, keepdim=True)
    delta_x = x - mean_x  # 经过了广播

    mask_x = ~torch.isnan(delta_x)
    count_x = mask_x.sum(dim=2, keepdim=True).float()
    mask_x0 = (count_x == 0) | (count_x == 1)
    count_x[mask_x0] = torch.nan

    var_x = ((delta_x ** 2).nansum(dim=2, keepdim=True) / (count_x - 1))

    return var_x.reshape(-1, var_x.shape[1])


def ts_median(x1, window=10):
    """Rolling calculate the median value of the sequence for each token's past d day"""
    x = rolling_window(x=x1.clone(), window=window)

    median_x = x.nanmedian(dim=2, keepdim=True)[0]

    return median_x.reshape(-1, median_x.shape[1])


def ts_avgdev(x1, window=10):
    """Rolling calculate the average deviation value of the sequence for each token's past d day"""
    x = rolling_window(x=x1.clone(), window=window)

    mean_x = x.nanmean(dim=2, keepdim=True)  # E(x) # 忽略掉空值计算出来的，都是空值的行会是nan
    delta_x = x - mean_x
    avgdev = delta_x.nanmean(dim=2, keepdim=True) + 0.1

    return avgdev.reshape(-1, avgdev.shape[1])


def ts_skew(x1, window=10):
    """Rolling calculate the skewness of the sequence for each token's past d day：
    (d/(d-1)(d-2))*sum((delta/std)**3)"""
    x = rolling_window(x=x1.clone(), window=window)

    mean_x = x.nanmean(dim=2, keepdim=True)  # mean
    delta_x = x - mean_x  # x - mean

    mask_x = ~torch.isnan(delta_x)
    count_x = mask_x.sum(dim=2, keepdim=True).float()
    mask_x0 = (count_x == 0) | (count_x == 1)
    count_x[mask_x0] = torch.nan

    # std can potentially result in a null value when divided by 0, so we plus 1 here
    std_x = ((delta_x ** 2).nansum(dim=2, keepdim=True) / (count_x - 1)) ** 0.5 + 1

    ju_x = (delta_x / std_x) ** 3

    mask_x = ~torch.isnan(ju_x)
    count_x = mask_x.sum(dim=2, keepdim=True).float()
    mask_x0 = (count_x == 0)
    count_x[mask_x0] = torch.nan

    skew = ju_x.nansum(dim=2, keepdim=True) * count_x / ((count_x - 1) * (count_x - 2)) + 0.0001

    return skew.reshape(-1, skew.shape[1])


def ts_kurt(x1, window=10):
    """Rolling calculate the kurtosis of the sequence for each token's past d day：
    (d(d+1)/(d-1)(d-2)(d-3))*sum((delta/std)**4) - 3(d-1)**2/(d-2)(d-3)"""
    x = rolling_window(x=x1.clone(), window=window)

    mean_x = x.nanmean(dim=2, keepdim=True)  # mean
    delta_x = x - mean_x + 1  # x - mean

    mask_x = ~torch.isnan(delta_x)
    count_x = mask_x.sum(dim=2, keepdim=True).float()
    mask_x0 = (count_x == 0) | (count_x == 1)
    count_x[mask_x0] = torch.nan

    std_x = ((delta_x ** 2).nansum(dim=2, keepdim=True) / (count_x - 1)) ** 0.5 + 1  # std

    ju_x = (delta_x / std_x) ** 4

    mask_x = ~torch.isnan(ju_x)
    count_x = mask_x.sum(dim=2, keepdim=True).float()
    mask_x0 = (count_x == 0)
    count_x[mask_x0] = torch.nan

    kurt = ju_x.nansum(dim=2, keepdim=True) * (count_x * (count_x + 1)) / (
                (count_x - 1) * (count_x - 2) * (count_x - 3)) - 3 * (count_x - 1) ** 2 / (
                       (count_x - 2) * (count_x - 3))
    return kurt.reshape(-1, kurt.shape[1])


def ts_correlation(x1, x2, window=10):
    """Rolling calculate of the correlation coefficient between the window period d days of x1 and x2 """
    x = rolling_window(x=x1.clone(), window=window)
    y = rolling_window(x=x2.clone(), window=window)

    mean_x = x.nanmean(dim=2, keepdim=True)  # E(x)
    mean_y = y.nanmean(dim=2, keepdim=True)  # E(y)
    delta_x = x - mean_x  # x - E(x)
    delta_y = y - mean_y  # y - E(y)

    x_y = delta_x * delta_y  # [x - E(x)][y - E(y)]
    sum_x_y = x_y.nansum(dim=2, keepdim=True)

    mask_x_y = ~torch.isnan(x_y)
    count_x_y = mask_x_y.sum(dim=2, keepdim=True).float()
    mask_x_y0 = (count_x_y == 0) | (count_x_y == 1)
    count_x_y[mask_x_y0] = torch.nan

    cov = sum_x_y / (count_x_y - 1)  # E[[x - E(x)][y - E(y)]]
    std_x = ((delta_x ** 2).nansum(dim=2, keepdim=True) / (count_x_y - 1)) ** 0.5
    std_y = ((delta_y ** 2).nansum(dim=2, keepdim=True) / (count_x_y - 1)) ** 0.5

    corr = cov / (std_x * std_y + 0.1)

    return corr.reshape(-1, corr.shape[1])


def ts_covariance(x1, x2, window=10):
    """Rolling calculate of the covariance between the window period d days of x1 and x2"""
    x = rolling_window(x=x1.clone(), window=window)
    y = rolling_window(x=x2.clone(), window=window)

    mean_x = x.nanmean(dim=2, keepdim=True)  # E(x)
    mean_y = y.nanmean(dim=2, keepdim=True)  # E(y)
    delta_x = x - mean_x  # x - E(x)
    delta_y = y - mean_y  # y - E(y)

    x_y = delta_x * delta_y  # [x - E(x)][y - E(y)]
    sum_x_y = x_y.nansum(dim=2, keepdim=True)

    mask_x_y = ~torch.isnan(x_y)
    count_x_y = mask_x_y.sum(dim=2, keepdim=True).float()
    mask_x_y0 = (count_x_y == 0) | (count_x_y == 1)
    count_x_y[mask_x_y0] = torch.nan

    cov = sum_x_y / (count_x_y - 1)

    return cov.reshape(-1, cov.shape[1])


def ts_rank(x1):
    """The percentile ranking of each token on a time series for a factor"""
    x = x1.clone()

    mask = ~torch.isnan(x)
    count = mask.sum(dim=0, keepdim=True).float()
    mask1 = count == 0
    count[mask1] = torch.nan

    sorted_indices = torch.argsort(x, dim=0, descending=False)
    ranks = torch.argsort(sorted_indices, dim=0)
    ranks = ranks.float()
    ranks[x.isnan()] = torch.nan

    return ranks / (count - 1)


def ts_roll_rank(x1, window=10):
    """Calculate the rank of each currency on a rolling time series based on a certain feature"""
    x = rolling_window(x=x1.clone(), window=window)

    mask = ~torch.isnan(x)
    count = mask.sum(dim=2, keepdim=True).float()
    mask1 = count == 0
    count[mask1] = torch.nan

    rank_x = torch.argsort(torch.argsort(x, dim=2), dim=2)
    rank_x = rank_x.float()
    rank_x[x.isnan()] =torch.nan
    ranks = (rank_x / count)[:, :, -1].unsqueeze(-1)

    return ranks.reshape(-1, ranks.shape[1])


def ts_roll_zscore(x1, window=10):
    """Rolling calculate the zscore of the sequence for each token's past d day"""
    x = rolling_window(x=x1.clone(), window=window)

    mask = ~torch.isnan(x)
    count = mask.sum(dim=2, keepdim=True).float()
    mask1 = (count == 0) | (count == 1)
    count[mask1] = torch.nan

    mean_x = x.nanmean(dim=2, keepdim=True)  # E(x)
    delta_x = x - mean_x  # x - E(x)

    std_x = ((delta_x ** 2).nansum(dim=2, keepdim=True) / (count - 1)) ** 0.5 + 1

    zscore = (delta_x / std_x)[:, :, -1].unsqueeze(-1)

    return zscore.reshape(-1, zscore.shape[1])


def ts_delta(x1, window=10):
    """Subtract the value of the previous window day from the value of a certain day"""
    x = x1.clone()
    rows, cols = x1.shape

    nan_tensor = x.new_full((rows, cols), float('nan'))

    if rows >= window:
        nan_tensor[-rows + window:, :] = x[:-window, :]
    else:
        nan_tensor = nan_tensor
    delta_x = x - nan_tensor

    return delta_x


def ts_delay(x1, window=10):
    """The value of the previous window days on a certain day"""
    rows, cols = x1.shape

    nan_tensor = x1.new_full((rows, cols), float('nan'))

    if rows >= window:
        nan_tensor[-rows + window:, :] = x1[:-window, :]
    else:
        nan_tensor = nan_tensor

    return nan_tensor


def ts_pctchange(x1, window=10):
    """Percentage change in window period of time series"""
    x = x1.clone()
    return ts_delta(x, window) / ts_delay(x, window)


def ts_regbeta(x1, x2, window=10):
    """The slope of the linear regression between x1 and x2：sum([x - E(x)][y - E(y)]) / var(x)"""
    x11 = x1.clone()
    x22 = x2.clone()
    mask11 = torch.isnan(x11)
    mask22 = torch.isnan(x22)
    x11[mask11] = 1
    x22[mask22] = 1

    if torch.equal(x11[0:2][:30], x22[0:2][:30]):
        return ts_stddev(x1=x1, window=window)

    x = rolling_window(x=x1, window=window)
    y = rolling_window(x=x2, window=window)

    mean_x = x.nanmean(dim=2, keepdim=True)  # E(x)
    mean_y = y.nanmean(dim=2, keepdim=True)  # E(y)
    delta_x = x - mean_x  # x - E(x)
    delta_y = y - mean_y  # y - E(y)

    x_y = delta_x * delta_y  # [x - E(x)][y - E(y)]
    sum_x_y = x_y.nansum(dim=2, keepdim=True)  # sum([x - E(x)][y - E(y)])

    mask_x_y = ~torch.isnan(x_y)
    count_x_y = mask_x_y.sum(dim=2, keepdim=True).float()
    mask_x_y0 = (count_x_y == 0) | (count_x_y == 1)
    count_x_y[mask_x_y0] = torch.nan

    var_x = (delta_x ** 2).nansum(dim=2, keepdim=True) / (count_x_y - 1)

    beta = sum_x_y / var_x

    return beta.reshape(-1, beta.shape[1])


def ts_regalpha(x1, x2, window=10):
    """The intercept of the linear regression between x1 and x2：mean(y) - beta * mean(x)"""
    x = rolling_window(x=x1, window=window)
    y = rolling_window(x=x2, window=window)

    mean_x = x.nanmean(dim=2, keepdim=True)  # E(x)
    mean_y = y.nanmean(dim=2, keepdim=True)  # E(y)

    beta = ts_regbeta(x1=x1.clone(), x2=x2.clone(), window=window)
    beta = beta.unsqueeze(2)

    alpha = mean_y - beta * mean_x

    return alpha.reshape(-1, alpha.shape[1])


def ts_regres(x1, x2, window=10):
    """The residual of the linear regression between x1 and x2：y - beta * x - alpha"""
    x = rolling_window(x=x1, window=window)
    y = rolling_window(x=x2, window=window)

    beta = ts_regbeta(x1=x1.clone(), x2=x2.clone(), window=window)
    beta = beta.unsqueeze(2)
    alpha = ts_regalpha(x1=x1.clone(), x2=x2.clone(), window=window)
    alpha = alpha.unsqueeze(2)

    res = y[:, :, -1].unsqueeze(2) - beta * x[:, :, -1].unsqueeze(2) - alpha

    return res.reshape(-1, res.shape[1])


def mode_operation(x, x_cut, mode):
    """for cutting operator"""
    if mode == 1:
        x_cut_sum = x_cut.nansum(dim=2, keepdim=True)
        all_is_nan = torch.isnan(x_cut).all(dim=2, keepdim=True)
        x_cut_sum[all_is_nan] = torch.nan
        return x_cut_sum.reshape(-1, x_cut_sum.shape[1])
    elif mode == 2:
        x[~torch.isnan(x_cut)] *= -1
        x_sum = x.nansum(dim=2, keepdim=True)
        all_is_nan = torch.isnan(x).all(dim=2, keepdim=True)
        x_sum[all_is_nan] = torch.nan
        return x_sum.reshape(-1, x_sum.shape[1])


def ts_dec_sort_cut(x1, window=10, n=3, mode=1):
    """Cut out the largest n days within the rolling d-day window，and do mode"""
    x = rolling_window(x=x1, window=window)
    x_cut = x.clone()
    mask = torch.isnan(x)
    x[mask] = -1e10

    rank_x = torch.argsort(torch.argsort(x, dim=2, descending=True), dim=2).float() + 1
    rank_x[mask] = torch.nan

    mask0 = rank_x <= n
    rank_x[~mask0] = torch.nan
    mask_cut = ~torch.isnan(rank_x)

    x_cut = torch.where(mask_cut, x_cut, torch.full_like(x_cut, fill_value=torch.nan))
    x[mask] = torch.nan

    return mode_operation(x=x, x_cut=x_cut, mode=mode)


def ts_asc_sort_cut(x1, window=10, n=3, mode=1):
    """Cut out the smallest n days within the rolling d-day window，and do mode"""
    x = rolling_window(x=x1, window=window)
    mask = torch.isnan(x)

    rank_x = torch.argsort(torch.argsort(x, dim=2, descending=False), dim=2).float() + 1
    rank_x[mask] = torch.nan

    mask0 = rank_x <= n
    rank_x[~mask0] = torch.nan
    mask_cut = ~torch.isnan(rank_x)

    x_cut = x.clone()
    x_cut = torch.where(mask_cut, x_cut, torch.full_like(x_cut, fill_value=torch.nan))

    return mode_operation(x=x, x_cut=x_cut, mode=mode)


def ts_dec_zscore_cut(x1, window=10, a=1.5, mode=1):
    """Within the rolling d-day window, cut out the days with z values greater than a and then do mode.
    If none of the conditions are met, cut the day with maximum z values"""
    x_zscore = rolling_window(x=ts_roll_zscore(x1.clone(), window=window), window=window)
    x = rolling_window(x=x1, window=window)

    mask_x_zscore = torch.isnan(x_zscore)
    mean_x_zscore = x_zscore.nanmean(dim=2, keepdim=True)
    x_zscore[mask_x_zscore] = mean_x_zscore.repeat(1, 1, window)[mask_x_zscore]
    max_x_zscore, argmax_x_zscore = torch.max(x_zscore, dim=2, keepdim=True)
    mask_max = torch.zeros_like(x_zscore, dtype=torch.bool)
    mask_max.scatter_(2, argmax_x_zscore, True)

    mask = x_zscore > a
    mask = mask_max | mask
    mask_reverse = ~mask

    if mask_reverse.all() == True:
        return ts_max(x1.clone(), window=window)
    else:
        x_cut = x.clone()
        x_cut = torch.where(mask, x_cut, torch.full_like(x_cut, fill_value=torch.nan))

    return mode_operation(x=x, x_cut=x_cut, mode=mode)


def ts_asc_zscore_cut(x1, window=10, a=1.5, mode=1):
    """Within the rolling d-day window, cut out the days with z values smaller than a and then do mode.
    If none of the conditions are met, cut the day with minimum z values"""
    x_zscore = rolling_window(x=ts_roll_zscore(x1.clone(), window=window), window=window)
    x = rolling_window(x=x1, window=window)

    mask_x_zscore = torch.isnan(x_zscore)
    mean_x_zscore = x_zscore.nanmean(dim=2, keepdim=True)
    x_zscore[mask_x_zscore] = mean_x_zscore.repeat(1, 1, window)[mask_x_zscore]
    min_x_zscore, argmin_x_zscore = torch.min(x_zscore, dim=2, keepdim=True)
    mask_min = torch.zeros_like(x_zscore, dtype=torch.bool)
    mask_min.scatter_(2, argmin_x_zscore, True)

    mask = x_zscore < -a
    mask = mask_min | mask
    mask_reverse = ~mask

    if mask_reverse.all() == True:
        return ts_min(x1.clone(), window=window)
    else:
        x_cut = x.clone()
        x_cut = torch.where(mask, x_cut, torch.full_like(x_cut, fill_value=torch.nan))

    return mode_operation(x=x, x_cut=x_cut, mode=mode)


def ts_group_dec_sort_cut(x1, x2, window=10, n=3, mode=1):
    """Within the rolling d-day window, cut out x1 according to the maximum n-day x2 and perform a mode operation"""
    x = rolling_window(x=x1, window=window)
    y = rolling_window(x=x2, window=window)
    mask = torch.isnan(y)
    y[mask] = -1e10

    rank_y = torch.argsort(torch.argsort(y, dim=2, descending=True), dim=2).float() + 1
    rank_y[mask] = torch.nan

    mask0 = rank_y <= n
    rank_y[~mask0] = torch.nan
    mask_cut = ~torch.isnan(rank_y)

    x_cut = x.clone()
    x_cut = torch.where(mask_cut, x_cut, torch.full_like(x_cut, fill_value=torch.nan))

    return mode_operation(x=x, x_cut=x_cut, mode=mode)


def ts_group_asc_sort_cut(x1, x2, window=10, n=3, mode=1):
    """Within the rolling d-day window, cut out x1 according to the minimum n-day x2 and perform a mode operation"""
    x = rolling_window(x=x1, window=window)
    y = rolling_window(x=x2, window=window)
    mask = torch.isnan(y)

    rank_y = torch.argsort(torch.argsort(y, dim=2, descending=False), dim=2).float() + 1
    rank_y[mask] = torch.nan

    mask0 = rank_y <= n
    rank_y[~mask0] = torch.nan
    mask_cut = ~torch.isnan(rank_y)

    x_cut = x.clone()
    x_cut = torch.where(mask_cut, x_cut, torch.full_like(x_cut, fill_value=torch.nan))

    return mode_operation(x=x, x_cut=x_cut, mode=mode)


def ts_group_dec_zscore_cut(x1, x2, window=10, a=1.5, mode=1):
    """Within the rolling d-day window, cut x1 according to days where the z-value of x2 is greater than a.
    If none of the conditions are met, cut x1 according to days where the z-value of x2 is highest"""
    y_zscore = rolling_window(x=ts_roll_zscore(x2.clone(), window=window), window=window)
    x = rolling_window(x=x1, window=window)

    mask_y_zscore = torch.isnan(y_zscore)
    mean_y_zscore = y_zscore.nanmean(dim=2, keepdim=True)
    y_zscore[mask_y_zscore] = mean_y_zscore.repeat(1, 1, window)[mask_y_zscore]
    max_y_zscore, argmax_y_zscore = torch.max(y_zscore, dim=2, keepdim=True)
    mask_max = torch.zeros_like(y_zscore, dtype=torch.bool)
    mask_max.scatter_(2, argmax_y_zscore, True)

    mask = y_zscore > a
    mask = mask_max | mask

    mask_reverse = ~mask

    if mask_reverse.all() == True:
        return ts_max(x1.clone(), window=window)
    else:
        x_cut = x.clone()
        x_cut = torch.where(mask, x_cut, torch.full_like(x_cut, fill_value=torch.nan))

    return mode_operation(x=x, x_cut=x_cut, mode=mode)


def ts_group_asc_zscore_cut(x1, x2, window=10, a=1.5, mode=1):
    """Within the rolling d-day window, cut x1 according to days where the z-value of x2 is smaller than a.
    If none of the conditions are met, cut x1 according to days where the z-value of x2 is smallest"""
    y_zscore = rolling_window(x=ts_roll_zscore(x2.clone(), window=window), window=window)
    x = rolling_window(x=x1, window=window)

    mask_y_zscore = torch.isnan(y_zscore)
    mean_y_zscore = y_zscore.nanmean(dim=2, keepdim=True)
    y_zscore[mask_y_zscore] = mean_y_zscore.repeat(1, 1, window)[mask_y_zscore]
    min_y_zscore, argmin_y_zscore = torch.min(y_zscore, dim=2, keepdim=True)
    mask_min = torch.zeros_like(y_zscore, dtype=torch.bool)
    mask_min.scatter_(2, argmin_y_zscore, True)

    mask = y_zscore < -a
    mask = mask_min | mask

    mask_reverse = ~mask

    if mask_reverse.all() == True:
        return ts_min(x1.clone(), window=window)
    else:
        x_cut = x.clone()
        x_cut = torch.where(mask, x_cut, torch.full_like(x_cut, fill_value=torch.nan))

    return mode_operation(x=x, x_cut=x_cut, mode=mode)
