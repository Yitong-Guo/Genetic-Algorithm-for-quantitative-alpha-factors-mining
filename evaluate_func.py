import torch
import numpy as np


def check_for_nan(value):
    """
    Auxiliary function of evaluation function:
    If the evaluation function returns a null value, set it as 0
    """
    if np.isnan(value):
        return True
    return False


def evaluate(x, y, evaluation):
    """
    The order of the input evaluation function in the list is the order of the returned values;
    Can return single value or double value
    """
    assert x.shape == y.shape, "x and y must have the same shape"

    evaluation_result = [0] * len(evaluation)
    for func_num in range(len(evaluation)):
        if evaluation[func_num] == 'RankIC':
            rankic, _ = rankic_cal(x, y)
            evaluation_result[func_num] = rankic.item()
        elif evaluation[func_num] == 'RankICIR':
            _, rankicir = rankic_cal(x, y)
            evaluation_result[func_num] = rankicir.item()
        elif evaluation[func_num] == 'NDCG':
            ndcg = ndcg_cal(x, y, 10).item()
            evaluation_result[func_num] = ndcg
        elif evaluation[func_num] == 'Hedge_Return':
            hedge_return, _ = hedge_return_std_cal(x, y)
            evaluation_result[func_num] = hedge_return.item()
        elif evaluation[func_num] == 'Hedge_Return_Std':
            _, hedge_return_std = hedge_return_std_cal(x, y)
            evaluation_result[func_num] = hedge_return_std.item()
        else:
            raise ValueError("The function {} has not been accepted in this programme yet".format(evaluation[func_num]))

    for i in range(len(evaluation_result)):
        if check_for_nan(evaluation_result[i]):  # to check if is nan
            evaluation_result[i] = 0

    return evaluation_result


def rankic_cal(x, y):
    mask_x = torch.isnan(x)
    y[mask_x] = torch.nan
    mask_y = torch.isnan(y)
    x[mask_y] = torch.nan  # Align the nan positions of two tensors with each other

    rank_x = torch.argsort(torch.argsort(x, dim=1), dim=1)
    rank_y = torch.argsort(torch.argsort(y, dim=1), dim=1)

    rank_x = rank_x.float()
    rank_x[x.isnan()] = torch.nan
    rank_y = rank_y.float()
    rank_y[y.isnan()] = torch.nan

    mean_rank_x = rank_x.nanmean(dim=1, keepdim=True)
    mean_rank_y = rank_y.nanmean(dim=1, keepdim=True)

    delta_rank_x = rank_x - mean_rank_x
    delta_rank_y = rank_y - mean_rank_y

    # Calculate the covariance between two rows when ignoring nan values
    x_y = delta_rank_x * delta_rank_y  # [x-E(x)][y-E(y)]
    sum_x_y = x_y.nansum(dim=1, keepdim=True)

    # Find count_x_y, which is the number of non null values in each row of x_y.
    # As calculating E[[x-E(x)][y-E(y)] requires dividing by n-1, it is not a simple torch.nanmean
    mask_x_y = ~torch.isnan(x_y)
    count_x_y = mask_x_y.sum(dim=1, keepdim=True).float()
    mask_x_y0 = (count_x_y == 0) | (count_x_y == 1)
    count_x_y[mask_x_y0] = torch.nan

    cov = sum_x_y / (count_x_y - 1)

    std_x = ((delta_rank_x ** 2).nansum(dim=1, keepdim=True) / (count_x_y - 1)) ** 0.5
    std_y = ((delta_rank_y ** 2).nansum(dim=1, keepdim=True) / (count_x_y - 1)) ** 0.5

    rankic_series = cov / (std_x * std_y)
    rankic_series_mask = ~torch.isnan(rankic_series)
    rankic = rankic_series[rankic_series_mask].mean(dim=0)

    count_series_not_nan = rankic_series_mask.sum(dim=0).float()
    rankicir = (((rankic_series - rankic) ** 2).nansum(dim=0) / (count_series_not_nan - 1)) ** 0.5

    return abs(rankic),  abs(rankicir)


def hedge_return_std_cal(x, y):
    """According to factor ranking, hedge the top 20% of tokens against the bottom 20% of tokens"""
    mask_x = torch.isnan(x)
    y[mask_x] = torch.nan
    mask_y = torch.isnan(y)
    x[mask_y] = torch.nan

    def get_percent_data(x, y, mode):
        tensor1 = x.clone()
        tensor2 = y.clone()
        mask_na = x.isnan()
        if mode == 'up':
            tensor1[torch.isnan(tensor1)] = -1e15
            rank_t = torch.argsort(torch.argsort(tensor1, dim=1, descending=True), dim=1, descending=False)
        elif mode == 'down':
            tensor1[torch.isnan(tensor1)] = 1e15
            rank_t = torch.argsort(torch.argsort(tensor1, dim=1), dim=1)
        else:
            raise ValueError("mode in hedge return calculation must be 'up' or 'down' ")

        rank_t = rank_t.float()
        rank_t[mask_na] = torch.nan

        non_nan = ~torch.isnan(rank_t)
        num_non_nan = non_nan.sum(dim=1, keepdim=True)

        # Calculate the number of values in each row based on the proportion
        k = torch.clamp(torch.round(num_non_nan * 0.2).long(), min=1)
        k = k.expand_as(rank_t)

        mask = rank_t < k  # Make all items that do not meet the proportion empty

        tensor2[~mask] = torch.nan
        tensor = tensor2.nanmean(dim=1, keepdim=True)
        return tensor

    # Calculate the average value of the entire column divided by the standard deviation (ignoring NaN values)
    hedge_series = (get_percent_data(x, y, 'up') - get_percent_data(x, y, 'down')) / 2

    mask_hedge_series = ~torch.isnan(hedge_series)
    count_hedge = mask_hedge_series.sum(dim=0).float()  # for calculating standard deviation

    mean = hedge_series.nanmean(dim=0)
    mean_std = hedge_series.nanmean(dim=0) / \
               torch.sqrt(torch.nansum((hedge_series - torch.nanmean(hedge_series)) ** 2) / (count_hedge - 1))
    return abs(mean), abs(mean_std)


def power_cal(factor_values, future_returns, k):
    factor_values = torch.nan_to_num(factor_values, nan=-float('inf'))
    future_returns = torch.nan_to_num(future_returns, nan=-float('inf'))

    sorted_indices = factor_values.argsort(descending=True, dim=1)
    sorted_return_indices = future_returns.argsort(descending=True, dim=1)

    n = sorted_indices.size(1)

    group_size = n // 10

    labels = torch.arange(9, -1, -1).repeat_interleave(group_size)

    if labels.size(0) < n:
        labels = torch.cat([labels, torch.full((n - labels.size(0),), 0)])

    expanded_labels = labels.unsqueeze(0).expand_as(sorted_indices)

    ideal_tensor = torch.zeros_like(sorted_indices)
    ideal_tensor.scatter_(1, sorted_return_indices, expanded_labels)

    top_k_indices = sorted_indices[:, :k]
    ideal_indices = sorted_return_indices[:, :k]

    top_k_values = ideal_tensor.gather(1, top_k_indices)
    ideal_values = ideal_tensor.gather(1, ideal_indices)

    result = torch.pow(2, top_k_values)
    ideal = torch.pow(2, ideal_values)

    return result - 1, ideal - 1


def ndcg_cal(factor_values, future_returns, k):

    discounts = torch.log2(torch.arange(2, k + 2, device=factor_values.device).float())

    factor, future = power_cal(factor_values, future_returns, k)

    dcg = factor / discounts
    idcg = future / discounts
    ndcg = dcg / idcg

    return ndcg.mean()

