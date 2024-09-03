import torch as t


def xs_cutquartile_torch(x, alpha=1.5):
    """
    Outliers on the quartiles Faradaic cross-section
    alpha = 1.5, 2, 2.5, 3
    """
    Q1 = t.nanquantile(x, 0.25, dim=1, keepdim=True)
    Q3 = t.nanquantile(x, 0.75, dim=1, keepdim=True)
    IQR = Q3 - Q1

    lower_bound = Q1 - alpha * IQR
    upper_bound = Q3 + alpha * IQR

    capped_tensor = t.clamp(x, min=lower_bound, max=upper_bound)

    return capped_tensor


def xs_cutzscore_torch(x, alpha=1.5):
    """
    Z-score Farad returns outliers on the cross-section, ignoring NaN values.
    alpha = 1.5, 2, 2.5, 3
    """
    nan_mask = t.isnan(x)

    mean = t.nansum(x, dim=1) / t.sum(~nan_mask, dim=1)
    std = t.sqrt(t.nansum((x - mean.unsqueeze(1))**2, dim=1) / (t.sum(~nan_mask, dim=1) - 1))

    std[std == 0] = 1e-6

    z_scores = (x - mean.unsqueeze(1)) / std.unsqueeze(1)

    is_outlier = t.abs(z_scores) > alpha

    capped_tensor = t.where(nan_mask, t.nan, t.where(is_outlier, alpha * t.sign(z_scores), z_scores))
    org_tensor = capped_tensor * std.unsqueeze(-1) + mean.unsqueeze(-1)

    return org_tensor


def rank_pct_torch(x):
    """
    Quantile on the cross-section, ignore NaN value
    """
    nan_mask = t.isnan(x)

    x_no_nan = t.where(nan_mask, t.tensor(float('-inf'), device=x.device), x)

    _, indices = t.sort(x_no_nan, dim=1)

    nan_counts = nan_mask.sum(dim=1, keepdim=True)

    ranks = t.zeros_like(x, dtype=t.float32)

    ranks.scatter_(1, indices, t.arange(1, x.size(1) + 1, dtype=t.float32).repeat(x.size(0), 1))

    ranks[nan_mask] = float('nan')

    percentiles = (ranks - nan_counts) / (x.size(1) - nan_counts)

    return percentiles


def xs_regres_torch(x, y):
    """
    Regression on cross-section, ignoring NaN value
    """
    nan_mask_x = t.isnan(x)
    nan_mask_y = t.isnan(y)
    nan_mask = nan_mask_x | nan_mask_y

    result = t.full_like(y, float('nan'), dtype=t.float32)

    valid_mask = ~nan_mask

    for i in range(x.size(0)):
        valid_row_mask = valid_mask[i]

        if valid_row_mask.sum() >= 2:
            x_valid = x[i, valid_row_mask]
            y_valid = y[i, valid_row_mask]

            X_design = t.cat([x_valid.unsqueeze(-1), t.ones_like(x_valid).unsqueeze(-1)], dim=-1)
            X_design = X_design.float()
            y_valid = y_valid.float()

            coefficients, *_ = t.linalg.lstsq(X_design, y_valid, rcond=None)

            y_pred = X_design @ coefficients

            residuals = y_valid.squeeze() - y_pred.squeeze()

            result[i, valid_row_mask] = residuals

    return result


def xs_sortreverse_torch(x, rank=1, mode=0):
    """
    Multiply the corresponding value of the rank name before/after the x-section by -1
    rank = 1, 2, ..., 10
    mode = 0(up), 1(down), 2(both)
    """
    result_tensor = t.clone(x)

    for i in range(x.size(0)):
        time_slice = x[i, :]
        nan_num = t.sum(t.isnan(time_slice))
        time_slice = t.nan_to_num(time_slice, nan=float('inf'))
        sorted_indices = t.argsort(time_slice, descending=False)

        if mode == 0 or mode == 2:
            top_indices = sorted_indices[:rank]
            result_tensor[i, top_indices] *= -1

        if mode == 1 or mode == 2:
            bottom_indices = sorted_indices[-rank - nan_num:-nan_num]
            result_tensor[i, bottom_indices] *= -1

    return result_tensor


def xs_zscorereverse_torch(x, alpha=1.5, mode=0):
    """
    The product of the z-value of the x-section greater than/less than alpha multiplied by -1
    alpha = 1.5, 2, 2.5, 3
    mode = 0(up), 1(down), 2(both)
    """
    nan_mask = t.isnan(x)

    mean = t.nansum(x, dim=1) / t.sum(~nan_mask, dim=1)
    std = t.sqrt(t.nansum((x - mean.unsqueeze(1)) ** 2, dim=1) / (t.sum(~nan_mask, dim=1) - 1))

    std[std == 0] = 1e-6

    z_scores = (x - mean.unsqueeze(1)) / std.unsqueeze(1)

    large_outlier = z_scores > alpha
    small_outlier = z_scores < -alpha

    result_tensor = x.clone()
    result_tensor[large_outlier] *= -1 if mode == 0 or mode == 2 else 1
    result_tensor[small_outlier] *= -1 if mode == 1 or mode == 2 else 1

    return result_tensor


def xs_grouping_sortreverse_torch(x, y, rank=1, mode=0):
    """
    The x multiplied by -1 corresponding to the rank names before/after sorting the y-section grouping
    rank = 1, 2, ..., 10
    mode = 0(up), 1(down), 2(both)
    """
    result_tensor = t.clone(x)

    for i in range(x.size(0)):
        time_slice = y[i, :]
        nan_num = t.sum(t.isnan(time_slice)).item()
        time_slice = t.nan_to_num(time_slice, nan=float('inf'))
        sorted_indices = t.argsort(time_slice, descending=False)

        if mode == 0 or mode == 2:
            top_indices = sorted_indices[:rank]
            result_tensor[i, top_indices] *= -1

        if mode == 1 or mode == 2:
            if nan_num != 0:
                bottom_indices = sorted_indices[-rank - nan_num:-nan_num]
            else:
                bottom_indices = sorted_indices[-rank:]
            result_tensor[i, bottom_indices] *= -1

    return result_tensor


def xs_grouping_zscorereverse_torch(x, y, alpha=1.5, mode=0):
    """
    The x multiplied by -1 corresponding to the rank names before/after sorting the y_zscore section grouping
    alpha = 1.5, 2, 2.5, 3
    mode = 0(up), 1(down), 2(both)
    """
    nan_mask = t.isnan(y)

    mean = t.nansum(y, dim=1) / t.sum(~nan_mask, dim=1)
    std = t.sqrt(t.nansum((y - mean.unsqueeze(1)) ** 2, dim=1) / (t.sum(~nan_mask, dim=1) - 1))

    std[std == 0] = 1e-6

    z_scores = (y - mean.unsqueeze(1)) / std.unsqueeze(1)

    large_outlier = z_scores > alpha
    small_outlier = z_scores < -alpha

    result_tensor = x.clone()
    result_tensor[large_outlier] *= -1 if mode == 0 or mode == 2 else 1
    result_tensor[small_outlier] *= -1 if mode == 1 or mode == 2 else 1

    return result_tensor


if __name__ == '__main__':
    test_tensor = t.tensor([[1., float('nan'), 3., float('nan'), 5.], [1., 1., float('nan'), 1., 10.]])
    y_tensor = t.tensor([[2., 2., 10., 4., 6.], [5., 5., 8., 9., 9.]])

    print(xs_regres_torch(y_tensor, test_tensor))
