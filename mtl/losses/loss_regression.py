import torch


class LossRegression(torch.nn.Module):
    @staticmethod
    def forward_one_image(y_hat, y):
        valid_mask = y == y  # filter out NaN values
        y_hat = y_hat[valid_mask]
        y = y[valid_mask]

        if y.numel() == 0:
            return None, False

        # L1 penalty
        #loss = (y_hat - y).abs().mean()
        # L2 penalty
        loss = (y_hat - y).pow(2).mean()

        return loss, True

    def forward(self, y_hat, y):
        assert torch.is_tensor(y_hat) and torch.is_tensor(y)
        assert y.dim() == 3 or y.dim() == 4 and y.shape[1] == 1
        if y.dim() == 4:
            y = y.squeeze(1)
        assert y_hat.dim() == 3 or y_hat.dim() == 4 and y_hat.shape[1] == 1
        if y_hat.dim() == 4:
            y_hat = y_hat.squeeze(1)
        assert y_hat.shape == y.shape
        N, H, W = y_hat.shape

        out_loss = torch.tensor(0, device=y.device, dtype=y.dtype, requires_grad=True)
        out_cnt = 0
        for i in range(N):
            loss, have_loss = self.forward_one_image(y_hat[i], y[i])
            if have_loss:
                out_loss = out_loss + loss
                out_cnt += 1

        return out_loss / max(out_cnt, 1)
