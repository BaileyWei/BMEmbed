import torch
from torch import nn, Tensor
from accelerate.logging import get_logger


logger = get_logger(__name__)


class AllGather(torch.autograd.Function):

    """
    all_gather with gradient back-propagation
    """

    @staticmethod
    def forward(ctx, tensor_list, tensor, group, async_op):
        torch.distributed.all_gather(tensor_list, tensor, group=group, async_op=async_op)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = torch.distributed.get_rank()

        dist_ops = [
            torch.distributed.reduce(grad_list[i], i, async_op=True) for i in range(torch.distributed.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank], None, None


all_gather_with_grad = AllGather.apply


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def listMLE(y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))

def mismatched_sizes_all_gather(tensor: Tensor, group=None, async_op=False, mismatched_axis=0):
    # all_gather doesn't support tensor lists where the first dimension is mismatched. This does.
    assert torch.distributed.is_initialized(), "torch.distributed not initialized"
    world_size = torch.distributed.get_world_size()
    # let's get the sizes for everyone
    mismatched_sizes = torch.tensor([tensor.shape[mismatched_axis]], dtype=torch.int64, device="cuda")
    sizes = [torch.zeros_like(mismatched_sizes) for _ in range(world_size)]
    # logger.info(sizes)
    torch.distributed.all_gather(sizes, mismatched_sizes, group=group, async_op=async_op)
    sizes = torch.cat(sizes).cpu().tolist()
    # now pad to the max dim-0 size
    max_size = max(sizes)
    # logger.info(max_size)
    padded = torch.zeros((*tensor.shape[:mismatched_axis], max_size, *tensor.shape[mismatched_axis + 1:]),
                         device=tensor.device, dtype=tensor.dtype)
    # selects the place where we're adding information
    padded_to_fill = padded.narrow(mismatched_axis, 0, tensor.shape[mismatched_axis])
    padded_to_fill[...] = tensor
    # gather the padded tensors
    tensor_list = [torch.zeros(padded.shape, device=padded.device, dtype=padded.dtype) for _ in range(world_size)]
    all_gather_with_grad(tensor_list, padded, group, async_op)
    # trim off the padding
    for rank in range(world_size):
        # checks that the rest is 0
        assert not tensor_list[rank].narrow(mismatched_axis, sizes[rank],
                                            padded.shape[mismatched_axis] - sizes[rank]).count_nonzero().is_nonzero(), \
            "This would remove non-padding information"
        tensor_list[rank] = tensor_list[rank].narrow(mismatched_axis, 0, sizes[rank])
    return tensor_list

class HardNegativeNLLLoss():
    def __init__(
            self,
            scale: float = 20.0,
            similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
            self,
            q_reps: Tensor,
            d_reps_pos: Tensor,
            d_reps_neg: Tensor = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)

        scores = self.similarity_fct(full_q_reps, d_reps) * self.scale
        labels = (
            torch.tensor(
                range(len(scores)), dtype=torch.long, device=scores.device
            )
        )

        loss = self.cross_entropy_loss(scores, labels)
        return loss


class ListwiseLoss():
    def __init__(
            self,
            scale: float = 20.0,
            label_scaling: float = 1.0,
            similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.label_scaling = label_scaling
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
            self,
            q_reps: Tensor,
            d_reps_pos: Tensor,
            d_reps_neg: Tensor = None,
            labels: Tensor = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        # if torch.distributed.is_initialized():
        #     full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
        #     full_d_reps_pos = torch.cat(full_d_reps_pos)
        #
        #     full_q_reps = mismatched_sizes_all_gather(q_reps)
        #     full_q_reps = torch.cat(full_q_reps)
        #
        #     full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
        #     full_d_reps_neg = torch.cat(full_d_reps_neg)
        # else:
        full_d_reps_pos = d_reps_pos
        full_q_reps = q_reps
        full_d_reps_neg = d_reps_neg

        # d_reps = torch.stack((full_d_reps_pos, full_d_reps_neg), dim=1)
        num_neg = int(len(full_d_reps_neg) / len(full_q_reps))
        d_reps = torch.stack(
            [torch.cat([full_d_reps_pos[i:i + 1], full_d_reps_neg[i * num_neg: i * num_neg + num_neg]], dim=0) for i in
             range(len(full_q_reps))])
        assert d_reps.shape[0] == len(full_q_reps), d_reps.shape[1] == num_neg + 1
        scores = (full_q_reps.reshape(len(full_q_reps), 1, -1) @ d_reps.transpose(1, 2)) * self.scale
        scores_prob = torch.softmax(scores.squeeze(1), dim=1)
        # d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        # scores = self.similarity_fct(full_q_reps, d_reps) * self.scale
        if labels is None:
            labels = (
                torch.tensor([0] * len(scores), dtype=torch.long, device=scores.device
                             )
            )
        else:
            labels = torch.softmax(labels * self.label_scaling, dtype=torch.bfloat16, dim=1)

        loss = -torch.sum(labels * torch.log(scores_prob), dim=1).mean()

        # loss = self.cross_entropy_loss(scores, labels)
        return loss


class ListMLELoss():
    def __init__(
            self,
            scale: float = 20.0,
            label_scaling: float = 1.0,
            similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.label_scaling = label_scaling
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
            self,
            q_reps: Tensor,
            d_reps_pos: Tensor,
            d_reps_neg: Tensor = None,
            labels: Tensor = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        # if torch.distributed.is_initialized():
        #     full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
        #     full_d_reps_pos = torch.cat(full_d_reps_pos)
        #
        #     full_q_reps = mismatched_sizes_all_gather(q_reps)
        #     full_q_reps = torch.cat(full_q_reps)
        #
        #     full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
        #     full_d_reps_neg = torch.cat(full_d_reps_neg)
        # else:
        full_d_reps_pos = d_reps_pos
        full_q_reps = q_reps
        full_d_reps_neg = d_reps_neg

        # d_reps = torch.stack((full_d_reps_pos, full_d_reps_neg), dim=1)
        num_neg = int(len(full_d_reps_neg) / len(full_q_reps))
        d_reps = torch.stack(
            [torch.cat([full_d_reps_pos[i:i + 1], full_d_reps_neg[i * num_neg: i * num_neg + num_neg]], dim=0) for i in
             range(len(full_q_reps))])
        assert d_reps.shape[0] == len(full_q_reps), d_reps.shape[1] == num_neg + 1
        scores = (full_q_reps.reshape(len(full_q_reps), 1, -1) @ d_reps.transpose(1, 2)) * self.scale
        # scores_prob = torch.softmax(scores.squeeze(1), dim=1)
        # d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        # scores = self.similarity_fct(full_q_reps, d_reps) * self.scale
        # if labels is None:
        #     labels = (
        #         torch.tensor([0] * len(scores), dtype=torch.long, device=scores.device
        #                      )
        #     )
        # else:
        #     labels = torch.softmax(labels * self.label_scaling, dtype=torch.bfloat16, dim=1)

        loss = listMLE(scores.squeeze(1),labels*self.label_scaling)

        return loss