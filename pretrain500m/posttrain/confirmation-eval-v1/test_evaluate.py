import torch
from evaluate import DOMAINS, masked_domain_sums, summarize


def test_masked_domain_accounting_and_summary():
    logits = torch.zeros(2, 2, 3)
    targets = torch.tensor([[0, 1], [2, 1]])
    masks = torch.tensor([[True, False], [True, True]])
    sums, counts = masked_domain_sums(logits, targets, masks, ('general_web', 'math'))
    assert counts[DOMAINS.index('general_web')].item() == 1
    assert counts[DOMAINS.index('math')].item() == 2
    assert sums[DOMAINS.index('general_web')].item() > 0


def test_summary_equal_domain_mean():
    sums = torch.tensor([2., 4., 6., 8., 10.], dtype=torch.float64)
    counts = torch.tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    result = summarize(sums, counts)
    assert result['loss_equal_domain'] == 3.0
    assert result['target_tokens_by_domain']['code'] == 2
