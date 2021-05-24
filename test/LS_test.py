import torch

k = 3  # Num. classes
memory = torch.zeros(k, k)
idx_count = torch.zeros(k)

y = torch.tensor([[1], [0], [0], [1]], dtype=torch.long)

y_h = torch.tensor([
		[0.3, 0.6, 0.1],  # Correct
		[0.7, 0.2, 0.1],  # Correct
		[0.3, 0.3, 0.4],
		[0.2, 0.7, 0.1]   # Correct
	])

'''
1. Calculate correct classified examples
2. Filter `y_h` based on the correct classified
3. Add `y_h_f` rows to the `j` (based on y_h_idx) column of `memory`
4. Keep count of # samples added for each `y_h_idx` column
5. Average memory by dividing column-wise by result of step (4)
'''

# 1. Calculate predicted classes
y_h_idx = y_h.argmax(dim=-1)  # tensor([1, 0, 2, 1])
# 2. Filter only correct
mask = torch.eq(y_h_idx, y.squeeze(dim=-1))
y_h_c = y_h[mask]
y_h_idx_c = y_h_idx[mask]  # tensor([1, 0, 1])
# 3. Add y_h probabilities rows as columns to `memory` 
memory.index_add_(1, y_h_idx_c, y_h_c.T)
# 4. Update `idx_count`
idx_count.index_add_(0, y_h_idx_c, torch.ones_like(y_h_idx_c, dtype=torch.float32))
# 5. Divide memory by `idx_count` to obtain average (column-wise)
idx_count[torch.eq(idx_count, 0)] = 1  # Avoid 0 denominator
memory /= idx_count.unsqueeze(dim=-1)

# Expected result:
# [
# 	[0.7, 0.25, 0.00],
# 	[0.2, 0.65, 0.00],
# 	[0.1, 0.10, 0.00]
# ]
