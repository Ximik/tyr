import Tyr.Distributed

/-!
# Tyr.Sharding

Size and offset of one shard when a dimension is split across ranks, the first
`fullSize % numShards` shards taking one extra element.
-/

namespace torch.Sharding

open torch

/-! ## Sharding Specification -/

/-- Compute the shard size for a given full size -/
def shardSize (fullSize numShards shardIdx : Nat) : Nat :=
  let baseSize := fullSize / numShards
  let remainder := fullSize % numShards
  if shardIdx < remainder then baseSize + 1 else baseSize

/-- Compute shard offset (starting index) -/
def shardOffset (fullSize numShards shardIdx : Nat) : Nat :=
  let baseSize := fullSize / numShards
  let remainder := fullSize % numShards
  -- First `remainder` shards get one extra element
  if shardIdx < remainder then
    shardIdx * (baseSize + 1)
  else
    remainder * (baseSize + 1) + (shardIdx - remainder) * baseSize

end torch.Sharding
