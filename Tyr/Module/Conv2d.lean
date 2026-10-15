import Tyr.Typed.Ops

namespace torch

structure Conv2d (dtype : DType := .Float32) (device : Device := .CPU) where
  in_channels : UInt64
  out_channels : UInt64
  kernel_size : UInt64 × UInt64
  stride : UInt64 × UInt64 := (1,1)
  padding : UInt64 × UInt64 := (0,0)
  dilation : UInt64 × UInt64 := (1,1)
  groups : UInt64 := 1
  bias : Bool := true

  weight : Tensor ⟨#[out_channels, in_channels/groups, kernel_size.fst, kernel_size.snd], dtype, device⟩ :=
    Tensor.zerosFor _
  outputBias : Tensor ⟨#[out_channels], dtype, device⟩ := Tensor.zerosFor _


/-- Initialize weights while preserving the configuration's tensor specification. -/
def Conv2d.init {d : DType} {dev : Device} (c : Conv2d d dev) : IO (Conv2d d dev) := do
  if c.groups == 0 || c.in_channels == 0 || c.out_channels == 0 then
    throw (IO.userError "Conv2d requires positive channel and group counts")
  if c.in_channels % c.groups != 0 || c.out_channels % c.groups != 0 then
    throw (IO.userError "Conv2d groups must divide both input and output channels")
  if c.kernel_size.fst == 0 || c.kernel_size.snd == 0 then
    throw (IO.userError "Conv2d requires positive kernel dimensions")
  let fanIn := (c.in_channels.toNat / c.groups.toNat) *
    c.kernel_size.fst.toNat * c.kernel_size.snd.toNat
  let k := 1.0 / Float.sqrt (Float.ofNat fanIn)
  let weight ← Tensor.uniformLike c.weight (-k) k
  return { c with weight }

def Conv2d.outShape {d : DType} {dev : Device} (c : Conv2d d dev) (s : Shape) : Shape :=
  let (n, _, hin, win) := (s[0]!, s[1]!, s[2]!, s[3]!);
  let (px,py) := c.padding;
  let (sx,sy) := c.stride;
  let (dx,dy) := c.dilation;
  let (kx,ky) := c.kernel_size;
  let hout := (hin + 2 * px - dx * (kx-1) - 1)/sx + 1;
  let wout := (win + 2 * py - dy * (ky-1) - 1)/sy + 1;
  #[n, c.out_channels, hout, wout]


end torch
