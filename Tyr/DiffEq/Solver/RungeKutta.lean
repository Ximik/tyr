import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## Explicit Runge–Kutta Infrastructure -/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

@[reducible] private def hDivRightScalarInst [DiffEqSpace Y] : HDiv Y Scalar Y where
  hDiv y a := (1.0 / a) * y

attribute [local instance] hDivRightScalarInst

structure ButcherTableau (s : Nat) where
  a : Vector s (Array Time)
  b : Vector s Time
  c : Vector s Time
  bErr : Option (Vector s Time) := none
  order : Nat := 1

inductive ExplicitRKDenseKind where
  | hermite
  | splitAtStage (stage : Nat) (tag : String)
  | dopri5Poly4
  | tsit5Poly4
  deriving Repr, BEq, Inhabited

structure ExplicitRK (s : Nat) where
  tableau : ButcherTableau s
  denseKind : ExplicitRKDenseKind := .hermite

namespace ExplicitRK

inductive DenseInfo (Y : Type) where
  | hermite (info : LocalHermiteDenseInfo Y)
  | poly4 (info : LocalPolynomial4DenseInfo Y)

namespace DenseInfo

def toInterpolation [DiffEqSpace Y] (info : DenseInfo Y) : DenseInterpolation Y :=
  match info with
  | .hermite info => info.toInterpolation
  | .poly4 info => info.toInterpolation

end DenseInfo

private def stageState {s : Nat} [DiffEqSpace Y] (rk : ExplicitRK s)
    (ks : Array Y) (y0 : Y) (stage : Nat) : Y := Id.run do
  let zero := zeroLike y0
  let rows := rk.tableau.a.toArray
  let row := rows.getD stage #[]
  let mut sum := zero
  for j in [:stage] do
    let aij := row.getD j 0.0
    let kj := ks.getD j zero
    sum := sum + aij * kj
  return y0 + sum

private def baseHermiteDenseInfo [DiffEqSpace Y]
    (t0 t1 : Time) (y0 y1 : Y) (m0 m1 : Y) : LocalHermiteDenseInfo Y := {
  t0 := t0
  t1 := t1
  y0 := y0
  y1 := y1
  m0 := m0
  m1 := m1
}

private def dopri5Poly4DenseInfo [DiffEqSpace Y]
    (t0 t1 : Time) (y0 y1 : Y) (ks : Array Y) (fallback : LocalHermiteDenseInfo Y) :
    DenseInfo Y :=
  if ks.size == 7 then
    let cMid : Array Time := #[
      (6025192743.0 / 30085553152.0) / 2.0,
      0.0,
      (51252292925.0 / 65400821598.0) / 2.0,
      (-2691868925.0 / 45128329728.0) / 2.0,
      (187940372067.0 / 1594534317056.0) / 2.0,
      (-1776094331.0 / 19743644256.0) / 2.0,
      (11237099.0 / 235043384.0) / 2.0
    ]
    let zero := zeroLike y0
    let yMid := y0 + weightedSumArray cMid ks y0
    let f0 := ks.getD 0 zero
    let f1 := ks.getD 6 zero
    -- Quartic polynomial in normalized step time theta.
    let c4 := 2.0 * (f1 - f0) - 8.0 * (y1 + y0) + 16.0 * yMid
    let c3 := 5.0 * f0 - 3.0 * f1 + 18.0 * y0 + 14.0 * y1 - 32.0 * yMid
    let c2 := f1 - 4.0 * f0 - 11.0 * y0 - 5.0 * y1 + 16.0 * yMid
    .poly4 {
      t0 := t0
      t1 := t1
      c4 := c4
      c3 := c3
      c2 := c2
      c1 := f0
      c0 := y0
    }
  else
    .hermite fallback

private def tsit5Poly4DenseInfo [DiffEqSpace Y]
    (t0 t1 : Time) (y0 : Y) (ks : Array Y) (fallback : LocalHermiteDenseInfo Y) :
    DenseInfo Y :=
  if ks.size == 7 then
    -- Diffrax Tsit5 dense interpolation, rewritten into quartic power basis.
    let c4Weights : Array Time := #[
      -1.0530884977290216,
      0.1017,
      2.490627285651252793,
      -16.54810288924490272,
      47.37952196281928122,
      -34.87065786149660974,
      2.5
    ]
    let c3Weights : Array Time := #[
      2.91325546182191264,
      -0.2234,
      -5.9410338721315048,
      30.33818863028232116,
      -88.17890489476640425,
      65.09189467479367863,
      -4.0
    ]
    let c2Weights : Array Time := #[
      -2.7637061972748258,
      0.1317,
      3.93029623689475116,
      -12.41107716693367635,
      37.50931341651104134,
      -27.89652628919728627,
      1.5
    ]
    let c1Weights : Array Time := #[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    .poly4 {
      t0 := t0
      t1 := t1
      c4 := weightedSumArray c4Weights ks y0
      c3 := weightedSumArray c3Weights ks y0
      c2 := weightedSumArray c2Weights ks y0
      c1 := weightedSumArray c1Weights ks y0
      c0 := y0
    }
  else
    .hermite fallback

private def denseInfo {s : Nat} [DiffEqSpace Y] (rk : ExplicitRK s)
    (t0 t1 : Time) (y0 y1 : Y) (dt : Time) (ks : Array Y) (m0 m1 : Y) :
    DenseInfo Y :=
  let base := baseHermiteDenseInfo t0 t1 y0 y1 m0 m1
  match rk.denseKind with
  | .hermite => .hermite base
  | .splitAtStage stage tag =>
      let cs := rk.tableau.c.toArray
      if hStage : stage < cs.size then
        let c := cs[stage]'hStage
        if c <= 0.0 || c >= 1.0 then
          .hermite base
        else
          let zero := zeroLike y0
          let tSplit := t0 + c * dt
          let ySplit := stageState rk ks y0 stage
          let mSplit := ks.getD stage zero
          .hermite {
            base with split? := some (tSplit, ySplit, mSplit), splitKind? := some tag
          }
      else
        .hermite base
  | .dopri5Poly4 =>
      dopri5Poly4DenseInfo t0 t1 y0 y1 ks base
  | .tsit5Poly4 =>
      tsit5Poly4DenseInfo t0 t1 y0 ks base

def solver {s : Nat} {Term Y VF Args : Type}
    [TermLike Term Y VF Time Args]
    [DiffEqSpace Y] (rk : ExplicitRK s) : AbstractSolver Term Y VF Time Args := {
  SolverState := Unit
  DenseInfo := DenseInfo Y
  termStructure := TermStructure.single
  odeStepAdjoint? := some (.explicitRK {
    a := rk.tableau.a.toArray
    b := rk.tableau.b.toArray
    c := rk.tableau.c.toArray
  })
  order := fun _ => rk.tableau.order
  strongOrder := fun _ => 0.0
  init := fun _ _ _ _ _ => ()
  step := fun term t0 t1 y0 args state _madeJump =>
    let inst := (inferInstance : TermLike Term Y VF Time Args)
    let dt := inst.contr term t0 t1
    let zero := zeroLike y0
    let ks := Id.run do
      let mut ks : Array Y := #[]
      let rows := rk.tableau.a.toArray
      let cs := rk.tableau.c.toArray
      for i in [:rows.size] do
        let row := rows.getD i #[]
        let mut sum := zero
        for j in [:i] do
          let aij := row.getD j 0.0
          let kj := ks.getD j zero
          sum := sum + aij * kj
        let ti := t0 + cs.getD i 0.0 * dt
        let yi := y0 + sum
        let ki := inst.vf_prod term ti yi args dt
        ks := ks.push ki
      return ks
    let y1 := y0 + weightedSum rk.tableau.b ks y0
    let yErr :=
      match rk.tableau.bErr with
      | none => none
      | some bErr =>
          let high := weightedSum rk.tableau.b ks y0
          let low := weightedSum bErr ks y0
          some (high - low)
    -- dt-scaled endpoint tangents used by cubic Hermite dense output.
    let m0 := ks.getD 0 zero
    let m1 := inst.vf_prod term t1 y1 args dt
    let dense := denseInfo rk t0 t1 y0 y1 dt ks m0 m1
    {
      y1 := y1
      yError := yErr
      denseInfo := dense
      solverState := state
      result := Result.successful
    }
  func := fun term t y args =>
    let inst := (inferInstance : TermLike Term Y VF Time Args)
    inst.vf term t y args
  interpolation := fun info => info.toInterpolation
}

end ExplicitRK

def vec2 {α : Type} (a b : α) : Vector 2 α := ⟨#[a, b], by simp⟩
def vec3 {α : Type} (a b c : α) : Vector 3 α := ⟨#[a, b, c], by simp⟩
def vec4 {α : Type} (a b c d : α) : Vector 4 α := ⟨#[a, b, c, d], by simp⟩
def vec5 {α : Type} (a b c d e : α) : Vector 5 α := ⟨#[a, b, c, d, e], by simp⟩
def vec6 {α : Type} (a b c d e f : α) : Vector 6 α := ⟨#[a, b, c, d, e, f], by simp⟩
def vec7 {α : Type} (a b c d e f g : α) : Vector 7 α := ⟨#[a, b, c, d, e, f, g], by simp⟩
def vec8 {α : Type} (a b c d e f g h : α) : Vector 8 α :=
  ⟨#[a, b, c, d, e, f, g, h], by simp⟩
def vec9 {α : Type} (a b c d e f g h i : α) : Vector 9 α :=
  ⟨#[a, b, c, d, e, f, g, h, i], by simp⟩
def vec10 {α : Type} (a b c d e f g h i j : α) : Vector 10 α :=
  ⟨#[a, b, c, d, e, f, g, h, i, j], by simp⟩
def vec11 {α : Type} (a b c d e f g h i j k : α) : Vector 11 α :=
  ⟨#[a, b, c, d, e, f, g, h, i, j, k], by simp⟩
def vec12 {α : Type} (a b c d e f g h i j k l : α) : Vector 12 α :=
  ⟨#[a, b, c, d, e, f, g, h, i, j, k, l], by simp⟩
def vec13 {α : Type} (a b c d e f g h i j k l m : α) : Vector 13 α :=
  ⟨#[a, b, c, d, e, f, g, h, i, j, k, l, m], by simp⟩
def vec14 {α : Type} (a b c d e f g h i j k l m n : α) : Vector 14 α :=
  ⟨#[a, b, c, d, e, f, g, h, i, j, k, l, m, n], by simp⟩

private def vecSub {n : Nat} (a b : Vector n Time) : Vector n Time :=
  Vector.zipWith (· - ·) a b

/-- Ralston's 2nd-order method. -/
def ralstonTableau : ButcherTableau 2 := {
  a := vec2 #[0.0, 0.0] #[2.0 / 3.0, 0.0]
  b := vec2 (1.0 / 4.0) (3.0 / 4.0)
  c := vec2 0.0 (2.0 / 3.0)
  bErr := none
  order := 2
}

/-- Bogacki–Shampine 3(2) method (Bosh3). -/
def bosh3Tableau : ButcherTableau 4 := {
  a := vec4
    #[0.0, 0.0, 0.0, 0.0]
    #[1.0 / 2.0, 0.0, 0.0, 0.0]
    #[0.0, 3.0 / 4.0, 0.0, 0.0]
    #[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0]
  b := vec4 (2.0 / 9.0) (1.0 / 3.0) (4.0 / 9.0) 0.0
  c := vec4 0.0 (1.0 / 2.0) (3.0 / 4.0) 1.0
  bErr := some (vec4 (7.0 / 24.0) (1.0 / 4.0) (1.0 / 3.0) (1.0 / 8.0))
  order := 3
}

/-- Dormand–Prince 5(4) method (Dopri5). -/
def dopri5Tableau : ButcherTableau 7 := {
  a := vec7
    #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #[1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #[3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0]
    #[19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,
      0.0, 0.0, 0.0]
    #[9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
      -5103.0 / 18656.0, 0.0, 0.0]
    #[35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
      11.0 / 84.0, 0.0]
  b := vec7 (35.0 / 384.0) 0.0 (500.0 / 1113.0) (125.0 / 192.0)
    (-2187.0 / 6784.0) (11.0 / 84.0) 0.0
  c := vec7 0.0 (1.0 / 5.0) (3.0 / 10.0) (4.0 / 5.0) (8.0 / 9.0) 1.0 1.0
  bErr := some (vec7 (1951.0 / 21600.0) 0.0 (22642.0 / 50085.0) (451.0 / 720.0)
    (-12231.0 / 42400.0) (649.0 / 6300.0) (1.0 / 60.0))
  order := 5
}

/-- Classic 4th-order Runge–Kutta method (RK4). -/
def rk4Tableau : ButcherTableau 4 := {
  a := vec4
    #[0.0, 0.0, 0.0, 0.0]
    #[1.0 / 2.0, 0.0, 0.0, 0.0]
    #[0.0, 1.0 / 2.0, 0.0, 0.0]
    #[0.0, 0.0, 1.0, 0.0]
  b := vec4 (1.0 / 6.0) (1.0 / 3.0) (1.0 / 3.0) (1.0 / 6.0)
  c := vec4 0.0 (1.0 / 2.0) (1.0 / 2.0) 1.0
  bErr := none
  order := 4
}

/-- Tsitouras' 5/4 method (Tsit5). -/
def tsit5Tableau : ButcherTableau 7 := by
  let bSol := vec7
    0.09646076681806523
    0.01
    0.4798896504144996
    1.3790085741037419
    (-3.2900695154360807)
    2.324710524099774
    0.0
  let bErr := vec7
    (0.09646076681806523 - 0.09468075576583946)
    (0.01 - 0.009183565540343253)
    (0.4798896504144996 - 0.48777052842476157)
    (1.3790085741037419 - 1.234297566930479)
    (-3.2900695154360807 - -2.7077123499835255)
    (2.324710524099774 - 1.866628418170587)
    (-1.0 / 66.0)
  exact {
    a := vec7
      #[]
      #[161.0 / 1000.0]
      #[-0.008480655492356989, 0.33548065549235697]
      #[2.8971530571054935, -6.359448489975075, 4.362295432869581]
      #[5.325864828439257, -11.748883564062828, 7.495539342889836, -0.09249506636175525]
      #[5.86145544294642, -12.92096931784711, 8.159367898576158, -0.071584973281401, -0.028269050394068383]
      #[0.09646076681806523, 0.01, 0.4798896504144996, 1.3790085741037419,
        -3.2900695154360807, 2.324710524099774]
    b := bSol
    c := vec7 0.0 0.161 0.327 0.9 0.9800255409045097 1.0 1.0
    bErr := some (vecSub bSol bErr)
    order := 5
  }

/-- Dormand–Prince 8(7) method (Dopri8). -/
def dopri8Tableau : ButcherTableau 14 := by
  let bSol := vec14
    (14005451.0 / 335480064.0)
    0.0
    0.0
    0.0
    0.0
    (-59238493.0 / 1068277825.0)
    (181606767.0 / 758867731.0)
    (561292985.0 / 797845732.0)
    (-1041891430.0 / 1371343529.0)
    (760417239.0 / 1151165299.0)
    (118820643.0 / 751138087.0)
    (-528747749.0 / 2220607170.0)
    0.25
    0.0
  let bErr := vec14
    ((14005451.0 / 335480064.0) - (13451932.0 / 455176623.0))
    0.0
    0.0
    0.0
    0.0
    ((-59238493.0 / 1068277825.0) - (-808719846.0 / 976000145.0))
    ((181606767.0 / 758867731.0) - (1757004468.0 / 5645159321.0))
    ((561292985.0 / 797845732.0) - (656045339.0 / 265891186.0))
    ((-1041891430.0 / 1371343529.0) - (-3867574721.0 / 1518517206.0))
    ((760417239.0 / 1151165299.0) - (465885868.0 / 322736535.0))
    ((118820643.0 / 751138087.0) - (53011238.0 / 667516719.0))
    ((-528747749.0 / 2220607170.0) - (2.0 / 45.0))
    0.25
    0.0
  exact {
    a := vec14
      #[]
      #[1.0 / 18.0]
      #[1.0 / 48.0, 1.0 / 16.0]
      #[1.0 / 32.0, 0.0, 3.0 / 32.0]
      #[5.0 / 16.0, 0.0, -75.0 / 64.0, 75.0 / 64.0]
      #[3.0 / 80.0, 0.0, 0.0, 3.0 / 16.0, 3.0 / 20.0]
      #[29443841.0 / 614563906.0, 0.0, 0.0, 77736538.0 / 692538347.0,
        -28693883.0 / 1125000000.0, 23124283.0 / 1800000000.0]
      #[16016141.0 / 946692911.0, 0.0, 0.0, 61564180.0 / 158732637.0,
        22789713.0 / 633445777.0, 545815736.0 / 2771057229.0,
        -180193667.0 / 1043307555.0]
      #[39632708.0 / 573591083.0, 0.0, 0.0, -433636366.0 / 683701615.0,
        -421739975.0 / 2616292301.0, 100302831.0 / 723423059.0,
        790204164.0 / 839813087.0, 800635310.0 / 3783071287.0]
      #[246121993.0 / 1340847787.0, 0.0, 0.0, -37695042795.0 / 15268766246.0,
        -309121744.0 / 1061227803.0, -12992083.0 / 490766935.0,
        6005943493.0 / 2108947869.0, 393006217.0 / 1396673457.0,
        123872331.0 / 1001029789.0]
      #[-1028468189.0 / 846180014.0, 0.0, 0.0, 8478235783.0 / 508512852.0,
        1311729495.0 / 1432422823.0, -10304129995.0 / 1701304382.0,
        -48777925059.0 / 3047939560.0, 15336726248.0 / 1032824649.0,
        -45442868181.0 / 3398467696.0, 3065993473.0 / 597172653.0]
      #[185892177.0 / 718116043.0, 0.0, 0.0, -3185094517.0 / 667107341.0,
        -477755414.0 / 1098053517.0, -703635378.0 / 230739211.0,
        5731566787.0 / 1027545527.0, 5232866602.0 / 850066563.0,
        -4093664535.0 / 808688257.0, 3962137247.0 / 1805957418.0,
        65686358.0 / 487910083.0]
      #[403863854.0 / 491063109.0, 0.0, 0.0, -5068492393.0 / 434740067.0,
        -411421997.0 / 543043805.0, 652783627.0 / 914296604.0,
        11173962825.0 / 925320556.0, -13158990841.0 / 6184727034.0,
        3936647629.0 / 1978049680.0, -160528059.0 / 685178525.0,
        248638103.0 / 1413531060.0, 0.0]
      #[14005451.0 / 335480064.0, 0.0, 0.0, 0.0, 0.0,
        -59238493.0 / 1068277825.0, 181606767.0 / 758867731.0,
        561292985.0 / 797845732.0, -1041891430.0 / 1371343529.0,
        760417239.0 / 1151165299.0, 118820643.0 / 751138087.0,
        -528747749.0 / 2220607170.0, 0.25]
    b := bSol
    c := vec14
      0.0
      (1.0 / 18.0)
      (1.0 / 12.0)
      (1.0 / 8.0)
      (5.0 / 16.0)
      (3.0 / 8.0)
      (59.0 / 400.0)
      (93.0 / 200.0)
      (5490023248.0 / 9719169821.0)
      (13.0 / 20.0)
      (1201146811.0 / 1299019798.0)
      1.0
      1.0
      1.0
    bErr := some (vecSub bSol bErr)
    order := 8
  }

end DiffEq
end torch
