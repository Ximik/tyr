import Lean.Widget.UserWidget
import Lean.Widget.InteractiveCode
import Lean.Widget.Commands
import Lean.Elab.Command
import Lean.Elab.Tactic
import Lean.Elab.Deriving
import Tyr.Basic
import Tyr.TensorStruct
import Tyr.Torch

/-!
# Tyr.Widget

`Tyr.Widget` provides interactive Lean infoview widgets for tensor/module inspection.
It includes frontend + RPC structures for tensor heatmaps and hierarchical module trees,
plus deriving support for module display conversion.

## Major Components

- Widget data models (`TensorDisplayProps`, `ModuleNode`, `ModuleDisplayProps`).
- `TensorWidget` frontend module for interactive visualization.
- Tensor-structure to display-tree conversion helpers.
- Deriving/registration support for `ToModuleDisplay`.

## Scope

This module is focused on developer ergonomics and visualization in the Lean UI.
It complements, but does not replace, runtime/model computation modules.

## VS Code Quick Start

Widgets render in the Lean infoview when you place your cursor on a line containing
`#tensor ...` or `#module ...`.

1. Open a `.lean` file in VS Code with the Lean 4 extension enabled.
2. Add `import Tyr.Widget` (or import a module that already pulls it in).
3. Write `#tensor` / `#module` commands.
4. Place cursor on those lines to open the interactive panel.

```lean
import Tyr.Widget

open torch

#tensor (Tensor.zeros #[8, 8])
#tensor (Tensor.randn #[16, 32])   -- IO tensor expression also works
```

For module trees, pass `ModuleDisplayProps` (or `IO ModuleDisplayProps`):

```lean
import Tyr.Module.Linear
import Tyr.Widget

open torch

def demoModule : IO ModuleDisplayProps := do
  let lin ← Linear.init 16 32
  toModuleDisplayProps lin "Linear(16->32)"

#module demoModule
```

## Deriving Support

You can derive `ToModuleDisplay` for your own structures, then call
`toModuleDisplayProps` and inspect the result with `#module`.

```lean
import Tyr.Module.Linear
import Tyr.Widget

open torch

structure TinyMLP where
  fc1 : Linear 8 16
  fc2 : Linear 16 4
  note : Static String
  deriving ToModuleDisplay

def tinyTree : IO ModuleDisplayProps := do
  let fc1 ← Linear.init 8 16
  let fc2 ← Linear.init 16 4
  let m : TinyMLP := { fc1, fc2, note := "demo" }
  toModuleDisplayProps m "TinyMLP"

#module tinyTree
```
-/

namespace torch

open Lean Widget Server Elab Command

/-! ## Tensor Visualization Widget

Penzai/Treescope-style tensor visualization for the VSCode Lean4 infoview.
Features:
- Diverging colormap (red ← 0 → blue)
- Axis labels with dimension sizes
- Faceted layout for 3D+ tensors with hierarchical gaps
- Collapsible tree structure for modules
-/

/-- Props for a single tensor display -/
structure TensorDisplayProps where
  shape : Array UInt64
  dtype : String
  device : String
  values : Array Float
  requiresGrad : Bool
  numel : UInt64
  stats : String
  gradValues : Option (Array Float) := none
  gradStats : Option String := none
  name : Option String := none  -- Optional name for the tensor
  axisNames : Option (Array String) := none  -- Optional axis names
  deriving ToJson, FromJson, Server.RpcEncodable

/-- A node in a module tree (for collapsible structure) -/
inductive ModuleNode where
  | tensor : TensorDisplayProps → ModuleNode
  | group : String → Array (String × ModuleNode) → ModuleNode  -- name, children
  | static : String → String → ModuleNode  -- name, value as string (for Static fields)
  deriving ToJson, FromJson, Server.RpcEncodable

/-- Props for the full module widget -/
structure ModuleDisplayProps where
  root : ModuleNode
  deriving ToJson, FromJson, Server.RpcEncodable

@[widget_module]
def TensorWidget : Widget.Module where
  javascript := "
import * as React from 'react';
const e = React.createElement;
const { useState, useMemo, useEffect } = React;

const CELL_SIZE = 7;  // Treescope uses 7x7 pixel cells
const AXIS_LABEL_WIDTH = 24;
const AXIS_LABEL_HEIGHT = 18;

// Detect dark mode from VSCode theme
function useIsDarkMode() {
  const [isDark, setIsDark] = useState(() => {
    if (typeof document !== 'undefined') {
      return document.body.classList.contains('vscode-dark') ||
             document.body.classList.contains('vscode-high-contrast');
    }
    return false;
  });

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(
        document.body.classList.contains('vscode-dark') ||
        document.body.classList.contains('vscode-high-contrast')
      );
    });
    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  return isDark;
}

// Theme-aware colors
function useTheme() {
  const isDark = useIsDarkMode();
  return {
    isDark,
    text: isDark ? '#e0e0e0' : '#333',
    textMuted: isDark ? '#aaa' : '#666',
    textSubtle: isDark ? '#888' : '#999',
    background: isDark ? '#1e1e1e' : '#fff',
    backgroundAlt: isDark ? '#2d2d2d' : '#f5f5f5',
    border: isDark ? '#444' : '#ddd',
    badgeBg: isDark ? '#1e3a5f' : '#e3f2fd',
    badgeText: isDark ? '#64b5f6' : '#1565c0',
    tooltipBg: isDark ? 'rgba(60,60,60,0.95)' : 'rgba(0,0,0,0.85)',
  };
}

// Diverging and categorical colormap
function valueToColor(v, absMax, isCategorical) {
  if (!isFinite(v)) {
    if (v === Infinity) return '#0066ff';
    if (v === -Infinity) return '#ff0066';
    return '#888888';
  }
  if (isCategorical) {
    let hash = Math.floor(v) * 2654435761;
    hash = hash ^ (hash >> 16);
    const r = (hash & 0xFF0000) >> 16;
    const g = (hash & 0x00FF00) >> 8;
    const b = hash & 0x0000FF;
    return `rgb(${r % 155 + 100}, ${g % 155 + 100}, ${b % 155 + 100})`;
  }
  const norm = Math.max(-1, Math.min(1, v / (absMax + 1e-8)));
  if (norm >= 0) {
    const intensity = Math.floor((1 - norm) * 255);
    return `rgb(${intensity}, ${intensity}, 255)`;
  } else {
    const intensity = Math.floor((1 + norm) * 255);
    return `rgb(255, ${intensity}, ${intensity})`;
  }
}

function formatValue(v) {
  if (!isFinite(v)) {
    if (v === Infinity) return '+∞';
    if (v === -Infinity) return '-∞';
    return 'NaN';
  }
  if (Number.isInteger(v)) {
    return v.toString();
  }
  if (Math.abs(v) < 0.001 || Math.abs(v) >= 10000) {
    return v.toExponential(2);
  }
  return v.toFixed(4);
}

// Axis label component - Penzai style
function AxisLabel({ text, position, orientation, theme }) {
  const style = {
    fontSize: 9,
    fontFamily: 'monospace',
    fill: theme?.textMuted || '#666',
    fontWeight: 500
  };

  if (orientation === 'vertical') {
    return e('text', {
      x: position.x,
      y: position.y,
      transform: `rotate(-90, ${position.x}, ${position.y})`,
      textAnchor: 'middle',
      ...style
    }, text);
  }

  return e('text', {
    x: position.x,
    y: position.y,
    textAnchor: 'middle',
    ...style
  }, text);
}

// Single 2D facet with axis annotations
// fullRows/fullCols: actual tensor dimensions (for stride calculation)
// displayRows/displayCols: how many rows/cols to actually render
function Facet({ values, displayRows, displayCols, fullRows, fullCols, absMax, isCategorical, offset = 0, rowAxisLabel, colAxisLabel, showIndices = true, theme }) {
  const [hovered, setHovered] = useState(null);

  // Use full dimensions for stride if provided, otherwise use display dimensions
  const strideRows = fullRows || displayRows;
  const strideCols = fullCols || displayCols;

  const width = displayCols * CELL_SIZE + AXIS_LABEL_WIDTH;
  const height = displayRows * CELL_SIZE + AXIS_LABEL_HEIGHT;

  return e('div', { style: { display: 'inline-block', position: 'relative' } },
    e('svg', {
      width: width,
      height: height,
      style: { display: 'block', overflow: 'visible' }
    },
      // Column axis label (top)
      colAxisLabel && e(AxisLabel, {
        text: colAxisLabel,
        position: { x: AXIS_LABEL_WIDTH + (displayCols * CELL_SIZE) / 2, y: 8 },
        orientation: 'horizontal',
        theme
      }),

      // Row axis label (left, rotated)
      rowAxisLabel && e(AxisLabel, {
        text: rowAxisLabel,
        position: { x: 8, y: AXIS_LABEL_HEIGHT + (displayRows * CELL_SIZE) / 2 },
        orientation: 'vertical',
        theme
      }),

      // Column indices
      showIndices && displayCols <= 32 && Array.from({ length: displayCols }, (_, j) =>
        (j % Math.max(1, Math.floor(displayCols / 8)) === 0 || j === displayCols - 1) &&
        e('text', {
          key: `col-${j}`,
          x: AXIS_LABEL_WIDTH + j * CELL_SIZE + CELL_SIZE / 2,
          y: AXIS_LABEL_HEIGHT - 2,
          textAnchor: 'middle',
          fontSize: 6,
          fill: theme?.textSubtle || '#999'
        }, j)
      ),

      // Row indices
      showIndices && displayRows <= 32 && Array.from({ length: displayRows }, (_, i) =>
        (i % Math.max(1, Math.floor(displayRows / 8)) === 0 || i === displayRows - 1) &&
        e('text', {
          key: `row-${i}`,
          x: AXIS_LABEL_WIDTH - 2,
          y: AXIS_LABEL_HEIGHT + i * CELL_SIZE + CELL_SIZE / 2 + 2,
          textAnchor: 'end',
          fontSize: 6,
          fill: theme?.textSubtle || '#999'
        }, i)
      ),

      // Cells - use stride to correctly index into original tensor
      Array.from({ length: displayRows * displayCols }, (_, idx) => {
        const row = Math.floor(idx / displayCols), col = idx % displayCols;
        // Calculate actual index in the flattened tensor using stride
        const actualIdx = offset + row * strideCols + col;
        const v = values[actualIdx];
        const isSpecial = !isFinite(v);
        const isHovered = hovered && hovered.row === row && hovered.col === col;

        return e('g', { key: idx },
          e('rect', {
            x: AXIS_LABEL_WIDTH + col * CELL_SIZE,
            y: AXIS_LABEL_HEIGHT + row * CELL_SIZE,
            width: CELL_SIZE,
            height: CELL_SIZE,
            fill: valueToColor(v, absMax, isCategorical),
            stroke: isHovered ? (theme?.isDark ? '#fff' : '#000') : 'none',
            strokeWidth: isHovered ? 1.5 : 0,
            style: { cursor: 'crosshair' },
            onMouseEnter: () => setHovered({ row, col, value: v }),
            onMouseLeave: () => setHovered(null)
          }),
          isSpecial && e('text', {
            x: AXIS_LABEL_WIDTH + col * CELL_SIZE + CELL_SIZE / 2,
            y: AXIS_LABEL_HEIGHT + row * CELL_SIZE + CELL_SIZE / 2 + 2,
            textAnchor: 'middle',
            fontSize: 5,
            fill: '#fff',
            style: { pointerEvents: 'none' }
          }, v === Infinity ? '+' : v === -Infinity ? '-' : 'X')
        );
      })
    ),
    // Hover tooltip
    hovered && e('div', {
      style: {
        position: 'absolute',
        left: AXIS_LABEL_WIDTH + hovered.col * CELL_SIZE + CELL_SIZE,
        top: AXIS_LABEL_HEIGHT + hovered.row * CELL_SIZE,
        fontSize: 10,
        background: theme?.tooltipBg || 'rgba(0,0,0,0.85)',
        color: '#fff',
        padding: '3px 6px',
        borderRadius: 3,
        whiteSpace: 'nowrap',
        zIndex: 100,
        pointerEvents: 'none'
      }
    }, `[${hovered.row}, ${hovered.col}] = ${formatValue(hovered.value)}`)
  );
}

// Collapsible section for module tree
function Collapsible({ title, children, defaultOpen = true, depth = 0, theme }) {
  const [open, setOpen] = useState(defaultOpen);

  return e('div', {
    style: {
      marginLeft: depth * 12,
      borderLeft: depth > 0 ? `1px solid ${theme?.border || '#ddd'}` : 'none',
      paddingLeft: depth > 0 ? 8 : 0
    }
  },
    e('div', {
      style: {
        display: 'flex',
        alignItems: 'center',
        cursor: 'pointer',
        padding: '2px 0',
        userSelect: 'none'
      },
      onClick: () => setOpen(!open)
    },
      e('span', {
        style: {
          display: 'inline-block',
          width: 12,
          fontSize: 10,
          color: theme?.textMuted || '#666'
        }
      }, open ? '▼' : '▶'),
      e('span', {
        style: {
          fontFamily: 'monospace',
          fontSize: 11,
          fontWeight: 500,
          color: theme?.text || '#333'
        }
      }, title)
    ),
    open && e('div', { style: { marginTop: 4 } }, children)
  );
}

// Render a module node (recursive)
function ModuleNodeView({ node, name, depth = 0, theme, filterText, isVisible }) {
  if (isVisible && !isVisible(node, name)) return null;
  if (node.tensor) {
    const props = node.tensor;
    return e('div', { style: { marginLeft: depth * 12 } },
      name && e('div', {
        style: {
          fontFamily: 'monospace',
          fontSize: 10,
          color: theme?.textMuted || '#666',
          marginBottom: 2
        }
      }, name),
      e(TensorView, { ...props, compact: depth > 0, theme, filterText, isVisible })
    );
  }

  if (node.group) {
    const [groupName, children] = node.group;
    const computeParams = (n) => {
      let count = 0;
      if (n.tensor && n.tensor.requiresGrad && n.tensor.numel) count += n.tensor.numel;
      else if (n.group) n.group[1].forEach(c => count += computeParams(c[1]));
      return count;
    };
    const params = computeParams(node);
    const paramStr = params > 0 ? ` [${params.toLocaleString()} params]` : '';

    return e(Collapsible, {
      title: `${name || groupName} (${children.length} items)${paramStr}`,
      depth,
      defaultOpen: depth < 2 || !!filterText,
      theme
    },
      children.map(([childName, childNode], i) =>
        e(ModuleNodeView, {
          key: i,
          node: childNode,
          name: childName,
          depth: depth + 1,
          theme,
          filterText,
          isVisible
        })
      )
    );
  }

  // Static field (configuration/hyperparameter) - display as simple label
  if (node.static) {
    const [staticName, value] = node.static;
    return e('div', {
      style: {
        marginLeft: depth * 12,
        fontFamily: 'monospace',
        fontSize: 10,
        color: theme?.textMuted || '#666',
        padding: '2px 0'
      }
    }, `${name || staticName}: ${value}`);
  }

  return null;
}

// Main tensor display
function TensorView({ shape, dtype, device, values, gradValues, requiresGrad, stats, gradStats, numel, name, axisNames, compact = false, theme }) {
  const [showGrad, setShowGrad] = useState(false);

  // Parse stats from JSON
  const parsedStats = useMemo(() => {
    const s = showGrad && gradStats ? gradStats : stats;
    try {
      if (typeof s === 'string') return JSON.parse(s);
      return s;
    } catch {
      return { min: 0, max: 0, mean: 0 };
    }
  }, [stats, gradStats, showGrad]);

  const activeValues = showGrad && gradValues ? gradValues : values;

  const absMax = Math.max(Math.abs(parsedStats?.min || 0), Math.abs(parsedStats?.max || 0));
  const isCategorical = !showGrad && (dtype.toLowerCase().includes('int') || dtype.toLowerCase().includes('bool'));

  const ndim = shape.length;
  const getAxisLabel = (i) => axisNames?.[i] || `axis${i}`;

  // Header with shape info
  const Header = () => e('div', {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      marginBottom: compact ? 2 : 6,
      flexWrap: 'wrap'
    }
  },
    // Shape badge
    e('span', {
      style: {
        background: theme?.badgeBg || '#e3f2fd',
        color: theme?.badgeText || '#1565c0',
        padding: '2px 6px',
        borderRadius: 3,
        fontFamily: 'monospace',
        fontSize: 11,
        fontWeight: 600
      }
    }, shape.map((s, i) => {
      const label = axisNames?.[i];
      return label ? `${label}:${s}` : s;
    }).join(' × ')),

    // Dtype/device
    e('span', {
      style: {
        color: theme?.textMuted || '#666',
        fontSize: 10,
        fontFamily: 'monospace'
      }
    }, `${dtype} ${device}`),

    // Range info
    !compact && e('span', {
      style: {
        color: theme?.textSubtle || '#888',
        fontSize: 9,
        fontFamily: 'monospace'
      }
    }, `[${formatValue(parsedStats?.min || 0)} .. ${formatValue(parsedStats?.max || 0)}]`),

    // Requires Grad info & Toggle
    !compact && requiresGrad && e('div', {
      style: { display: 'flex', alignItems: 'center' }
    },
      e('span', {
        style: {
          background: theme?.badgeBg || '#e3f2fd',
          color: theme?.badgeText || '#1565c0',
          padding: '2px 6px',
          borderRadius: 3,
          fontSize: 9,
          fontWeight: 'bold',
          marginLeft: 4
        }
      }, 'requires_grad'),
      e('button', {
        style: {
          background: 'none',
          border: `1px solid ${theme?.border || '#ccc'}`,
          borderRadius: 3,
          fontSize: 9,
          padding: '2px 6px',
          marginLeft: 4,
          cursor: 'pointer',
          color: showGrad ? '#fff' : (theme?.text || '#333'),
          backgroundColor: showGrad ? (theme?.badgeText || '#1565c0') : 'transparent'
        },
        onClick: () => setShowGrad(!showGrad)
      }, 'show_grad')
    )
  );

  // 0D: scalar
  if (ndim === 0) {
    return e('div', { style: { fontFamily: 'monospace', padding: compact ? 4 : 8, color: theme?.text } },
      e(Header),
      e('div', { style: { fontSize: 14, fontWeight: 500 } }, formatValue(activeValues[0]))
    );
  }

  // 1D: horizontal strip
  if (ndim === 1) {
    const len = Number(shape[0]);
    const displayLen = Math.min(len, 128);

    return e('div', { style: { fontFamily: 'monospace', padding: compact ? 4 : 8 } },
      e(Header),
      e(Facet, {
        values: activeValues,
        displayRows: 1,
        displayCols: displayLen,
        fullRows: 1,
        fullCols: len,
        absMax,
        isCategorical,
        offset: 0,
        colAxisLabel: `${getAxisLabel(0)}:${len}`,
        showIndices: displayLen <= 64,
        theme
      }),
      len > displayLen && e('div', {
        style: { fontSize: 9, color: theme?.textSubtle || '#999', marginTop: 2 }
      }, `... ${len - displayLen} more`)
    );
  }

  // 2D: single facet
  if (ndim === 2) {
    const rows = Number(shape[0]), cols = Number(shape[1]);
    const displayRows = Math.min(rows, 64);
    const displayCols = Math.min(cols, 64);

    return e('div', { style: { fontFamily: 'monospace', padding: compact ? 4 : 8 } },
      e(Header),
      e(Facet, {
        values: activeValues,
        displayRows,
        displayCols,
        fullRows: rows,
        fullCols: cols,
        absMax,
        isCategorical,
        offset: 0,
        rowAxisLabel: `${getAxisLabel(0)}:${rows}`,
        colAxisLabel: `${getAxisLabel(1)}:${cols}`,
        showIndices: true,
        theme
      }),
      (rows > displayRows || cols > displayCols) && e('div', {
        style: { fontSize: 9, color: theme?.textSubtle || '#999', marginTop: 2 }
      }, `showing [${displayRows}×${displayCols}] of [${rows}×${cols}]`)
    );
  }

  // 3D+: faceted layout with hierarchical gaps
  const rows = Number(shape[ndim - 2]) || 1;
  const cols = Number(shape[ndim - 1]) || 1;
  const sliceSize = rows * cols;

  let numSlices = 1;
  for (let i = 0; i < ndim - 2; i++) {
    numSlices *= Number(shape[i]);
  }

  const displayRows = Math.min(rows, 32);
  const displayCols = Math.min(cols, 32);
  const maxSlices = 24;

  // Calculate slice indices for labeling
  const getSliceLabel = (flatIdx) => {
    const indices = [];
    let remaining = flatIdx;
    for (let i = ndim - 3; i >= 0; i--) {
      const dimSize = Number(shape[i]);
      indices.unshift(remaining % dimSize);
      remaining = Math.floor(remaining / dimSize);
    }
    return indices.map((idx, i) => `${getAxisLabel(i)}=${idx}`).join(', ');
  };

  // Group slices with gaps for hierarchy (for 4D+)
  const groupSize = ndim >= 4 ? Number(shape[ndim - 3]) : numSlices;

  return e('div', { style: { fontFamily: 'monospace', padding: compact ? 4 : 8 } },
    e(Header),
    e('div', {
      style: {
        display: 'flex',
        flexWrap: 'wrap',
        gap: 4,
        alignItems: 'flex-start'
      }
    },
      Array.from({ length: Math.min(numSlices, maxSlices) }, (_, i) =>
        e('div', {
          key: i,
          style: {
            marginRight: (i + 1) % groupSize === 0 ? 12 : 0  // Extra gap between groups
          }
        },
          e('div', {
            style: {
              fontSize: 8,
              color: theme?.textSubtle || '#888',
              marginBottom: 1,
              fontFamily: 'monospace'
            }
          }, getSliceLabel(i)),
          e(Facet, {
            values: activeValues,
            displayRows,
            displayCols,
            fullRows: rows,
            fullCols: cols,
            absMax,
            isCategorical,
            offset: i * sliceSize,
            rowAxisLabel: i === 0 ? `${getAxisLabel(ndim-2)}` : null,
            colAxisLabel: i === 0 ? `${getAxisLabel(ndim-1)}` : null,
            showIndices: i === 0,
            theme
          })
        )
      )
    ),
    numSlices > maxSlices && e('div', {
      style: { fontSize: 9, color: theme?.textSubtle || '#999', marginTop: 4 }
    }, `... ${numSlices - maxSlices} more slices`)
  );
}

// Main export - handles both single tensor and module tree
export default function Widget(props) {
  const theme = useTheme();
  const [filterText, setFilterText] = useState('');

  // Check if this is a module tree or single tensor
  if (props.root) {
    const computeTotalParams = (node) => {
      let count = 0;
      if (node.tensor && node.tensor.requiresGrad) {
        count += node.tensor.numel || 0;
      } else if (node.group) {
        node.group[1].forEach(child => count += computeTotalParams(child[1]));
      }
      return count;
    };

    const isVisible = (node, name) => {
      if (!filterText) return true;
      const lowerFilter = filterText.toLowerCase();
      if (name && name.toLowerCase().includes(lowerFilter)) return true;
      if (node.group && node.group[0].toLowerCase().includes(lowerFilter)) return true;
      if (node.group) {
        return node.group[1].some(child => isVisible(child[1], child[0]));
      }
      return false;
    };

    const totalParams = useMemo(() => computeTotalParams(props.root), [props.root]);

    return e('div', { style: { padding: 8 } },
      e('div', { style: { marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' } },
        e('input', {
          type: 'text',
          placeholder: 'Search modules...',
          value: filterText,
          onChange: (ev) => setFilterText(ev.target.value),
          style: {
            padding: '4px 8px',
            borderRadius: 4,
            border: `1px solid ${theme.border}`,
            background: theme.backgroundAlt,
            color: theme.text,
            width: 200,
            fontFamily: 'monospace',
            outline: 'none',
            fontSize: 12
          }
        }),
        totalParams > 0 && e('div', {
          style: { fontFamily: 'monospace', fontSize: 12, color: theme.textSubtle }
        }, `Trainable params: ${totalParams.toLocaleString()}`)
      ),
      e(ModuleNodeView, { node: props.root, name: null, depth: 0, theme, filterText, isVisible })
    );
  }

  // Single tensor
  return e(TensorView, { ...props, theme });
}
"

/-- Convert a tensor to widget props -/
def tensorToProps {σ : StaticSpec} (t : Tensor σ) : IO TensorDisplayProps := do
  let shape := t.runtimeShape
  let dtype := toString t.dtype
  let device := t.raw.deviceStr
  let requiresGrad := t.requiresGrad
  let stats := t.raw.stats
  let values := t.raw.getValues 10000  -- Get up to 10k values
  let numel := shape.foldl (· * ·) 1
  let gradient ←
    if requiresGrad then
      pure (some (← Tensor.gradOf t))
    else pure none
  let gradValues := gradient.map fun g => (g.raw.getValues 10000).toList.toArray
  let gradStats := gradient.map fun g => g.raw.stats
  pure { shape, dtype, device, requiresGrad, numel, stats, values := values.toList.toArray, gradValues, gradStats }

/-- Convert a tensor with name and axis labels -/
def tensorToPropsNamed {σ : StaticSpec} (t : Tensor σ) (name : String) (axisNames : Array String := #[]) : IO TensorDisplayProps := do
  let base ← tensorToProps t
  pure { base with name := some name, axisNames := if axisNames.isEmpty then none else some axisNames }

/-! ## ToModuleDisplay Typeclass

Automatically convert structures with tensors to ModuleDisplayProps for visualization.
Works with any type that has a TensorStruct instance.
-/

/-- Typeclass for converting values to ModuleNode for visualization -/
class ToModuleDisplay (α : Type) where
  /-- Convert a value to a ModuleNode with the given name -/
  toModuleNode : α → String → IO ModuleNode

/-- Convert any ToModuleDisplay to full props -/
def toModuleDisplayProps [ToModuleDisplay α] (value : α) (name : String := "root") : IO ModuleDisplayProps := do
  pure { root := ← ToModuleDisplay.toModuleNode value name }

/-- Instance for single tensors -/
instance {σ : StaticSpec} : ToModuleDisplay (Tensor σ) where
  toModuleNode t name := do
    pure (ModuleNode.tensor (← tensorToPropsNamed t name))

/-- Instance for Optional values -/
instance [ToModuleDisplay α] : ToModuleDisplay (Option α) where
  toModuleNode opt name :=
    match opt with
    | some x => ToModuleDisplay.toModuleNode x name
    | none => pure (ModuleNode.group s!"{name} (none)" #[])

/-- Instance for Arrays -/
instance [ToModuleDisplay α] : ToModuleDisplay (Array α) where
  toModuleNode arr name := do
    let children ← arr.mapIdxM fun i x => do
      pure (s!"[{i}]", ← ToModuleDisplay.toModuleNode x s!"{name}[{i}]")
    pure (ModuleNode.group s!"{name} [{arr.size}]" children)

/-- Instance for Lists -/
instance [ToModuleDisplay α] : ToModuleDisplay (List α) where
  toModuleNode lst name := do
    let children ← lst.toArray.mapIdxM fun i x => do
      pure (s!"[{i}]", ← ToModuleDisplay.toModuleNode x s!"{name}[{i}]")
    pure (ModuleNode.group s!"{name} [{lst.length}]" children)

/-- Instance for tuples -/
instance [ToModuleDisplay α] [ToModuleDisplay β] : ToModuleDisplay (α × β) where
  toModuleNode pair name := do
    pure (ModuleNode.group name #[
      ("fst", ← ToModuleDisplay.toModuleNode pair.1 "fst"),
      ("snd", ← ToModuleDisplay.toModuleNode pair.2 "snd")
    ])

/-- Instance for Static (configuration/hyperparameters) - displays value as label -/
instance [ToString α] : ToModuleDisplay (Static α) where
  toModuleNode s name := pure (ModuleNode.static name (toString s.val))

/-- Evaluate an IO expression that produces TensorDisplayProps -/
private unsafe def evalTensorPropsIO (e : Expr) : TermElabM TensorDisplayProps := do
  let type ← Meta.inferType e
  let propsIO ← Meta.evalExpr (IO TensorDisplayProps) type e
  propsIO

/-- Safe wrapper for evalTensorPropsIO -/
@[implemented_by evalTensorPropsIO]
private opaque evalTensorPropsIOSafe (e : Expr) : TermElabM TensorDisplayProps

/-- Check if a type is IO applied to something -/
private def isIOType (type : Expr) : Bool :=
  type.isAppOf ``IO || type.isAppOf ``EIO || type.isAppOf ``BaseIO

/-- Evaluate an IO expression that produces ModuleDisplayProps -/
private unsafe def evalModulePropsIO (e : Expr) : TermElabM ModuleDisplayProps := do
  let type ← Meta.inferType e
  let propsIO ← Meta.evalExpr (IO ModuleDisplayProps) type e
  propsIO

/-- Safe wrapper for evalModulePropsIO -/
@[implemented_by evalModulePropsIO]
private opaque evalModulePropsIOSafe (e : Expr) : TermElabM ModuleDisplayProps

/-- Evaluate a non-IO expression that produces ModuleDisplayProps -/
private unsafe def evalModuleProps (e : Expr) : TermElabM ModuleDisplayProps := do
  let type ← Meta.inferType e
  Meta.evalExpr ModuleDisplayProps type e

/-- Safe wrapper for evalModuleProps -/
@[implemented_by evalModuleProps]
private opaque evalModulePropsSafe (e : Expr) : TermElabM ModuleDisplayProps

/-- Command to visualize a tensor in the infoview.

Works with both pure tensors and `IO` tensor expressions.

Examples:
```lean
#tensor (Tensor.zeros #[4, 4])
#tensor (Tensor.randn #[3, 8])
```

Tip: place your cursor on the `#tensor` line in VS Code to show the widget panel.
-/
elab "#tensor " t:term : command => do
  let props ← liftTermElabM do
    -- First elaborate the term to check its type
    let e ← Term.elabTerm t none
    let e ← instantiateMVars e
    let type ← Meta.inferType e
    if isIOType type then
      -- Sequence tensor construction before reading the gradient snapshot.
      let propsIO ← Term.elabTerm (← `(do tensorToProps (← $t)))
        (some (mkApp (mkConst ``IO) (mkConst ``TensorDisplayProps)))
      Term.synthesizeSyntheticMVarsNoPostponing
      let propsIO ← instantiateMVars propsIO
      evalTensorPropsIOSafe propsIO
    else
      -- Non-IO case: just apply tensorToProps
      let propsExpr ← Term.elabTerm (← `(tensorToProps $t))
        (some (mkApp (mkConst ``IO) (mkConst ``TensorDisplayProps)))
      Term.synthesizeSyntheticMVarsNoPostponing
      let propsExpr ← instantiateMVars propsExpr
      evalTensorPropsIOSafe propsExpr
  -- Encode props for RPC and save widget info
  let propsJson := Server.RpcEncodable.rpcEncode props
  liftCoreM <| Widget.savePanelWidgetInfo (hash TensorWidget.javascript) propsJson t

/-- Command to visualize a module tree in the infoview.

Works with `ModuleDisplayProps` and `IO ModuleDisplayProps` expressions.
Most model structs use `deriving ToModuleDisplay`, then `toModuleDisplayProps`.

Examples:
```lean
def linTree : IO ModuleDisplayProps := do
  let lin ← Linear.init 8 16
  toModuleDisplayProps lin "Linear(8->16)"

#module linTree
```

Tip: place your cursor on the `#module` line in VS Code to show the widget panel.
-/
elab "#module " t:term : command => do
  let props ← liftTermElabM do
    let e ← Term.elabTerm t none
    let e ← instantiateMVars e
    let type ← Meta.inferType e
    if isIOType type then
      evalModulePropsIOSafe e
    else
      evalModulePropsSafe e
  let propsJson := Server.RpcEncodable.rpcEncode props
  liftCoreM <| Widget.savePanelWidgetInfo (hash TensorWidget.javascript) propsJson t

/-! ## Deriving Handler for ToModuleDisplay -/

open Lean Elab Command Term Meta in
/-- Generate ToModuleDisplay instance for a structure -/
private def mkToModuleDisplayInstanceCmd (typeName : Name) : CommandElabM Unit := do
  let env ← getEnv

  -- Get structure info
  let some structInfo := getStructureInfo? env typeName
    | throwError "{typeName} is not a structure"

  -- Get the inductive info to find parameters
  let some (.inductInfo indInfo) := env.find? typeName
    | throwError "{typeName} is not an inductive type"

  let fieldNames := structInfo.fieldNames

  -- Build field entries: (name, toModuleNode x.field name)
  let fieldEntries : Array (TSyntax `term) ← fieldNames.mapM fun fname => do
    let fnameStr := toString fname
    `(term| ($(Lean.quote fnameStr), ← ToModuleDisplay.toModuleNode x.$(mkIdent fname) $(Lean.quote fnameStr)))

  let fieldsArray ← `(term| #[$[$fieldEntries],*])

  -- Get the type parameters by analyzing the inductive type
  let numParams := indInfo.numParams

  if numParams == 0 then
    -- No parameters - simple instance
    let instCmd ← `(command|
      instance : ToModuleDisplay $(mkIdent typeName) where
        toModuleNode x name := do pure (ModuleNode.group name $fieldsArray)
    )
    elabCommand instCmd
  else
    -- Has parameters - need to build binders
    -- Build the syntax in TermElabM, then elaborate in CommandElabM
    let instCmd ← liftTermElabM do
      Meta.forallTelescopeReducing indInfo.type fun params _ => do
        -- Take only the actual parameters (not indices)
        let paramBinders := params[:numParams]

        -- Build implicit binder syntax for each parameter
        let binderStxs ← paramBinders.toArray.mapM fun param => do
          let paramDecl ← param.fvarId!.getDecl
          let paramName := paramDecl.userName
          let paramType ← Meta.inferType param
          let paramTypeStx ← PrettyPrinter.delab paramType
          `(bracketedBinder| {$(mkIdent paramName) : $paramTypeStx})

        -- Build the applied type (e.g., Linear in_dim out_dim)
        let paramIdents ← paramBinders.toArray.mapM fun param => do
          let paramDecl ← param.fvarId!.getDecl
          pure (mkIdent paramDecl.userName)

        let appliedType ← `($(mkIdent typeName) $paramIdents*)

        -- Generate instance command with binders
        `(command|
          instance $[$binderStxs]* : ToModuleDisplay $appliedType where
            toModuleNode x name := do pure (ModuleNode.group name $fieldsArray)
        )

    elabCommand instCmd

open Lean Elab Deriving in
/-- Deriving handler for ToModuleDisplay -/
def mkToModuleDisplayHandler (typeNames : Array Name) : CommandElabM Bool := do
  if typeNames.isEmpty then return false

  for typeName in typeNames do
    mkToModuleDisplayInstanceCmd typeName

  return true

-- Register the deriving handler
open Lean Elab Deriving in
initialize
  registerDerivingHandler ``ToModuleDisplay mkToModuleDisplayHandler

end torch
