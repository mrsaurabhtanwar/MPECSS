# Benchmark Suites

This folder contains 886 real-world math problems used to test the MPECSS solver. These problems come from three major collections used by researchers worldwide.

> **Note:** Because the problem data is very large (~900 MB), it is not included directly in GitHub. Please download **`benchmarks.zip`** from our [Releases page](https://github.com/mrsaurabhtanwar/MPECSS/releases) and extract it here.

## The Three Suites

| Suite | Problems | What it covers |
| :--- | :---: | :--- |
| **MacMPEC** | 191 | Classic research problems, traffic flow, and bridge design. |
| **MPECLib** | 92 | Economic games, friction in machines, and electricity markets. |
| **NOSBENCH** | 603 | Advanced control problems for robots and complex systems. |
| **Total** | **886** | |

## How the Files are Organized

- **`macmpec/`**: 191 academic test cases.
- **`mpeclib/`**: 92 industrial and economic problems.
- **`nosbench/`**: 603 large-scale control problems.

## How to Load a Problem

If you are writing your own code, you can load these problems easily:

```python
from mpecss.helpers.loaders.macmpec_loader import load_macmpec

# Load a specific problem file
problem = load_macmpec("benchmarks/macmpec/macmpec-json/bard1.nl.json")
```

---
For more technical details on the file formats and original authors, see the specific README files in each folder.
