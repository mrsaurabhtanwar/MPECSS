# Changelog

All notable changes to the MPECSS solver are documented here.

## [1.0.3] - 2026-03-24
### Added
- **Safety Net**: Added a new "Phase III" fallback. If the main solver gets stuck, this step automatically steps in to find a valid solution.
- **Smart Labels**: The solver now clearly tells you if a solution is "S-stationary" (perfect) or "B-stationary" (reliable).
### Improved
- **Cleanup**: Removed ~1,900 lines of unused code and performed comprehensive documentation enhancement.
- **Accuracy**: Fine-tuned internal tolerances for better stability.

## [1.0.2] - 2026-03-23
### Updated
- Synchronized with the latest research findings.
- Updated project metadata for better compatibility.

## [1.0.1] - 2026-03-22
### Added
- Full support for the **886-problem benchmark suite**.
- Easy-to-use command line tools like `mpecss-macmpec`.

## [1.0.0] - 2026-03-22
### Initial Release
- The first official version of the MPECSS solver.
- Features Phase I (Finding a starting point) and Phase II (Main Solving Loop).
- Support for MacMPEC, MPECLib, and NOSBENCH problem formats.