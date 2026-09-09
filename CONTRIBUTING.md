# Contributing to DentalTensor Vision

Thank you for your interest in contributing to **DentalTensor Vision**!

---

## Code of Conduct
We are committed to providing a friendly, safe, and welcoming environment for all contributors. Please be respectful, constructive, and collaborative.

---

## How Can You Contribute?
1. **Reporting Bugs**: Open an issue detailing the steps to reproduce, sample image characteristics (do NOT post private patient data), and unexpected behavior.
2. **Improving Documentation**: Fix typos, clarify instructions, or add integration tutorials.
3. **Adding Unit Tests**: Enhance test coverage, particularly around edge-case image dimensions or thresholding logic.
4. **Research Tools**: Propose tools for calibration, evaluation, or benchmarking.

---

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/DentalTensor-Vision.git
   cd DentalTensor-Vision
   ```

2. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[all]"
   ```

3. Run the test suite:
   ```bash
   pytest
   ```

---

## Important Engineering Principles

- **Zero-Distortion Preprocessing**: Never introduce CLAHE, non-linear contrast alterations, or lossy re-encoding in the default inference pipeline.
- **Privacy & Safety**: Never commit private, patient-identifiable intraoral photographs.
- **License Compliance**: DentalTensor Vision is licensed under AGPL-3.0-only. All contributions must adhere to this license and maintain required third-party attributions.

---

## Submitting Pull Requests
1. Create a descriptive topic branch (`git checkout -b feature/my-enhancement`).
2. Ensure all tests pass (`pytest`).
3. Commit clean, atomic changes with descriptive commit messages.
4. Open a Pull Request referencing any relevant issues.
