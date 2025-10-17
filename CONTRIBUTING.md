# Contributing to mlpregression

First off, thank you for considering contributing to mlpregression! It's people like you that make this project better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to jhendric98@gmail.com.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible using our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

**Great bug reports include:**

- A clear and descriptive title
- Exact steps to reproduce the problem
- Expected vs. actual behavior
- Code samples or test cases
- Your environment (OS, Python version, package versions)
- Any relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

**Good enhancement suggestions include:**

- A clear use case or problem being solved
- Detailed description of proposed solution
- Examples of how it would be used
- Alternative solutions you've considered
- Any potential drawbacks or considerations

### Pull Requests

We actively welcome your pull requests! Here's the process:

1. Fork the repository and create your branch from `main`
2. If you've added code, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows our style guidelines
6. Issue the pull request

## Development Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12 (3.13 not yet supported by TensorFlow)
- git
- UV (recommended) or pip

### Getting Started

1. **Fork and clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/mlpregression.git
cd mlpregression
```

2. **Install UV (if not already installed):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Sync development dependencies:**

```bash
uv sync --dev
```

**Alternative with pip:**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

4. **Verify your setup:**

```bash
pytest
```

### Project Structure

```
mlpregression/
├── mlpregression/       # Main package code
├── tests/               # Test files
├── examples/            # Example notebooks and scripts
├── docs/               # Documentation
└── models/             # Pre-trained model weights
```

## Coding Standards

We use automated tools to maintain code quality. Please ensure your code passes all checks before submitting a PR.

### Code Style

- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:

```bash
# Format code
black mlpregression tests

# Sort imports
isort mlpregression tests

# Lint
flake8 mlpregression tests

# Type check
mypy mlpregression
```

### Python Guidelines

- Use type hints for all function signatures
- Write docstrings for all public functions, classes, and modules
- Follow PEP 8 style guide
- Keep functions focused and single-purpose
- Prefer explicit over implicit
- Use meaningful variable names

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Brief description of function.

    Longer description if needed, explaining behavior,
    edge cases, or implementation details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When and why this is raised

    Example:
        >>> result = function_name(42, "test")
        >>> print(result)
        True
    """
    pass
```

## Testing

We use pytest for testing. All new features should include tests.

### Running Tests

```bash
# Run all tests with UV
uv run pytest

# Run with coverage
uv run pytest --cov=mlpregression --cov-report=html

# Run specific test file
uv run pytest tests/test_model.py

# Run specific test
uv run pytest tests/test_model.py::TestModel::test_create_model_default

# Run in verbose mode
uv run pytest -v
```

**Alternative with pip:**

```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run tests
pytest
pytest --cov=mlpregression --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use classes to group related tests
- Write both unit tests and integration tests
- Aim for high code coverage (>80%)

Example test:

```python
def test_validate_input_from_string():
    """Test input validation from comma-separated string."""
    input_str = "1.2,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"
    result = validate_input(input_str)
    assert result.shape == (1, 13)
    assert result[0, 0] == pytest.approx(1.2)
```

## Pull Request Process

1. **Create a feature branch:**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes:**
   - Write clear, concise commit messages
   - Keep commits atomic and focused
   - Follow conventional commits format (optional but appreciated)

3. **Test your changes:**

```bash
# Run tests
pytest

# Check code style
black --check mlpregression tests
flake8 mlpregression tests
mypy mlpregression
```

4. **Update documentation:**
   - Update README.md if needed
   - Update docstrings
   - Update CHANGELOG.md under "Unreleased"

5. **Push and create PR:**

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Link to related issues
- Screenshots if applicable
- Checklist of completed items

6. **PR Review Process:**
   - Maintainers will review your PR
   - Address any requested changes
   - Once approved, maintainers will merge

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
feat: add custom optimizer support to create_model()

- Added optimizer parameter to create_model()
- Added tests for different optimizers
- Updated documentation

Closes #123
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

## Questions?

Don't hesitate to ask questions! You can:

- Open an issue with the "question" label
- Email: jhendric98@gmail.com
- Check existing documentation

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for significant contributions
- GitHub contributors page
- Release notes

Thank you for contributing to mlpregression! 🎉

