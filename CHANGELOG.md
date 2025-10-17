# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2024-10-17

### Added

- **UV Package Manager Support**: Full migration to UV for faster dependency management
- **Modern Dependency Groups**: Using `dependency-groups.dev` instead of deprecated format
- **Hatchling Build Backend**: Replaced setuptools with hatchling for modern builds

### Changed

- **Python Version Support**: Limited to 3.10-3.12 (TensorFlow compatibility)
- **TensorFlow Version**: Updated to 2.18.1 with proper version constraints
- **Package Manager**: UV is now the recommended package manager
- **Build System**: Switched from setuptools to hatchling
- **Dockerfile**: Updated to use UV and Python 3.12
- **Documentation**: All installation and development docs updated for UV

### Removed

- **Pip-specific Files**: Removed setup.py, setup.cfg, requirements.txt, requirements-dev.txt
- **Black/isort Configuration**: Replaced with ruff for formatting and linting

## [2.0.0] - 2024-10-17

### Added

- **PyPI Package Structure**: Complete restructure to PyPI-compliant package layout
- **Modern Dependencies**: Upgraded to TensorFlow 2.15+, Python 3.10+, Flask 3.0+
- **Type Hints**: Full type annotation coverage across the codebase
- **Comprehensive Documentation**:
  - Professional README with badges and detailed sections
  - Installation guide
  - Usage guide
  - API reference documentation
  - Contributing guidelines
  - Code of Conduct
  - Security policy
- **Package Configuration**:
  - `pyproject.toml` for modern Python packaging
  - `setup.py` and `setup.cfg` for setuptools
  - `MANIFEST.in` for package data
- **Utility Functions**:
  - `validate_input()` for input validation and preprocessing
  - `format_prediction()` for output formatting
  - `get_feature_names()` and `get_feature_descriptions()` for feature information
- **Enhanced API Server**:
  - Health check endpoint (`/health`)
  - Feature information endpoint (`/api/features`)
  - Improved error handling and validation
  - Logging support
  - Environment variable configuration
- **Testing Infrastructure**:
  - Comprehensive unit tests with pytest
  - Test coverage reporting
  - Integration tests
- **Development Tools**:
  - Black for code formatting
  - isort for import sorting
  - flake8 for linting
  - mypy for type checking
- **Docker Improvements**:
  - Multi-stage build
  - Python 3.10+ base image
  - Non-root user
  - Health checks
  - Optimized layer caching
- **GitHub Templates**:
  - Bug report template
  - Feature request template
- **Examples**: Jupyter notebook moved to `examples/` directory

### Changed

- **Project Structure**: Migrated to standard package layout with `mlpregression/` package directory
- **Model API**:
  - Renamed `def_model()` to `create_model()` with backward compatibility
  - Added configurable parameters for model creation
  - Updated to Keras 3.x API
- **Server API**:
  - Modernized Flask application structure
  - Added proper JSON responses for all endpoints
  - Improved error messages
- **Dependencies**: All dependencies upgraded to modern, actively maintained versions
- **Documentation**: Complete rewrite with professional formatting and comprehensive examples

### Fixed

- Security vulnerabilities in outdated dependencies
- Input validation edge cases
- Error handling in prediction endpoints

### Deprecated

- `def_model()` function (use `create_model()` instead, backward compatible)

## [1.0.0] - 2018-XX-XX

### Added

- Initial release
- Basic MLP model for Boston Housing prediction
- Flask server with prediction endpoint
- Jupyter notebook demo
- Docker support
- Basic README

---

## Version History

- **2.0.0** (2024-10-17): Complete package modernization and PyPI compliance
- **1.0.0** (2018): Initial release

