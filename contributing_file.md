# Contributing to Sports Dance Education Scheduling System

First off, thank you for considering contributing to our project! It's people like you that make the Sports Dance Education Scheduling System such a great tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Process](#development-process)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@sportsdancescheduling.com.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- Git
- Virtual environment tool (venv or conda)
- CUDA toolkit (optional, for GPU acceleration)

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/sports-dance-scheduling.git
   cd sports-dance-scheduling
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/original-owner/sports-dance-scheduling.git
   ```

4. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

6. Set up pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps which reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed after following the steps**
- **Explain which behavior you expected to see instead and why**
- **Include screenshots and animated GIFs if possible**
- **Include your configuration file (sanitized of sensitive data)**
- **Include crash reports and stack traces**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and expected behavior**
- **Explain why this enhancement would be useful**
- **List any alternatives you've considered**

### Your First Code Contribution

Unsure where to begin contributing? You can start by looking through these issues:

- Issues labeled `good first issue` - simple issues good for beginners
- Issues labeled `help wanted` - issues where we need community help
- Issues labeled `documentation` - improving or writing documentation

## Development Process

### Branch Naming Convention

- `feature/` - New features (e.g., `feature/add-calendar-export`)
- `bugfix/` - Bug fixes (e.g., `bugfix/fix-scheduling-conflict`)
- `hotfix/` - Urgent fixes for production (e.g., `hotfix/critical-security-patch`)
- `docs/` - Documentation changes (e.g., `docs/update-api-docs`)
- `test/` - Test additions or modifications (e.g., `test/add-integration-tests`)
- `refactor/` - Code refactoring (e.g., `refactor/optimize-database-queries`)

### Development Workflow

1. Create a new branch from `develop`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes:
   - Write clean, readable code
   - Add or update tests as needed
   - Update documentation

3. Run tests locally:
   ```bash
   pytest tests/
   python sports_dance_scheduling.py --test-mode
   ```

4. Lint your code:
   ```bash
   black .
   isort .
   flake8 .
   mypy .
   ```

5. Commit your changes (see commit message guidelines below)

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a Pull Request

## Style Guidelines

### Python Style

We follow PEP 8 with some modifications:

- Line length: 100 characters (instead of 79)
- Use Black for automatic formatting
- Use type hints where appropriate
- Docstrings for all public functions and classes (Google style)

Example:
```python
def calculate_schedule_fitness(
    schedule: np.ndarray,
    constraints: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Calculate the fitness score of a schedule.
    
    Args:
        schedule: 3D array representing the schedule (classes x instructors x timeslots)
        constraints: Dictionary of scheduling constraints
        weights: Optional weights for different objectives
    
    Returns:
        Fitness score between 0 and 1, where 1 is optimal
    
    Raises:
        ValueError: If schedule dimensions don't match constraints
    """
    # Implementation here
    pass
```

### Code Organization

- Keep functions focused and single-purpose
- Use meaningful variable names
- Group related functionality into classes
- Separate concerns (data processing, model logic, API, etc.)
- Avoid global variables except for configuration

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Test additions or modifications
- **chore**: Build process or auxiliary tool changes
- **ci**: CI configuration changes

### Examples

```
feat(scheduler): add support for multi-venue scheduling

Implemented algorithm to handle scheduling across multiple venues
simultaneously. This includes:
- Venue capacity constraints
- Equipment availability checking
- Travel time between venues

Closes #123
```

```
fix(database): resolve connection pool exhaustion issue

Fixed bug where connections weren't being properly returned to the
pool after use, causing pool exhaustion under heavy load.

Fixes #456
```

## Testing

### Writing Tests

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use pytest fixtures for common test data
- Mock external dependencies
- Test edge cases and error conditions

### Test Structure

```python
import pytest
from sports_dance_scheduling import ScheduleOptimizer

class TestScheduleOptimizer:
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return ScheduleOptimizer()
    
    def test_basic_schedule_generation(self, optimizer):
        """Test that basic schedule generation works correctly."""
        # Arrange
        constraints = {...}
        
        # Act
        schedule = optimizer.generate(constraints)
        
        # Assert
        assert schedule is not None
        assert schedule.shape == expected_shape
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sports_dance_scheduling

# Run specific test file
pytest tests/test_optimizer.py

# Run with verbose output
pytest -v

# Run only marked tests
pytest -m "not slow"
```

## Documentation

### Docstring Guidelines

All public modules, functions, classes, and methods should have docstrings. We use Google style docstrings:

```python
def function_with_docstring(param1: int, param2: str) -> bool:
    """Brief description of function.
    
    Longer description if needed, explaining in more detail what
    the function does and any important notes about its behavior.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string
    
    Example:
        >>> result = function_with_docstring(5, "test")
        >>> print(result)
        True
    """
    pass
```

### Documentation Updates

- Update README.md for user-facing changes
- Update API documentation for endpoint changes
- Add docstrings for new functions and classes
- Update configuration examples if adding new options
- Include examples in documentation where helpful

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```

2. **Run the full test suite** and ensure all tests pass

3. **Update documentation** as needed

4. **Add tests** for new functionality

5. **Check code style** and fix any issues

### Pull Request Template

When creating a PR, please fill out the template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All tests pass locally
- [ ] Added new tests for new functionality
- [ ] Updated existing tests as needed

## Checklist
- [ ] My code follows the project style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

## Related Issues
Closes #(issue number)

## Screenshots (if applicable)
```

### Review Process

1. At least one maintainer review is required
2. All CI checks must pass
3. No merge conflicts
4. Documentation is updated
5. Tests are included and passing

### After Your PR is Merged

- Delete your branch (locally and on your fork)
- Pull the changes from upstream
- Celebrate your contribution! 🎉

## Recognition

Contributors will be recognized in several ways:

- Listed in CONTRIBUTORS.md file
- Mentioned in release notes for significant contributions
- GitHub contributor badge
- Special recognition for consistent contributors

## Getting Help

If you need help at any point:

- Check the documentation
- Search existing issues
- Ask in our Discord channel
- Email developers@sportsdancescheduling.com

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

Thank you for contributing to the Sports Dance Education Scheduling System!