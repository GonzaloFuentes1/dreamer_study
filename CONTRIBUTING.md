# Contributing to Dreamer Study

Thank you for your interest in contributing to the Dreamer Study project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, constructive, and professional in all interactions.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Provide clear descriptions and reproduction steps
- Include system information (OS, Python version, GPU, etc.)
- Attach relevant logs or error messages

### Submitting Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/dreamer_study.git
   cd dreamer_study
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Check code style
   flake8 .
   black --check .
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

6. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd dreamer_study

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Install pre-commit hooks (optional)
pre-commit install
```

## Code Style

- Follow PEP 8 for Python code
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use meaningful variable and function names
- Add docstrings to functions and classes

### Example

```python
def train_agent(config: dict, env_id: str, device: str) -> dict:
    """
    Train a Dreamer agent on the specified environment.
    
    Args:
        config: Configuration dictionary
        env_id: Environment identifier
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        dict: Training metrics
    """
    # Implementation here
    pass
```

## Priority Areas

We especially welcome contributions in these areas:

1. **Fix Dreamer V3** - Help fix the V3 implementation issues
2. **Complete Dreamer V4** - Implement the V4 variant
3. **Add Documentation** - Improve docs, add examples
4. **Testing** - Add unit tests and integration tests
5. **Performance** - Optimize training speed
6. **Environments** - Add support for more environments

## Testing

Before submitting a PR, ensure:

- [ ] Code runs without errors
- [ ] Existing tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Code follows style guidelines

## Documentation

When adding features:

- Update relevant README files
- Add docstrings to new functions/classes
- Include usage examples
- Update API documentation if needed

## Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when relevant

Example:
```
Add symexp twohot implementation for V3

- Implement symexp_encode and symexp_decode functions
- Add twohot_encode for categorical distributions
- Update reward model to use symexp twohot
- Fixes #123
```

## Questions?

If you have questions about contributing:

- Open an issue on GitHub
- Check existing documentation
- Review closed issues and PRs

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
