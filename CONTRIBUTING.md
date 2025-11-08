# Contributing to FESTA

Thank you for your interest in contributing to FESTA! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/LLAVA-V5-2.git
   cd LLAVA-V5-2
   ```
3. **Set up your environment**:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Development Workflow

1. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Test your changes**:
   ```bash
   # Run test with 2 samples
   python3 src/festa_with_apis.py
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Code Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add comments for complex logic

## Testing

- Always test your changes with at least 2 samples before submitting
- Ensure all metrics are calculated correctly
- Verify visualizations are generated properly
- Check that CSV exports work correctly

## Areas for Contribution

### High Priority
- Performance optimizations for GPU usage
- Additional calibration metrics
- Support for more VLM models
- Improved error handling and logging

### Medium Priority
- Additional visualization types
- Export formats (Excel, HTML reports)
- Configuration UI or wizard
- Batch processing improvements

### Documentation
- Tutorial videos or walkthroughs
- More example use cases
- API documentation improvements
- Translation to other languages

## Bug Reports

When reporting bugs, please include:
- Python version
- GPU/CPU configuration
- Full error traceback
- Steps to reproduce
- Expected vs actual behavior
- Sample configuration (without API keys!)

## Feature Requests

When suggesting features, please:
- Describe the use case
- Explain why it would be valuable
- Provide examples if possible
- Consider backward compatibility

## Questions?

- Check existing documentation in the repo
- Review closed issues for similar questions
- Open a new issue with the "question" label

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on what is best for the community
- Show empathy towards others

Thank you for contributing to FESTA! 🎉

