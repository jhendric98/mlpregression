---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''

---

## Bug Description

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Actual behavior**
A clear and concise description of what actually happened.

## Environment

**System Information:**
- OS: [e.g. macOS 12.0, Ubuntu 20.04, Windows 11]
- Python version: [e.g. 3.10.0]
- mlpregression version: [e.g. 2.0.0]
- TensorFlow version: [e.g. 2.15.0]

**Installation method:**
- [ ] pip install mlpregression
- [ ] pip install from GitHub
- [ ] Docker
- [ ] From source

## Code Sample

**Minimal code to reproduce the issue:**

```python
# Paste your code here
from mlpregression import create_model

model = create_model()
# ... rest of your code
```

**Input data (if applicable):**

```
# Paste input data that causes the issue
1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72
```

## Error Output

**Error message:**

```
# Paste the full error message and stack trace here
Traceback (most recent call last):
  File "...", line ..., in ...
    ...
Error: ...
```

**Logs (if applicable):**

```
# Paste relevant log output here
```

## Screenshots

**If applicable, add screenshots to help explain your problem.**

## Additional Context

**Add any other context about the problem here.**

- Does this happen consistently or intermittently?
- Have you tried any workarounds?
- Is this a regression from a previous version?
- Any other relevant information

## Checklist

- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided a minimal code example that reproduces the issue
- [ ] I have included my environment information
- [ ] I have included the full error message and stack trace
- [ ] I have tested with the latest version of mlpregression

## Possible Solution

**If you have ideas on how to fix the bug, please describe them here.**

## Impact

**How does this bug affect your use of mlpregression?**

- [ ] Blocks my work completely
- [ ] Significant impact on functionality
- [ ] Minor inconvenience
- [ ] Just noticed it, no immediate impact
