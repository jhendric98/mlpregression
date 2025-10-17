# Security Policy

## Supported Versions

We actively support the following versions of mlpregression with security updates:

| Version | Supported          | End of Support |
| ------- | ------------------ | -------------- |
| 2.0.x   | :white_check_mark: | TBD            |
| < 2.0   | :x:                | 2024-10-17     |

## Security Considerations

### Model Security

- **Model Integrity**: The pre-trained model weights (`models/model.h5`) should be verified for integrity
- **Input Validation**: All user inputs are validated before processing to prevent injection attacks
- **Resource Limits**: The API server implements reasonable limits to prevent resource exhaustion

### API Security

- **Input Sanitization**: All API inputs are sanitized and validated
- **Error Handling**: Error messages don't expose sensitive system information
- **Rate Limiting**: Consider implementing rate limiting in production deployments
- **HTTPS**: Always use HTTPS in production environments

### Docker Security

- **Non-root User**: The Docker container runs as a non-root user
- **Minimal Base Image**: Uses official Python slim images with minimal attack surface
- **No Secrets in Image**: No sensitive information is baked into the Docker image

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in mlpregression, please report it responsibly:

### How to Report

1. **Email**: Send details to jhendric98@gmail.com with subject line "SECURITY: mlpregression vulnerability"
2. **Do NOT** create a public GitHub issue for security vulnerabilities
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

### What to Expect

- **Acknowledgment**: We'll acknowledge receipt within 48 hours
- **Initial Assessment**: Initial assessment within 5 business days
- **Regular Updates**: We'll provide updates on our progress every 5 business days
- **Resolution Timeline**: We aim to resolve critical vulnerabilities within 30 days

### Disclosure Policy

- **Coordinated Disclosure**: We follow responsible disclosure practices
- **Public Disclosure**: Vulnerabilities will be publicly disclosed after a fix is available
- **Credit**: Security researchers will be credited (unless they prefer to remain anonymous)

## Security Best Practices for Users

### Production Deployment

1. **Environment Variables**: Use environment variables for configuration, never hardcode secrets
2. **Network Security**: Deploy behind a reverse proxy with proper SSL/TLS configuration
3. **Access Control**: Implement proper authentication and authorization
4. **Monitoring**: Monitor for unusual API usage patterns
5. **Updates**: Keep dependencies updated and monitor for security advisories

### Docker Deployment

```bash
# Run with security best practices
docker run -d \
  --name mlpregression \
  --user 1000:1000 \
  --read-only \
  --tmpfs /tmp \
  --cap-drop ALL \
  -p 5002:5002 \
  mlpregression:latest
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  containers:
  - name: mlpregression
    image: mlpregression:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

## Known Security Considerations

### Data Privacy

- **Training Data**: The Boston Housing dataset contains historical socioeconomic data that may reflect biases
- **Input Data**: API inputs are logged for debugging; ensure no sensitive information is included
- **Predictions**: Model predictions should not be used for discriminatory purposes

### Model Limitations

- **Adversarial Inputs**: The model may be vulnerable to adversarial inputs designed to produce incorrect predictions
- **Data Drift**: Model performance may degrade over time as real-world data changes
- **Bias**: Historical training data may contain biases that affect predictions

## Security Updates

Security updates will be:

1. **Announced**: Via GitHub releases and security advisories
2. **Documented**: In CHANGELOG.md with security impact noted
3. **Versioned**: Following semantic versioning with patch releases for security fixes

## Contact

For security-related questions or concerns:

- **Email**: jhendric98@gmail.com
- **Subject**: Include "SECURITY" in the subject line
- **Response Time**: We aim to respond within 48 hours

## Attribution

This security policy is based on industry best practices and follows guidelines from:

- [OWASP Security Guidelines](https://owasp.org/)
- [GitHub Security Advisories](https://docs.github.com/en/code-security/security-advisories)
- [Python Security Guidelines](https://python.org/dev/security/)
