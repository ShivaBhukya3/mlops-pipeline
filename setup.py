from setuptools import setup, find_packages

setup(
    name="mlops-fraud-detection",
    version="1.0.0",
    packages=find_packages(include=["src", "src.*", "dashboard", "dashboard.*"]),
    python_requires=">=3.10",
    description="Real-Time ML Pipeline for Fraud Detection — MLOps showcase project",
    author="MLOps Team",
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.7.1",
        "scikit-learn>=1.4.2",
        "xgboost>=2.0.3",
        "mlflow>=2.13.0",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "pyyaml>=6.0.1",
        "joblib>=1.4.2",
        "prometheus-client>=0.20.0",
    ],
    entry_points={
        "console_scripts": [
            "mlops-train=scripts.train:cli",
        ]
    },
)
