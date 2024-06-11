import pytest
import sys
import os

# sys.path.append('ml')
# print(os.getcwd())
from model import train_model


def test_train_model(X_train, y_train):
    trained_model = train_model(X_train=X_train, y_train=y_train)
    assert trained_model is not None


if __name__ == "__main__":
    pytest.main([__file__])
