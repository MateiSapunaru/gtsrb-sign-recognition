def test_basic_imports():
    import torch
    import torchvision
    import streamlit
    import cv2
    import numpy
    import pandas
    import sklearn

    assert torch is not None
    assert torchvision is not None
    assert streamlit is not None
    assert cv2 is not None
    assert numpy is not None
    assert pandas is not None
    assert sklearn is not None