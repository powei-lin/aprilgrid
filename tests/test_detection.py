import pytest
from aprilgrid import Detector
from cv2 import imread, IMREAD_GRAYSCALE


@pytest.mark.parametrize(
    "file_name,tag_family_name,expected_num",
    [("tests/data/t36h11_2bit_margin.jpg", "t36h11", 36), ("tests/data/t36h11_1bit_margin.png", "t36h11b1", 154)],
)
def test_detect(file_name, tag_family_name, expected_num):
    detector = Detector(tag_family_name)
    img = imread(file_name, IMREAD_GRAYSCALE)
    detections = detector.detect(img)
    assert len(detections) == expected_num
