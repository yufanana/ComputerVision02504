# usr/bin/env python3
"""
Ex 6.7

Applying  different Canny edge detector thresholds using
cv2 Trackbars. Press Q to quit the program.

cd exercises
python canny_tuner.py
"""

import cv2


def canny_trackbar(im) -> None:
    """
    Create a trackbar to tune threshold1 values for canny edge detection.
    Press 's' to save the current value and move on with the program.
    """

    def canny_t1_callback(new_val) -> None:
        """
        Callback function for the trackbar that runs Canny edge
        detection using the trackbar value as threshold1.
        """
        nonlocal canny_t1, canny_t2, im
        canny_t1 = new_val
        edges = cv2.Canny(im, canny_t1, canny_t2)
        cv2.imshow("Canny Edge Detection", edges)

    def canny_t2_callback(new_val) -> None:
        """
        Callback function for the trackbar that runs Canny edge
        detection using the trackbar value as threshold1.
        """
        nonlocal canny_t1, canny_t2, im
        canny_t2 = new_val
        edges = cv2.Canny(im, canny_t1, canny_t2)
        cv2.imshow("Canny Edge Detection", edges)

    canny_t1 = 100
    canny_t2 = 200

    # Create window with trackbar
    cv2.namedWindow("Canny Edge Detection")
    cv2.createTrackbar("T1","Canny Edge Detection",0,500,canny_t1_callback)
    cv2.createTrackbar("T2","Canny Edge Detection",0,500,canny_t2_callback)
    # Show original image in the window with trackbar
    cv2.imshow("Canny Edge Detection", im)

    # Allow trackbar to iterate until q-key is pressed to quit
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


def main():
    im_folder = "media/week06_data/"
    im = cv2.imread(im_folder + "TestIm2.png", cv2.IMREAD_GRAYSCALE)
    canny_trackbar(im)


if __name__ == "__main__":
    main()
