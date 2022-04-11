import os
import cv2
from pathlib import Path




training_set = './test_set_all'


correct_path = training_set + '/correct_class'
wrong_path = training_set + '/wrong_class'
reject_path = training_set + '/reject_class'
Path(correct_path).mkdir(parents=True, exist_ok=True)
Path(wrong_path).mkdir(parents=True, exist_ok=True)
Path(reject_path).mkdir(parents=True, exist_ok=True)


for num, image_name in enumerate(os.listdir(training_set)):

    if image_name.endswith(".jpg"):
        full_path = os.path.join(training_set, image_name)
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        cv2.imshow('cropped_weed', img)
        key = cv2.waitKey(0)
        if key & 0xFF == ord("y"):
            img_correct_path = str(Path(correct_path) / image_name)
            cv2.imwrite(img_correct_path, img)
            print("saved image {0} in: {1}".format(num, img_correct_path))
        elif key & 0xFF == ord("n"):
            img_wrong_path = str(Path(wrong_path) / image_name)
            cv2.imwrite(img_wrong_path, img)
            print("saved image {} in: {}".format(num, img_wrong_path))
        elif key & 0xFF == ord("r"):
            img_reject_path = str(Path(reject_path) / image_name)
            cv2.imwrite(img_reject_path, img)
            print("saved image {} in: {}".format(num, img_reject_path))
        elif key & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit(0)
        cv2.destroyAllWindows()