import cv2
import numpy as np
import os
import re
import importlib
import csv

def dice_coefficient(prediction, ground_truth):
  """
  Calculates the Dice coefficient between two segmentation masks.

  Args:
    prediction: The predicted segmentation mask.
    ground_truth: The ground truth segmentation mask.

  Returns:
    The Dice coefficient.
  """

  intersection = np.sum(prediction & ground_truth)
  sum_prediction = np.sum(prediction)
  sum_ground_truth = np.sum(ground_truth)

  return (2 * intersection) / (sum_prediction + sum_ground_truth)


def test_student_function(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
        student_function = getattr(module, function_name)
    except Exception as e:
        return [((), "", f"Error importing function: {e}", False)]
    subdirectory_path = os.path.join(os.getcwd(), "test")

    test_images = [filename for filename in os.listdir(subdirectory_path) if filename.endswith(".jpg")]
    total_score = 0.0
    is_passed = False
    i =0
    for test_image_name in test_images:
        i+=1
        try:
            test_image_path = os.path.join(os.path.join(os.getcwd(),'test'),test_image_name)
            output_image = student_function(test_image_path)
            ground_truth_image=cv2.imread(os.path.join(os.path.join(os.getcwd(),'ground truth'),test_image_name))
            # print(ground_truth_image)
            score = dice_coefficient(output_image, ground_truth_image)
            print(score)
            total_score += score
            is_passed=True

        except Exception as e:
            result = f"Error: {e}"
    if is_passed:
        return total_score
    return 0


if __name__ == "__main__":
    student_files = [filename for filename in os.listdir() if filename.endswith(".py") and filename != "tester.py"]
    with open(f"marks.csv","w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        
        csv_writer.writerow(["Roll Number", "Score"])
        for student_file in student_files:
            try:
                match = re.match(r"Q1_(\d+).py", student_file)   
                if match:
                    roll_number = match.groups()[0]
                    results = test_student_function(student_file[:-3], "solution")
                    csv_writer.writerow([ roll_number,results])
            except Exception as e:
                print(f"Error testing {student_file}: {e}")
