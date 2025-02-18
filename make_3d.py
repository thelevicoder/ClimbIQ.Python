import numpy as np
import cv2
import joblib
from contour import return_contours

def contours_to_3d(contours):
    contours_3d = []
    for contour in contours:
        contour_3d = np.array([[point[0][0], point[0][1], 0] for point in contour])
        contours_3d.append(contour_3d)
    print("Contours converted to 3D.")
    return contours_3d

def evaluate_hold(contour_3d, model):
    # Flatten the 3D contour to a 1D feature vector
    feature_vector = contour_3d.flatten()
    feature_vector = feature_vector.reshape(1, -1)
    prediction = model.predict(feature_vector)
    print(f"Evaluated hold with prediction: {prediction[0]}")
    return prediction[0]

def main():
    contours = return_contours()
    if not contours:
        print("No contours found.")
        return

    contours_3d = contours_to_3d(contours)
    model = joblib.load("hold_evaluation_model.pkl")
    print("Model loaded.")
    evaluations = [evaluate_hold(contour_3d, model) for contour_3d in contours_3d]

    for i, evaluation in enumerate(evaluations):
        print(f"Contour {i} evaluation: {evaluation}/10")

if __name__ == "__main__":
    main()
