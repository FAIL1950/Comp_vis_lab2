import cv2
import numpy as np

from func import getArea, showCentroidAndAxisAndEccentricity, VossEdgeTracing, getCurvature


def convert_to_grayscale(image_path: str)-> np.ndarray | None:
    image: np.ndarray = cv2.imread(image_path)

    if image is None:
        print("Ошибка: не удалось загрузить изображение.")
        return None

    res_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return res_image


def test(choose: int):
    gray_image = convert_to_grayscale('img.png')

    if gray_image is not None:
        cv2.imshow('Gray Image', gray_image)
        if(choose == 1):            #Площа
            print("=== Коло ===\n")
            print("Площа ділянки:", getArea(gray_image, (500, 300)))
        elif (choose == 2):         #центроїд, головна вісь та ексцентриситет
            print("=== Коло ===\n")
            showCentroidAndAxisAndEccentricity(gray_image, (500, 300))
            #print("=== Лінія ===\n")
            #showCentroidAndAxisAndEccentricity(gray_image, (883, 147))
            #print("=== Еліпс ===\n")
            #showCentroidAndAxisAndEccentricity(gray_image, (1048, 132))
        elif (choose == 3):
            edge = VossEdgeTracing(gray_image, (1151, 251), True)
            cv2.imshow('Edge', edge)
            # 356, 173
            # 986, 125
            # 993, 85
            # 951, 131
            # 1151, 251
        elif (choose == 4):
            curv = getCurvature(gray_image, VossEdgeTracing(gray_image, (356, 173), False), 50)
            cv2.imshow('Curv', curv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    test(2)