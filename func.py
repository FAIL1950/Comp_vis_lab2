import cv2
import numpy as np
from collections import deque
import math



def getRegion(input: np.ndarray, start_point: tuple)-> np.ndarray:
    height, width = input.shape[:2]
    x_start, y_start = start_point
    start_color = input[y_start, x_start]

    mask = np.zeros((height, width), dtype=np.uint8)
    visited = np.zeros((height, width), dtype=bool)

    queue = deque([(x_start, y_start)])
    visited[y_start, x_start] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        mask[y, x] = 255

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < width and 0 <= ny < height:
                if not visited[ny, nx]:
                    neighbor_color = input[ny, nx]
                    if neighbor_color == start_color:
                        queue.append((nx, ny))
                        visited[ny, nx] = True

    return mask

def getArea(input: np.ndarray, start_point: tuple)-> int:
    region: np.ndarray = getRegion(input, start_point)
    height, width = region.shape[:2]
    area = 0
    for i in range(height):
        for j in range(width):
            if region[i, j] == 255:
                area += 1
    return area

def getCentroid(input: np.ndarray,region: np.ndarray, show: bool) -> tuple | None | np.ndarray:
    height, width = region.shape[:2]
    m10 = np.int64(0)
    m01 = np.int64(0)
    m00 = np.int64(0)

    for i in range(height):
        for j in range(width):
            if region[i, j] == 255:
                I = np.int32(input[i, j])
                m10 += j * I
                m01 += i * I
                m00 += I
    if m00 == 0:
        return None

    xs = int(m10 / m00)
    ys = int(m01 / m00)
    if show:
        color_image = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        cv2.circle(color_image, (xs, ys), 5, (0, 0, 255), 5)
        return color_image
    return xs, ys

def getCentralMoment(input: np.ndarray, region: np.ndarray, xs, ys, a: int, b: int) -> int:
    height, width = region.shape[:2]
    m = np.int64(0)
    for i in range(height):
        for j in range(width):
            if region[i, j] == 255:
                I = np.int32(input[i, j])
                m+= pow(j-xs, a) * pow(i-ys, b) * I
    return m

def getMainAxis(input: np.ndarray, region: np.ndarray, xs, ys, show: bool):
    mu11 = getCentralMoment(input, region, xs,ys,1, 1)
    mu20 = getCentralMoment(input, region, xs,ys,2, 0)
    mu02 = getCentralMoment(input, region, xs,ys,0, 2)
    denominator = mu20 - mu02
    if denominator == 0:
        print("Помилка: Ділення на нуль при обрахуванні кута.")
        return None
    tmp = np.float64(2 * mu11 / denominator)
    theta = math.atan(tmp) / 2
    if show:
        length = 250
        end_x1 = int(xs + length * np.cos(theta))
        end_y1 = int(ys + length * np.sin(theta))
        end_x2 = int(xs - length * np.cos(theta))
        end_y2 = int(ys - length * np.sin(theta))
        color_image = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        cv2.line(color_image, (end_x1, end_y1), (end_x2, end_y2), (0, 255, 0), 2)
        return color_image
    return theta

def showCentroidAndAxisAndEccentricity(input: np.ndarray, start_point: tuple):
    region: np.ndarray = getRegion(input, start_point)
    xs, ys = getCentroid(input, region, False)
    res1 = getCentroid(input, region, True)
    res2 = getMainAxis(input, region, xs, ys, True)
    res = cv2.addWeighted(res1, 0.5, res2, 0.5, 0)
    print("Центроїд ділянки:", xs,ys)
    print("Theta головної вісі: ", getMainAxis(input, region, xs, ys, False))
    print("Ексцентриситет: ", getEccentricity(input, region, xs, ys))
    cv2.imshow('res', res)
    cv2.waitKey(0)

def getEccentricity(input: np.ndarray, region: np.ndarray, xs, ys):
    mu11 = getCentralMoment(input, region, xs, ys, 1, 1)
    mu20 = getCentralMoment(input, region, xs, ys, 2, 0)
    mu02 = getCentralMoment(input, region, xs, ys, 0, 2)
    difpow = pow(mu20 - mu02, 2)
    sumpow = pow(mu20 + mu02, 2)
    eccentricity = float(difpow - 4* pow(mu11,2)) / sumpow
    return eccentricity

def VossEdgeTracing(input: np.ndarray, start_point: tuple, show: bool) -> None | deque | np.ndarray:
    height, width = input.shape[:2]
    x_start, y_start = start_point
    start_color = input[y_start, x_start]
    directions = [(0, -1), (1,0), (0,1), (-1,0)]
    mult = 4
    q = None
    n = -1
    for dx, dy in directions:
        n += 1
        nx, ny = x_start + dx, y_start + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbor_color = input[ny, nx]
            if neighbor_color != start_color:
                q=(nx,ny)
                break
        else:
            q = (nx, ny)
            break
    if q is None:
        print("Помилка: Початковий піксель не є граничним.")
        return None
    begin = (q, (x_start, y_start))
    edge_points  = deque([(x_start, y_start)])
    k = getNewK( n,mult)
    qk = (-1,-1)
    pi = (x_start, y_start)
    #qx, qy = -1,-1
    while (qk, pi) != begin:
        qk = getQk(pi, k, directions)
        qx, qy = qk
        while 0 <= qx < width and 0 <= qy < height and input[qy, qx] == start_color:
            pi = qk
            #if pi == (1151, 251):
            #    print()
            edge_points.append(pi)
            k = invertK(k, mult)
            qk = getQk(pi, k, directions)
            qx, qy = qk
        k = getNewK(k,4)


    edge_points.pop()
    if show:
        color_image = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        for nx, ny in edge_points:
            color_image[ny, nx] = (0, 0, 255)
            print("Довжина границі: ", len(edge_points))
        return color_image
    return edge_points


def getNewK( k: int, mult: int) -> int:
    return (k+1) % mult

def invertK(k: int, mult: int) -> int:
    tmp = -1
    if k == 0:
        tmp = 2
    elif k == 1:
        tmp = 3
    elif k == 2:
        tmp = 0
    elif k == 3:
        tmp = 1
    return getNewK( tmp, mult)

def getQk(start_point: tuple, k: int, directions: list) -> tuple:
    x_start, y_start = start_point
    dx, dy = directions[k]
    nx, ny = x_start + dx, y_start + dy
    return nx, ny

def getCurvature(input: np.ndarray, edge_points: deque, k: int) -> np.ndarray | None:
    if k < 1:
        print("Помилка: Параметр k має бути >= 1. Спробуйте ввести інше значення.")
        return None
    if len(edge_points) < k:
        print("Помилка: Недостатньо точок для обчислення кривизни.")
        return None

    curvature_angles = []
    maxIndx = 0
    minIndx = 0
    maxCurv = None
    minCurv = None
    for i in range(len(edge_points)):
        x0, y0 = edge_points[i]
        x1, y1 = edge_points[(i + k) % len(edge_points)]
        x_1, y_1 = edge_points[i-k]

        a0 = x0
        b0 = y0

        a2 = np.float64(x1 + x_1 - 2*x0)/2
        b2 = np.float64(y1 + y_1 - 2*y0)/2

        a1 = np.float64(x1 - a0 - a2)
        b1 = np.float64(y1 - b0 - b2)
        ki = np.float64(2*(a1*b2 - b1*a2))/ pow(a1*a1+b1*b1,1.5)
        ki = abs(ki)
        curvature_angles.append(ki)

        if maxCurv == None:
            maxCurv = ki
            minCurv = ki
        elif ki > maxCurv:
            maxCurv = ki
            minIndx = i
        elif ki < minCurv:
            minCurv = ki
            maxIndx = i
    sum = 0
    for i in range(len(curvature_angles)):
        sum += curvature_angles[i]

    sum /= len(curvature_angles)

    print("Максимальна кривизна: ", maxCurv)
    print("Мінімальна кривизна: ", minCurv)
    print("Середнє значення кривизни: ", sum)

    color_image = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_image, edge_points[minIndx], 5, (255, 0, 0), 5)
    cv2.circle(color_image, edge_points[maxIndx], 5, (0, 0, 255), 5)
    return color_image

















