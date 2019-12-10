import cv2
import json
import numpy as np
import imageio


def draw(data, ratio=4):
    img = np.ones((data['max_y']//ratio+20, data['max_x']//ratio+20, 3)) * 255
    all_steps = [img]

    for stroke in data['strokes']:
        stroke['x'] = [10+x//ratio for x in stroke['x']]
        stroke['y'] = [10+x//ratio for x in stroke['y']]

        points = list(zip(stroke['x'], stroke['y']))
        if points:
            for i, start, end in zip(range(len(points)), points[:-1], points[1:]):
                cv2.line(img, start, end, color=(0, 0, 0), thickness=10)
                if i%10 == 1:
                    all_steps.append(img.copy())

    return img, all_steps


if __name__ == "__main__":

    path = "../input/jsons/a01-000u-01.json"

    with open(path, "r") as f:
        data = json.load(f)

    print(data['text'])

    img, all_steps = draw(data)

    cv2.imwrite("img.png", img)
    imageio.mimsave('img.gif', all_steps)