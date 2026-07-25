import cv2


def process_image(path):

    img = cv2.imread(path)

    if img is None:
        print("Image could not be loaded")
        return None

    height, width, channels = img.shape

    print("Image Width:", width)
    print("Image Height:", height)
    print("Channels:", channels)

    return width, height, channels