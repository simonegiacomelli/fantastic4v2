def cv2_imshow(img, title=None, figsize=(15, 8)):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    if title:
        plt.gcf().suptitle(title, fontsize=20)
    plt.imshow(img)
    plt.show()
