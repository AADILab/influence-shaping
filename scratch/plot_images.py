import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np

def plot_image_on_axes(image_path, x, y, ax=None, zoom=1.0, rotation=0, bgcolor="white"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
    ax.set_facecolor(bgcolor)
    img = Image.open(image_path)
    if rotation != 0:
        img = img.rotate(rotation, expand=True)
    img = np.array(img)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)
    return ax

if __name__ == "__main__":
    rover_image_path = "/Users/ever/Documents/Papers/ACM TELO (Submitted 2025)/new_assets/rover.png"
    drone_image_path = "/Users/ever/Documents/Papers/ACM TELO (Submitted 2025)/new_assets/drone.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_facecolor("lightgray")

    # Plot rover
    plot_image_on_axes(rover_image_path, 50, 50, ax=ax, zoom=0.05, rotation=45)
    # Plot drone in a different area (e.g., top right)
    plot_image_on_axes(drone_image_path, 40, 40, ax=ax, zoom=0.05, rotation=0)

    plt.show()
