import matplotlib.pyplot as plt
import matplotlib.patches as patches
import base64
from io import BytesIO
from PIL import Image


def convert_PIL_to_base64(image: Image, format="jpeg"):
    buffer = BytesIO()
    # Save the image to this buffer in the specified format
    image.save(buffer, format=format)
    # Get binary data from the buffer
    image_bytes = buffer.getvalue()
    # Encode binary data to Base64
    base64_encoded = base64.b64encode(image_bytes)
    # Convert Base64 bytes to string (optional)
    return base64_encoded.decode("utf-8")


def resize_maintain_aspect(image: Image, new_width: int):
    old_w, old_h = image.size
    ratio = new_width / old_w
    new_height = int(old_h * ratio)

    return image.resize((new_width, new_height))


def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(data["bboxes"], data["labels"]):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        plt.text(
            x1,
            y1,
            label,
            color="white",
            fontsize=8,
            bbox=dict(facecolor="red", alpha=0.5),
        )
    ax.axis("off")
    return fig
