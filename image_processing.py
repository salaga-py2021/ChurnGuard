from PIL import Image
import io

def process_image(image_path, size):
    """
    Resize an image to a new size.
    """

    # Load the image
    current_image = Image.open(image_path)

    # Resize the image
    resized_image = current_image.resize((546, 219))  # Adjust the size as needed

    # Convert the image to bytes
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr


