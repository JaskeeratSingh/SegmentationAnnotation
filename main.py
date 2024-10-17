import os
import cv2
import numpy as np
import argparse
from numpy.lib.stride_tricks import as_strided
# Global variables
drawing = False
polygon_points = []
image_files = []
masks = []
current_index = 0
image = None
mask = None
zoom_factor = 1.0  # Start with full image size
zoom_center = (450, 450)
# Constants
WINDOW_SIZE = 900
transformation = (1.0, 0, 0)
x_offset = 0
y_offset = 0
scale_factor = 0
center_coordinates = []
original_points = []
original_masks = []
centers_size = 0


def load_image(index):
    global image, mask, scale_factor
    if 0 <= index < len(image_files):
        image_path = image_files[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        width, height = image.shape
        scale_factor = width/WINDOW_SIZE
        mask = np.zeros_like(image)
    else:
        raise Exception("Index out of range.")


def display_image():
    global image, zoom_factor, zoom_center, masks, x_offset, y_offset

    if image is None:
        return
    new_width = int(WINDOW_SIZE * zoom_factor)
    new_height = int(WINDOW_SIZE * zoom_factor)
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    cropped_image = resized_image[y_offset:(
        y_offset + WINDOW_SIZE), x_offset:(x_offset + WINDOW_SIZE)]

    # Draw masks
    for mask_points in masks:
        cv2.polylines(cropped_image, [np.array(Transforms(mask_points, transformation)).astype(np.int32)],
                      isClosed=True, color=(255, 0, 0), thickness=2)
    if polygon_points:
        cv2.polylines(cropped_image, [np.array(
            polygon_points)], isClosed=False, color=(255, 0, 0), thickness=2)

    cv2.imshow('Annotation Tool', cropped_image)


def applyInverse(point, transform):
    point_x = point[0]
    point_y = point[1]
    scale, x_offset, y_offset = transform
    point_x = (point_x + x_offset) / scale
    point_y = (point_y + y_offset) / scale
    return [point_x, point_y]


def applyTransform(point, transform):
    point_x = point[0]
    point_y = point[1]
    scale, x_offset, y_offset = transform
    point_x = (point_x*scale) - x_offset
    point_y = (point_y*scale) - y_offset
    return [int(point_x), int(point_y)]


def inverseTransforms(points, transform):
    normalized_points = []
    for i in points:
        normalized_points.append(applyInverse(i, transform))

    return normalized_points


def Transforms(points, transform):
    normalized_points = []
    for i in points:
        normalized_points.append(applyTransform(i, transform))

    return normalized_points


def mouse_handling(event, x, y, flags, param):
    global drawing, polygon_points, zoom_center, masks, zoom_factor, x_offset, y_offset, transformation, scale_factor, original_masks

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        polygon_points = [(int(x), int(y))]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        polygon_points = inverseTransforms(polygon_points, transformation)
        original_transformation = (scale_factor, 0, 0)
        original_points = Transforms(
            polygon_points, original_transformation)
        original_masks.append(original_points)
        masks.append(polygon_points)
        polygon_points = []

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            polygon_points.append((int(x), int(y)))

    if event == cv2.EVENT_MOUSEWHEEL:
        zoom_center = (x, y)
        mouse_x = zoom_center[0]
        mouse_y = zoom_center[1]
        if flags > 0:  # Scroll up
            if (zoom_factor * 2 <= 4.0):
                zoom_factor *= 2

        else:  # Scroll down
            if (zoom_factor * 0.5 >= 1.0):
                zoom_factor *= 0.5
        new_width = int(WINDOW_SIZE * zoom_factor)
        new_height = int(WINDOW_SIZE * zoom_factor)
        x_offset = int((mouse_x / WINDOW_SIZE) *
                       (new_width - WINDOW_SIZE))
        y_offset = int((mouse_y / WINDOW_SIZE) *
                       (new_height - WINDOW_SIZE))
        transformation = (zoom_factor, x_offset, y_offset)
        load_image(current_index)  # Reload original image for resizing
        display_image()  # Display the updated image


def AnnotationTool(image_directory):
    global current_index, masks, centers_size

    # Load all image files from the directory
    global image_files
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]

    if not image_files:
        raise Exception("No images found in the specified directory.")

    load_image(current_index)
    cv2.namedWindow('Annotation Tool')
    cv2.setMouseCallback('Annotation Tool', mouse_handling)

    while True:
        display_image()
        key = cv2.waitKey(1)

        if key == ord('d'):  # Next image
            if current_index < len(image_files) - 1:
                current_index += 1
                load_image(current_index)

        elif key == ord('a'):  # Previous image
            if current_index > 0:
                current_index -= 1
                load_image(current_index)

        elif key == ord('c'):  # Clear mask
            if len(masks) > 0:
                masks.pop()
                original_masks.pop()
                load_image(current_index)

        elif key == ord('s'):  # Save mask
            print("Saving...")
            cropped_images, masks = extract_cropped_images_and_masks(
                original_masks, image)
            save_to_npz(cropped_images, masks, "Contours_finetuning",
                        os.path.splitext(image_files[current_index])[0] + ".npz")
            centers = calculate_centroids(original_masks)
            centers_mask_whole = create_mask_with_dilations(
                centers, image.shape[0])
            mask_tiles, _, _ = tiles_from_matrix(
                np.expand_dims(centers_mask_whole, -1), centers_size, overlap=64)
            image_tiles, _, _ = tiles_from_matrix(
                np.expand_dims(image, -1), centers_size, overlap=64)
            non_zero_mask = np.any(mask_tiles != 0, axis=(1, 2, 3))
            filtered_tiles = image_tiles[non_zero_mask]
            filtered_masks = mask_tiles[non_zero_mask]
            filtered_tiles_squeezed = np.squeeze(filtered_tiles)
            filtered_masks_squeezed = np.squeeze(filtered_masks)
            save_to_npz(filtered_tiles_squeezed, filtered_masks_squeezed, "Centers_finetuning",
                        os.path.splitext(image_files[current_index])[0] + ".npz")
            # for i in range(len(original_masks)):
            #     cv2.fillPoly(
            #         mask, [np.array(original_masks[i]).astype(np.int32)], (255, 0, 0))
            # rename_filename = os.path.splitext(image_files[current_index])[
            #     0] + '_binary_mask.png'
            # cv2.imwrite(rename_filename, mask)
            print("Saved :)")

        elif key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()


def tiles_from_matrix(
    image_matrix: np.ndarray,  # HWC
    tile_size: int = 128,
    overlap: int = 64
) -> np.ndarray:

    if len(image_matrix.shape) < 3:
        image_matrix = np.expand_dims(image_matrix, axis=2)
    matrix_Y, matrix_X, channels = image_matrix.shape

    no_overlap_size = tile_size - overlap
    num_X_tiles = int(np.floor((matrix_X - tile_size) / no_overlap_size) + 1)
    num_Y_tiles = int(np.floor((matrix_Y - tile_size) / no_overlap_size) + 1)

    # >>> Using optimized sliding to generate tiles <<< #
    strides = image_matrix.strides
    # >>> Get number of bytes to move to get to next element in matrix <<< #
    stride_y, stride_x, stride_c = strides

    # >>> Calculate shape for the strided view <<< #
    new_shape = (num_X_tiles, num_Y_tiles, tile_size, tile_size, channels)
    new_strides = (no_overlap_size * stride_x, no_overlap_size *
                   stride_y, stride_y, stride_x, stride_c)

    # >>> Create the tiles in new_shape <<< #
    strided_matrix = as_strided(
        image_matrix, shape=new_shape, strides=new_strides)

    # >>> Reshape to have num_Y_tiles * num_X_tiles tiles<<< #
    tiled_matrix = strided_matrix.reshape(-1, tile_size, tile_size, channels)
    return tiled_matrix, num_X_tiles, num_Y_tiles


def save_to_npz(cropped_images, masks, output_dir, filename):
    # Create directory if it doesn't exist
    filename = filename.split("\\")[-1]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    print(output_path)
    np.savez(output_path, images=cropped_images, masks=masks)
    print(f'Saved to {output_path}')


def calculate_centroids(polygons):
    centroids = []

    for points in polygons:
        points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        M = cv2.moments(points_array)

        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroid = (cx, cy)
        else:
            centroid = None

        centroids.append(centroid)

    return centroids


def crop_around_coord(image, center, padding_val=255, size=128):
    tl_x = int(center[0] - 64)
    tl_y = int(center[1] - 64)

    if tl_x < 0:
        image = np.pad(image, ((0, 0), (abs(tl_x), 0)),
                       constant_values=padding_val)
        tl_x = 0
    if tl_y < 0:
        image = np.pad(image, ((abs(tl_y), 0), (0, 0)),
                       constant_values=padding_val)
        tl_y = 0

    crop = image[tl_y:tl_y+size, tl_x:tl_x+size]
    crop = np.pad(crop, ((
        0, size - crop.shape[0]), (0, size - crop.shape[1])), constant_values=padding_val)
    return crop


def extract_cropped_images_and_masks(polygons, image):
    centroids = calculate_centroids(polygons)
    cropped_images = []
    masks = []

    for points, center in zip(polygons, centroids):
        if center is None:
            continue  # Skip if centroid calculation failed

        # Crop the image around the centroid
        cropped_image = crop_around_coord(image, center)

        # Create a mask
        mask = np.zeros_like(image, dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        # Fill the polygon in the mask
        cv2.fillPoly(mask, [points_array], (1))

        # Crop the mask to the same region as the cropped image
        cropped_mask = crop_around_coord(mask, center)

        # Append to results
        cropped_images.append(cropped_image)
        masks.append(cropped_mask)

    return cropped_images, masks


def create_mask_with_dilations(centers, centers_size):
    # Initialize the mask with zeros
    mask = np.zeros((centers_size, centers_size), dtype=np.uint8)

    # Define the "+" shape kernel
    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.uint8)

    # Process each center
    for center in centers:
        x, y = int(round(center[0])), int(round(center[1]))
        # Set the center location to 1
        mask[y, x] = 1

    # Apply dilation
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    return dilated_mask


def crop_around_coord(image, center, padding_val=255, size=128):
    tl_x = int(center[0] - 64)
    tl_y = int(center[1] - 64)

    # Pad the top and left if the crop goes out of the image
    if tl_x < 0:
        image = np.pad(image, ((0, 0), (abs(tl_x), 0)),
                       constant_values=padding_val)
        tl_x = 0
    if tl_y < 0:
        image = np.pad(image, ((abs(tl_y), 0), (0, 0)),
                       constant_values=padding_val)
        tl_y = 0

    # Crop the image
    crop = image[tl_y:tl_y+size, tl_x:tl_x+size]

    # Pad the bottom and right if needed
    crop = np.pad(crop, ((
        0, size - crop.shape[0]), (0, size - crop.shape[1])), constant_values=padding_val)
    return crop


def calculate_centroids(polygons):
    centroids = []

    for points in polygons:
        # Convert the list of points to a NumPy array and reshape it
        points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        # Calculate moments
        M = cv2.moments(points_array)

        # Calculate centroid
        if M['m00'] != 0:  # Avoid division by zero
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroid = (cx, cy)
        else:
            centroid = None  # Handle case where area is zero

        centroids.append(centroid)

    return centroids


def main() -> None:
    global centers_size
    parser = argparse.ArgumentParser(
        description="This is a tool for image segmentation annotation.")
    parser.add_argument("-d", "--INPUT_DATA_DIR",
                        help="path/to/your/input_directory/", type=str, required=True)
    parser.add_argument("--centers_size",
                        help="Size of the centers (default: 128)", type=int, default=128)
    parser.add_argument("--contours_size",
                        help="Size of the contours (default: 128)", type=int, default=128)
    args = parser.parse_args()
    parser = argparse.ArgumentParser()
    centers_size = args.centers_size
    contours_size = args.contours_size
    AnnotationTool(args.INPUT_DATA_DIR)


if __name__ == '__main__':
    main()
