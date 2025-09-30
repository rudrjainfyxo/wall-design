import cv2
from midas_depth import estimate_depth, load_midas_model
from wall_mask_midas import get_wall_mask_midas

def process_frame(image, model, transform, device):
    depth = estimate_depth(image, model, transform, device)
    mask = get_wall_mask_midas(image, depth)
    output = image.copy()
    output[mask == 255] = (0, 255, 0)
    return output

def main(image_path):
    model, transform, device = load_midas_model()
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"❌ Could not load image at: {image_path}")

    result = process_frame(image, model, transform, device)
    cv2.imwrite("output.png", result)
    print("✅ Output saved to output.png")
    cv2.imshow("Wall Mask Output", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("1.jpg")  # Change to your test image path
