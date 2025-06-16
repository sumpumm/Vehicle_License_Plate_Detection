from ultralytics import YOLO

# Load the trained model
model = YOLO(r"runs\detect\train\weights\best.pt")

# Define image path
image_path = r"vehicles\Tata-Nexon-EV.jpg"

# Run inference
results = model(image_path)

# Optional: display results
results[0].show()  # Show the image with bounding boxes
# or
results[0].save(filename="output.jpg")  # Save the result to a file
