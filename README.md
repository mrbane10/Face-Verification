# FaceNet - Face Verification and Face Recognition

---

## **Introduction**
This project implements a **Face Recognition System** using the ideas proposed in the **FaceNet** paper. The system performs:
1. **Face Verification**: Verifies if a given image matches the identity of a person.
2. **Face Recognition**: Identifies the person in an image from a database of known individuals.

The core of this system relies on a pre-trained Inception-ResNet model to generate **128-dimensional embeddings** for facial images, which are used for comparison.

---

## **Dependencies**
Ensure the following libraries are installed:
- Python 3.11
- TensorFlow (Keras API)
- NumPy
- OpenCV (optional for image handling)
- PIL (Python Imaging Library)
- Pandas
- Matplotlib

---

## **Setup Instructions**
1. **Pre-trained Model**:
   - Download the pre-trained weights file (`model.h5`) and place it in the appropriate directory (`/path/to/your/model.h5`).

2. **Database of Images**:
   - Create a directory containing images of known individuals for the database (e.g., `images/`).
   - The file names should correspond to the person's identity (e.g., `danielle.png`, `younes.jpg`).

3. **Install Required Libraries**:
   - Use `pip install tensorflow numpy pandas pillow opencv-python` to install the required packages.

---

## **Core Features**
1. **Face Embeddings**:
   - The pre-trained Inception-ResNet model generates a **128-dimensional embedding** for each face image.
   - These embeddings represent the unique characteristics of a person's face.

2. **Triplet Loss**:
   - Implements the triplet loss function, which minimizes the distance between:
     - The anchor and positive embeddings (same person).
     - Maximizes the distance between the anchor and negative embeddings (different people).

3. **Face Verification**:
   - Verifies if a person in the given image matches a specific identity in the database by computing the L2 distance between embeddings.

4. **Face Recognition**:
   - Identifies the person in an image by finding the closest embedding match in the database.

---

## **Key Functions**
1. **`img_to_encoding(image_path, model)`**:
   - Converts an image into a 128-dimensional embedding using the pre-trained model.

2. **`verify(image_path, identity, database, model)`**:
   - Verifies if the person in the image matches the specified identity in the database.

3. **`who_is_it(image_path, database, model)`**:
   - Identifies the person in the image by comparing embeddings to those in the database.

---

## **How to Use**
1. **Load the Pre-trained Model**:
   ```python
   model = inception_resnet_v1.InceptionResNetV1()
   model.load_weights('path/to/model.h5')
   ```

2. **Add Individuals to the Database**:
   ```python
   database["danielle"] = img_to_encoding("images/danielle.png", model)
   database["younes"] = img_to_encoding("images/younes.jpg", model)
   ```

3. **Verify an Identity**:
   ```python
   dist, door_open = verify("images/test_image.jpg", "danielle", database, model)
   ```

4. **Recognize a Person**:
   ```python
   min_dist, identity = who_is_it("images/test_image.jpg", database, model)
   ```

---

## **Examples**
1. **Face Verification**:
   ```python
   verify("images/camera_1.jpg", "younes", database, model)
   ```

2. **Face Recognition**:
   ```python
   who_is_it("images/camera_1.jpg", database, model)
   ```

---

## **Results**
- The system outputs the predicted identity and the distance between the embeddings.
- If the distance is less than 0.7, the face is considered a match.

---

## **Limitations**
- Performance depends on the quality of the images in the database.
- Requires the face to be properly aligned and centered in the image.

---

## **References**
- FaceNet: A Unified Embedding for Face Recognition and Clustering ([Google Research](https://arxiv.org/abs/1503.03832))
- Inception-ResNet Architecture ([Paper](https://arxiv.org/abs/1602.07261))

---

Feel free to reach out for additional guidance or clarifications!
