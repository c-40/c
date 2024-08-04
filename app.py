import streamlit as st
import cv2
import numpy as np
import base64
from PIL import Image
import io

class ImageSeg:
    def __init__(self, img, threshold):
        self.img = img
        self.threshold = threshold

    def color_filter(self):
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([30, 40, 40])
        upper_bound = np.array([90, 255, 255])
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        filtered_img = cv2.bitwise_and(self.img, self.img, mask=mask)
        return filtered_img

    def preprocess_img(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        _, thresh_img = cv2.threshold(blurred_img, self.threshold, 255, cv2.THRESH_BINARY)
        return thresh_img

    def post_process(self, thresh_img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
        opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel)
        return opened_img

    def count_trees(self):
        filtered_img = self.color_filter()
        thresh_img = self.preprocess_img(filtered_img)
        processed_img = self.post_process(thresh_img)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
        return num_labels - 1

    def mark_trees(self):
        filtered_img = self.color_filter()
        thresh_img = self.preprocess_img(filtered_img)
        processed_img = self.post_process(thresh_img)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
        marked_img = np.copy(self.img)

        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return marked_img

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

st.title('Tree Counting and Marking App')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Process image
    final_count = 0
    best_threshold = 0

    for thresh in range(0, 100, 5):
        obj = ImageSeg(img, thresh)
        count = obj.count_trees()
        if count > final_count:
            final_count = count
            best_threshold = thresh

    final_obj = ImageSeg(img, best_threshold)
    marked_img = final_obj.mark_trees()
    marked_img_base64 = image_to_base64(marked_img)

    st.subheader('Tree Count:')
    st.write(final_count)

    st.subheader('Marked Image:')
    st.image(io.BytesIO(base64.b64decode(marked_img_base64)), channels="BGR")
