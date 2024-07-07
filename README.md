# Image Similarity Matching using SIFT and RANSAC

This project implements an image similarity matching algorithm using SIFT (Scale-Invariant Feature Transform) and RANSAC (Random Sample Consensus). The code compares pairs of images to determine their similarity score based on matched keypoints and filtered matches.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Function Descriptions](#function-descriptions)
- [Examples](#examples)
- [License](#license)

## Installation

### Prerequisites
- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- SciPy

### Setup
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ImageSimilarityMatching.git
    cd ImageSimilarityMatching
    ```
2. Install the required packages:
    ```sh
    pip install numpy opencv-python matplotlib scipy
    ```

## Usage
1. Ensure your images are stored in the `Test Images` directory or update the paths in the script accordingly.
2. Run the script:
    ```sh
    python image_similarity.py
    ```

## Function Descriptions

### `filter_matches(image1, image2)`
This function takes two images, extracts keypoints and descriptors using SIFT, and applies several filtering techniques to find and display the best matches between the two images.

- **Techniques Used:**
  - Manual CrossCheck
  - D.Lowe's Ratio Test
  - RANSAC for inlier detection
  - Distance Thresholding

- **Returns:**
  - `similarity_score`: A score indicating the similarity between the two images.
  - `result`: A string stating whether the images are similar or not.

### `lows_ratio(des1, des2, bf, d_lows_contant, cross_checked_matches)`
Applies D.Lowe's ratio test to filter matches.

### `distance_threshold(all_matches, inlier_matches)`
Filters matches based on a distance threshold determined by the normal distribution of distances.

### `manual_crossCheck(des1, des2, bf)`
Performs a manual cross-check of matches to filter out inconsistent matches.

### `_ransak(kp1, kp2, all_matches, lowes_filterd_matches)`
Applies the RANSAC algorithm to find inlier matches.

### `_display(gray1, kp1, gray2, kp2, list_of_matches)`
Displays the matched keypoints between two images.

### `_draw_normal_distribution(distances)`
Draws the normal distribution of distances between matches.

## Examples
The script processes a set of image pairs defined in the `images` list and prints the similarity score and result for each pair. It also displays the matched keypoints and the normal distribution of distances.

```python
images = [
    ['Test Images/image1a.jpeg', 'Test Images/image1b.jpeg'],
    ['Test Images/image2a.jpeg', 'Test Images/image2b.jpeg'],
    ['Test Images/image3a.jpeg', 'Test Images/image3b.jpeg'],
    ['Test Images/image4a.jpeg', 'Test Images/image4b.jpeg'],
    ['Test Images/image4b.jpeg', 'Test Images/image4c.png'],
    ['Test Images/image4a.jpeg', 'Test Images/image4c.png'],
    ['Test Images/image5a.jpeg', 'Test Images/image5b.jpeg'],
    ['Test Images/image6a.jpeg', 'Test Images/image6b.jpeg'],
    ['Test Images/image7a.jpeg', 'Test Images/image7b.jpeg'],
    ['Test Images/img1.jpg', 'Test Images/img2.jpg'],
    ['Test Images/box.png', 'Test Images/box_in_scene.png']
]

for i in range(len(images)):
    image1_name = images[i][0]
    image2_name = images[i][1]
    img1 = cv2.imread(image1_name, 0)
    img2 = cv2.imread(image2_name, 0)
    score, result = filter_matches(img1, img2)
    print(image1_name, '&', image2_name, '-->', result)
    print('Similarity Score is: ', score)
    print('------------------------------')
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

 
