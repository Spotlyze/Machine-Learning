# SPOTLYZE - MACHINE LEARNING PART
Spotlyze is an innovative application designed to analyze facial skin conditions such as acne, dark circles, normal, and wrinkles using machine learning. It provides personalized skincare product recommendations based on the user's skin issues. This application leverages TensorFlow's VGG16 transfer learning model for image classification and uses Nearest Neighbors and Cosine Similarity algorithms to suggest the most suitable skincare products from local brands, ensuring the recommendations are tailored to each user's needs.
## Key Features
- Facial Skin Condition Classification: Identifies acne, dark circles, and wrinkles with VGG16.
- Skincare Recommendation System: Suggests local skincare based on skin issues using Nearest Neighbors and Cosine Similarity.
- Data Augmentation: Improves model accuracy with ImageDataGenerator.
## Dataset
The dataset used for training the model consists of labeled images of facial skin conditions, including acne, dark circles, and wrinkles. These images were collected and stored in Google Drive, and the dataset is divided into training and validation sets for effective model training and testing.
- Link Dataset: https://drive.google.com/drive/folders/1TyLOGppiMQSrX2KYf5nbBdhl-UiWtn-B?usp=drive_link 
## Requirements
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-learn
## Documentation
- Research skin conditions and detection methods.
- Collect labeled facial images of acne, dark circles, normal and wrinkles.
- Split the dataset (80% training, 20% validation) and augment with ImageDataGenerator.
- Build and fine-tune a VGG16 model for skin condition classification.
- Train the model, monitor accuracy, and evaluate it on the validation set.
- Test the model with new, unseen images to assess classification performance.
- Use Nearest Neighbors and Cosine Similarity for personalized skincare recommendations.
## Results
- Classification Model: Achieved 94.69% accuracy and a loss of 0.1445 on the validation dataset. 
![alt text](https://github.com/Spotlyze/Spotlyze-Machine-Learning/blob/main/Model/classification_result.jpg)

- Skincare Recommendation: Provided personalized skincare suggestions tailored to user skin types and concerns.
![alt text](https://github.com/Spotlyze/Spotlyze-Machine-Learning/blob/main/Model/skincare_recommendation.jpg)
![alt text](https://github.com/Spotlyze/Spotlyze-Machine-Learning/blob/main/Model/skincare_recommendation_result.jpg )
