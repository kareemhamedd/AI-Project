Data Acquisition & Preprocessing Summary
member 1 task:
By: Kareem Hamed

As the Data Acquisition & Preprocessing Lead, I prepared all datasets required for model training. My role included collecting the datasets, organizing them, building the preprocessing pipeline, and producing the final processed data used by the training and evaluation team.

1. Dataset Acquisition

I collected the LFW dataset and the Mini-VGGFace2 dataset from Kaggle.
The original VGGFace2 dataset is extremely large and unsuitable for local development, so I used the Mini-VGGFace2 version instead, which provides the same structure but with reduced size.
Both datasets were organized inside the raw data directory. Only the needed subsets were extracted to be processed.

2. Preprocessing Pipeline

I implemented a complete preprocessing workflow consisting of three main stages.

Image Resizing:
All images were converted to RGB and resized to 224×224 to ensure uniform input dimensions for the neural network models. This produced the resized_lfw and resized_vgg directories.

Face Alignment:
Using MTCNN, I performed face detection and alignment on both datasets. The aligned images contain normalized, centered faces, which improves model performance and reduces variability caused by pose differences. The aligned outputs were stored in aligned_lfw and aligned_vgg.

Data Augmentation:
Augmentation was applied only to the VGGFace2 training set. This enhances the variety of training samples and reduces overfitting. The test sets and the LFW dataset were not augmented because they are used for evaluation. The augmented images were saved in the augmented_vgg directory.

3. Folder Structure Produced

The preprocessing pipeline generated structured directories that include raw data, resized images, aligned faces, and augmented training samples. This structure ensures that the data is clean, organized, and ready for efficient training and evaluation in the next project stages.

4. Summary of Work Completed

I collected the required datasets from Kaggle and organized them.
I performed image resizing for both datasets.
I implemented face alignment for both datasets using MTCNN.
I applied data augmentation to the VGGFace2 training set.
I generated all processed data folders needed for training.
I delivered all preprocessing scripts used to create the final dataset.

5. Completion Status

All data acquisition and preprocessing tasks have been fully completed.
The processed dataset is now ready for model training, model comparison, and the remaining project components.

 **important_Note**.
Raw datasets are not included in the repository due to size limitations.

To reproduce the preprocessing pipeline, download the datasets using the following drive link

https://drive.google.com/drive/folders/17pN9OrDXG-s7PUlv3tDM7k0DJxOenD35?usp=drive_link
 updated
 
