Facial Landmarks

![land1](https://github.com/user-attachments/assets/b0bb2a25-81db-4c1c-b77a-0d4228dc852b)
 
 MediaPipe and Euclidean Distance
 Inter-Pupillary Distance (IPD)
 Feature Standardization Using StandardScaler
 Handling Class Imbalance Using SMOTE
  
 Effectively detects stress with strong precision and recall scores
 Maintains a balanced classification performance
 Achieves high interpretability through feature importance ranking
 Exhibits good generalization (AUC = 0.72)
 Prioritizes critical facial areas such as the mouth and eyes for stress detection

  The model achieved an Area Under the Curve (AUC) score of 0.72

  ![72](https://github.com/user-attachments/assets/a502c8d5-cdcb-4769-ac1c-e238c269ad40)

  True Positives (Stress correctly identified): 835
  True Negatives (Non-Stress correctly identified): 633
  False Positives (Non-Stress misclassified as Stress): 251
  False Negatives (Stress misclassified as Non-Stress): 199

  ![Matrix](https://github.com/user-attachments/assets/5c077d9f-ae5a-483f-8f82-3efddd79d15d)

 Feature Importance Analysis

  Feature 11 (distance between landmarks (55, 62) — representing the upper to inner lip region) had the highest impact on the model’s predictions.
  This pair is significant because lip tension, trembling, or compression are strong visual cues associated with stress.
  Other highly ranked features included those involving the eyes and mouth width, further confirming their relevance in stress detection.

  ![pic4](https://github.com/user-attachments/assets/bff2d213-9373-4511-8ec2-37c4690a7b74)

  Web Application

  ![pic 2](https://github.com/user-attachments/assets/40aa4e33-84ec-495f-a78b-f156e4bd961a)

  ![Screenshot 2025-04-26 193742](https://github.com/user-attachments/assets/fdb8ff09-2b81-43d9-bcf9-73e57ece9156)


  

